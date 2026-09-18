"""Mount-pinned durable writes for NAS-backed FaceID data.

A successful path check is not enough: an SSHFS disconnect can expose the same
pathname as a local directory.  These helpers open the verified parent directory
first and perform all subsequent operations relative to that descriptor.  An
open directory descriptor remains pinned to the verified mount across a later
mount replacement, so a disconnect cannot redirect the write to the host path.
"""
from __future__ import annotations

import os
import time
import errno
import base64
import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable

import stat as stat_mod


def _writer_url() -> str | None:
    """Return the host NAS writer endpoint when the two-hop deployment uses it."""
    value = os.environ.get("FACEID_NAS_WRITER_URL", "").strip().rstrip("/")
    return value or None


def _writer_request(payload: dict) -> None:
    url = _writer_url()
    if not url:
        return
    token = os.environ.get("FACEID_NAS_WRITER_TOKEN", "")
    if not token:
        raise OSError(errno.EACCES, "FACEID_NAS_WRITER_TOKEN is required")
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    request = urllib.request.Request(
        url, data=body, method="POST",
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            if response.status != 204:
                raise OSError(errno.EIO, f"NAS writer returned HTTP {response.status}")
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
        raise OSError(errno.ENODEV, "NAS writer unavailable; refusing local fallback") from exc


def _writer_path(path: Path) -> str:
    value = os.environ.get("FACEID_NAS_WRITER_ROOT", "").strip()
    if not value:
        raise OSError(errno.EINVAL, "FACEID_NAS_WRITER_ROOT is required")
    root = Path(value)
    try:
        relative = str(Path(path).resolve().relative_to(root.resolve()))
    except ValueError as exc:
        raise OSError(errno.EXDEV, f"path is outside NAS writer root: {path}") from exc
    # mkdir of the writer root itself is a no-op; do not send an empty relative path.
    return relative


def _remote_mkdir(path: Path) -> bool:
    if not _writer_url():
        return False
    relative = _writer_path(path)
    if not relative or relative == ".":
        return True
    _writer_request({"op": "mkdir", "path": relative})
    return True


def _remote_write(path: Path, data: bytes) -> bool:
    if not _writer_url():
        return False
    _writer_request({"op": "write", "path": _writer_path(path),
                     "content": base64.b64encode(data).decode("ascii")})
    return True


def _remote_unlink(path: Path) -> bool:
    if not _writer_url():
        return False
    _writer_request({"op": "unlink", "path": _writer_path(path)})
    return True


def _remote_rmdir(path: Path) -> bool:
    if not _writer_url():
        return False
    _writer_request({"op": "rmdir", "path": _writer_path(path)})
    return True


def _remote_rename(source: Path, destination: Path) -> bool:
    if not _writer_url():
        return False
    _writer_request({"op": "rename", "source": _writer_path(source),
                     "destination": _writer_path(destination)})
    return True


def _fd_mount_id(fd: int) -> int | None:
    """Return Linux's mount-instance id for an already-open descriptor."""
    try:
        with open(f"/proc/self/fdinfo/{fd}", encoding="ascii") as stream:
            for line in stream:
                if line.startswith("mnt_id:"):
                    return int(line.split(":", 1)[1].strip())
    except (OSError, ValueError):
        return None
    return None


def _require_mount_id(fd: int) -> int:
    mount_id = _fd_mount_id(fd)
    if mount_id is None:
        raise OSError(errno.EXDEV, "unreadable NAS mount id")
    return mount_id


def _identity(path: Path) -> tuple[int, int, int]:
    """Return device/inode/mount identity of a directory."""
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        st = os.fstat(fd)
        return int(st.st_dev), int(st.st_ino), _require_mount_id(fd)
    finally:
        os.close(fd)


def _open_parent(path: Path, guard: Callable | None = None) -> tuple[int, str]:
    path = Path(path)
    parent = path.parent
    if not parent.is_dir():
        raise FileNotFoundError(parent)
    # Open first: the directory FD pins the inode before attestation.  A
    # pathname check followed by open has a check/open race on mount loss.
    fd = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        opened = os.fstat(fd)
        opened_mnt = _require_mount_id(fd)
        opened_identity = (int(opened.st_dev), int(opened.st_ino), opened_mnt)
        attested = guard(data=parent) if guard is not None else None
        if isinstance(attested, dict):
            expected_mnt = attested.get("mnt_id")
            expected_dev = attested.get("st_dev")
            if expected_mnt is None:
                raise OSError(errno.EXDEV, "unreadable NAS mount id")
            if opened_mnt != int(expected_mnt) or (
                expected_dev is not None and int(opened.st_dev) != int(expected_dev)
            ):
                raise OSError(errno.EXDEV, "fd is not on attested NAS mount")
        current = _identity(parent)
        if opened_identity != current:
            raise OSError(errno.EXDEV,
                          "verified NAS directory changed (or mount changed) during attestation")
        # Re-attest after the identity comparison as well.  All mutations below
        # remain relative to this already-open descriptor.
        if guard is not None:
            attested = guard(data=parent)
            if isinstance(attested, dict):
                expected_mnt = attested.get("mnt_id")
                if expected_mnt is None or int(expected_mnt) != opened_mnt:
                    raise OSError(errno.EXDEV, "fd is not on attested NAS mount")
        current = _identity(parent)
        if opened_identity != current:
            raise OSError(errno.EXDEV,
                          "verified NAS directory changed (or mount changed) after attestation")
        return fd, path.name
    except Exception:
        os.close(fd)
        raise


def atomic_write_bytes_pinned(path: Path, data: bytes, guard: Callable | None = None) -> None:
    """Write bytes and replace ``path`` using a verified parent directory FD."""
    if _remote_write(Path(path), data):
        return
    fd, name = _open_parent(Path(path), guard)
    temporary = f".{name}.{os.getpid()}.{time.time_ns()}.partial"
    tmp_fd = None
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        tmp_fd = os.open(temporary, flags, 0o600, dir_fd=fd)
        try:
            with os.fdopen(tmp_fd, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            tmp_fd = None
        except Exception:
            tmp_fd = None
            raise
        os.replace(temporary, name, src_dir_fd=fd, dst_dir_fd=fd)
    finally:
        if tmp_fd is not None:
            os.close(tmp_fd)
        try:
            os.unlink(temporary, dir_fd=fd)
        except FileNotFoundError:
            pass
        os.close(fd)


def atomic_write_text_pinned(path: Path, text: str, guard: Callable | None = None) -> None:
    atomic_write_bytes_pinned(Path(path), text.encode("utf-8"), guard)


def unlink_pinned(path: Path, guard: Callable | None = None) -> None:
    """Unlink a file relative to a verified parent directory descriptor."""
    if _remote_unlink(Path(path)):
        return
    fd, name = _open_parent(Path(path), guard)
    try:
        os.unlink(name, dir_fd=fd)
    except FileNotFoundError:
        pass
    finally:
        os.close(fd)


def rename_pinned(source: Path, destination: Path, guard: Callable | None = None) -> None:
    """Rename between directories while both verified parent FDs are open."""
    source, destination = Path(source), Path(destination)
    if _remote_rename(source, destination):
        return
    src_fd, src_name = _open_parent(source, guard)
    try:
        dst_fd, dst_name = _open_parent(destination, guard)
        try:
            os.rename(src_name, dst_name, src_dir_fd=src_fd, dst_dir_fd=dst_fd)
        finally:
            os.close(dst_fd)
    finally:
        os.close(src_fd)


def mkdir_pinned(path: Path, guard: Callable | None = None) -> None:
    """Create a directory relative to a verified parent descriptor."""
    path = Path(path)
    if path == path.parent:
        return
    if _writer_url():
        _remote_mkdir(path)
        return
    try:
        fd, name = _open_parent(path, guard)
    except FileNotFoundError:
        mkdir_pinned(path.parent, guard)
        fd, name = _open_parent(path, guard)
    try:
        try:
            os.mkdir(name, 0o700, dir_fd=fd)
        except FileExistsError:
            mode = os.stat(name, dir_fd=fd).st_mode
            if not stat_mod.S_ISDIR(mode):
                raise
    finally:
        os.close(fd)


def copy_file_pinned(source: Path, destination: Path,
                     guard: Callable | None = None) -> None:
    """Copy one durable file without opening the destination by pathname."""
    atomic_write_bytes_pinned(Path(destination), Path(source).read_bytes(), guard)


def rmdir_pinned(path: Path, guard: Callable | None = None) -> None:
    """Remove an empty directory relative to its verified parent FD."""
    path = Path(path)
    if _remote_rmdir(path):
        return
    fd, name = _open_parent(path, guard)
    try:
        os.rmdir(name, dir_fd=fd)
    finally:
        os.close(fd)


def remove_tree_pinned(path: Path, guard: Callable | None = None) -> None:
    """Recursively remove a tree using pinned file and directory operations."""
    path = Path(path)
    if not path.exists():
        return
    for child in list(path.iterdir()):
        if child.is_dir() and not child.is_symlink():
            remove_tree_pinned(child, guard)
        else:
            unlink_pinned(child, guard)
    rmdir_pinned(path, guard)


def copy_tree_pinned(source: Path, destination: Path,
                     guard: Callable | None = None) -> None:
    """Copy a tree while routing every destination mutation through pinned I/O."""
    source, destination = Path(source), Path(destination)
    mkdir_pinned(destination, guard)
    for child in source.iterdir():
        target = destination / child.name
        if child.is_dir() and not child.is_symlink():
            copy_tree_pinned(child, target, guard)
        else:
            copy_file_pinned(child, target, guard)
