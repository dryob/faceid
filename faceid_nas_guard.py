#!/usr/bin/env python3
"""NAS-local guard for the containerised FaceID deployment.

The durable-write chain is: app -> writer bridge (same host, local disk) ->
NAS faceid-data.  This guard fails closed unless the writer bridge
acknowledges a probe and the mounted data/model roots carry the NAS markers.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

WRITER_URL = os.environ.get("FACEID_NAS_WRITER_URL", "").strip().rstrip("/")
WRITER_TOKEN = os.environ.get("FACEID_NAS_WRITER_TOKEN", "")
DATA = Path(os.environ.get("FACEID_NAS_WRITER_ROOT", "/app/data"))
MODEL = Path(os.environ.get("FACEID_MODEL", "/app/model"))
MARKER = b"nas-root\n"


def _mount_id(fd: int):
    try:
        with open(f"/proc/self/fdinfo/{fd}", encoding="ascii") as stream:
            for line in stream:
                if line.startswith("mnt_id:"):
                    return int(line.split(":", 1)[1].strip())
    except (OSError, ValueError):
        return None
    return None


def _writer(op: str, **payload) -> None:
    if not WRITER_URL:
        raise RuntimeError("FACEID_NAS_WRITER_URL is required")
    if not WRITER_TOKEN:
        raise RuntimeError("FACEID_NAS_WRITER_TOKEN is required")
    body = json.dumps({"op": op, **payload}, separators=(",", ":")).encode("utf-8")
    request = urllib.request.Request(
        WRITER_URL, data=body, method="POST",
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {WRITER_TOKEN}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            if response.status != 204:
                raise RuntimeError(f"NAS writer returned HTTP {response.status}")
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
        raise RuntimeError("NAS writer unavailable; refusing local fallback") from exc


def verify(*, data: Path | None = None, model: Path | None = None) -> dict:
    """Fail closed unless the durable write chain and mounted roots are intact."""
    _writer("mkdir", path="runtime-probes")  # idempotent end-to-end probe
    for root in (DATA, MODEL):
        if not root.is_dir():
            raise RuntimeError(f"NAS-backed path missing: {root}")
        marker = root / ".camera-face-nas-root"
        try:
            if marker.read_bytes() != MARKER:
                raise RuntimeError(f"NAS marker mismatch: {marker}")
        except OSError as exc:
            raise RuntimeError(f"NAS marker unreadable: {marker}") from exc
    if data is not None:
        d = Path(data).resolve()
        if d != DATA.resolve() and DATA.resolve() not in d.parents:
            raise RuntimeError(f"data path outside NAS data root: {d}")
    if model is not None:
        m = Path(model).resolve()
        if m != MODEL.resolve() and MODEL.resolve() not in m.parents:
            raise RuntimeError(f"model path outside NAS model root: {m}")
    fd = os.open(DATA, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        st = os.fstat(fd)
        name = f".camera-face-nas-guard-probe.{os.getpid()}.{time.time_ns()}"
        probe_fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600, dir_fd=fd)
        try:
            os.write(probe_fd, b"ok\n")
            os.fsync(probe_fd)
        finally:
            os.close(probe_fd)
        read_fd = os.open(name, os.O_RDONLY, dir_fd=fd)
        try:
            if os.read(read_fd, 16) != b"ok\n":
                raise RuntimeError("NAS guard read-back mismatch")
        finally:
            os.close(read_fd)
        os.unlink(name, dir_fd=fd)
        return {"st_dev": int(st.st_dev), "st_ino": int(st.st_ino), "mnt_id": _mount_id(fd)}
    finally:
        os.close(fd)


if __name__ == "__main__":
    verify()
    print("NAS verification passed")
