"""FaceID — Gesichtserkennung für Frigate/HA. Start: python -m app.main"""
import json
import logging
import os
import threading
import time
from pathlib import Path

import uvicorn
import yaml

from . import logbuffer
from .engine import FaceEngine
from .frigate_api import frigate_client
from .gallery import Gallery
from .history import History
from .mqtt_listener import EventProcessor
from .webui import build_app
from .backup_util import start_auto_backup

BASE = Path(__file__).resolve().parent.parent


def _start_nas_watchdog(guard_module):
    """Exit the service if the durable NAS chain disappears after startup."""
    def watch():
        while True:
            time.sleep(5)
            try:
                guard_module.verify()
            except Exception:
                log.exception("NAS guard failed at runtime; stopping before local fallback")
                os._exit(74)
    thread = threading.Thread(target=watch, name="nas-guard", daemon=True)
    thread.start()


def _start_retention_worker(gallery, cfg):
    """Keep provisional and inactive derived data within configured TTLs."""
    interval = max(60.0, float(cfg["faceid"].get("retention_check_seconds", 3600)))
    unknown_ttl = float(cfg["faceid"].get("unknown_retention_seconds", 7 * 86400))
    person_ttl = float(cfg["faceid"].get("person_retention_seconds", 90 * 86400))
    promoted_ttl = float(cfg["faceid"].get("promoted_retention_seconds", 90 * 86400))

    def clean():
        while True:
            time.sleep(interval)
            try:
                gallery.expire_unknowns(ttl_seconds=unknown_ttl,
                                        promoted_ttl_seconds=promoted_ttl)
                gallery.expire_persons(ttl_seconds=person_ttl)
            except Exception:
                log.exception("retention sweep failed")

    threading.Thread(target=clean, name="faceid-retention", daemon=True).start()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logbuffer.install()   # damit die Weboberflaeche das Log zeigen kann
log = logging.getLogger("faceid")


def main():
    cfg = yaml.safe_load((BASE / "config.yaml").read_text())
    data_dir = BASE / "data"
    # Production launcher must verify the SSHFS -> host CIFS NAS chain before
    # opening any persistent gallery/history files. Local fallback is unsafe.
    guard = BASE / "faceid_nas_guard.py"
    if not guard.is_file():
        raise RuntimeError("NAS guard is required; refusing local fallback")
    import importlib.util
    spec = importlib.util.spec_from_file_location("faceid_nas_guard", guard)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load NAS guard")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.verify()
    _start_nas_watchdog(module)
    # Live-editierbare Einstellungen (Settings-Tab) liegen als Overlay in data/settings.json
    # und gewinnen über config.yaml — persistent auch beim Add-on (config.yaml wird dort
    # bei jedem Start neu generiert, /data überlebt).
    settings_f = data_dir / "settings.json"
    if settings_f.exists():
        try:
            cfg.setdefault("faceid", {}).update(json.loads(settings_f.read_text()))
        except (json.JSONDecodeError, OSError):
            log.warning("settings.json unreadable — ignoring it")
    log.info("loading InsightFace (buffalo_l) …")
    engine = FaceEngine(det_size=int(cfg["faceid"].get("det_size", 640)))
    guarded_write = lambda **kwargs: module.verify(**kwargs)
    gallery = Gallery(data_dir,
                      top_k=int(cfg["faceid"].get("match_top_k", 3)),
                      max_per_person=int(cfg["faceid"].get("max_faces_per_person", 40)),
                      storage_guard=guarded_write)
    gallery.expire_unknowns(
        ttl_seconds=float(cfg["faceid"].get("unknown_retention_seconds", 7 * 86400)),
        promoted_ttl_seconds=float(cfg["faceid"].get("promoted_retention_seconds", 90 * 86400)),
    )
    gallery.expire_persons(ttl_seconds=float(cfg["faceid"].get("person_retention_seconds", 90 * 86400)))
    _start_retention_worker(gallery, cfg)
    gallery.trimmed_keep = int(cfg["faceid"].get("trimmed_keep", 10))
    gallery.max_ignore_anchors = int(cfg["faceid"].get("max_ignore_anchors", 0))
    gallery.dedupe_threshold = float(cfg["faceid"].get("dedupe_threshold", 0.65))
    # Riskante Referenzen: relativ zur Entscheidungsgrenze, damit die Grenze mitwandert,
    # wenn jemand match_threshold aendert. 0 (oder margin < 0) schaltet es ab.
    _margin = float(cfg["faceid"].get("cross_risk_margin", 0.05))
    gallery.cross_risk_threshold = (
        max(0.0, float(cfg["faceid"].get("match_threshold", 0.5)) - _margin)
        if _margin >= 0 else 0.0)
    _ratio = float(cfg["faceid"].get("self_outlier_ratio", 0.25))
    gallery.self_outlier_ratio = _ratio if _ratio > 0 else 0.0
    # Beim Start einmal durchsehen. Referenzen altern nicht, aber die Pruefungen werden
    # besser — und wer eine Fehlerkennung hatte, soll die Ursache nicht selbst suchen
    # muessen. Es wird nichts geloescht, nur beiseitegelegt und protokolliert.
    if gallery.self_outlier_ratio > 0 or gallery.cross_risk_threshold > 0:
        for r in gallery.quality_scan(gallery.cross_risk_threshold):
            log.info("reference check: %s / %s set aside (%s)", r["person"], r["file"],
                     f"too close to {r['partner']}, {r['similarity']}" if r["kind"] == "cross"
                     else f"{r['mean_sim']} where {r['median']} is normal for this person")
    frigate = frigate_client(cfg)
    # Verlauf der Meldungen mit dem tatsaechlich benutzten Ausschnitt (0 = aus)
    history = History(data_dir, keep=int(cfg["faceid"].get("history_keep", 200)),
                      storage_guard=guarded_write)
    processor = EventProcessor(cfg, engine, gallery, frigate,
                               state_path=data_dir / "faceid-last-seen.json",
                               storage_guard=guarded_write)
    processor.history = history if history.keep > 0 else None
    processor.start()
    start_auto_backup(cfg["faceid"], data_dir, storage_guard=guarded_write)
    app = build_app(cfg, engine, gallery, processor, data_dir, BASE / "static",
                    storage_guard=guarded_write)
    uvicorn.run(app, host="0.0.0.0", port=int(cfg["faceid"].get("port", 8600)), log_level="warning")


if __name__ == "__main__":
    main()
