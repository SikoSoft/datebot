"""Snapshots of everything datebot knows, so nothing is ever lost.

What is what (see the readme):
- ``datebot.db``  the irreplaceable part: profiles, photo records and their embeddings, every yes/no
  (``ratings``), your captured swipes, behaviour interactions. Copied with SQLite's online backup, so it is
  consistent even while the UI or a capture is running.
- ``photos/``     the image files (content-named, never modified). Large; skippable with ``photos=False``.
- ``filters.json``, ``models/`` (incl. ``models/behavior``)  settings and trained models. Regenerable from
  the db, but kept so a restore is instantly usable.
- ``cache/``      CLIP text-prompt embeddings. Regenerable; kept because it is tiny.
Never included: ``backups/`` itself, ``browser/`` (a Chrome profile with logins) and raw capture dumps.
"""
from __future__ import annotations

import shutil
import sqlite3
from datetime import datetime
from pathlib import Path

from datebot import config

DB_NAME = "datebot.db"
SMALL = ("filters.json", "models", "cache")  # always included


def backups_dir() -> Path:
    path = config.data_dir() / "backups"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _copy(src: Path, dst: Path) -> None:
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)
    elif src.exists():
        shutil.copy2(src, dst)


def _copy_db(src: Path, dst: Path) -> None:
    """Consistent copy of a live SQLite db (plain file copy can catch it mid-write)."""
    source, target = sqlite3.connect(src), sqlite3.connect(dst)
    try:
        source.backup(target)
    finally:
        target.close()
        source.close()


def create(dest: Path | None = None, *, photos: bool = True, label: str = "") -> Path:
    """Write a snapshot and return its folder. Default: data/backups/<timestamp>[-label]."""
    data = config.data_dir()
    if dest is None:
        dest = backups_dir() / (datetime.now().strftime("%Y%m%d-%H%M%S") + (f"-{label}" if label else ""))
    if dest.exists() and any(dest.iterdir()):
        raise FileExistsError(f"{dest} already exists and is not empty")
    dest.mkdir(parents=True, exist_ok=True)
    if (data / DB_NAME).exists():
        _copy_db(data / DB_NAME, dest / DB_NAME)
    for name in SMALL:
        _copy(data / name, dest / name)
    if photos:
        _copy(data / "photos", dest / "photos")
    return dest


def verify(snapshot: Path) -> dict[str, int]:
    """Open a snapshot's db and return row counts; raises if it is missing or corrupt."""
    path = snapshot / DB_NAME
    if not path.exists():
        raise FileNotFoundError(f"{snapshot} has no {DB_NAME}")
    conn = sqlite3.connect(path)
    try:
        if conn.execute("pragma integrity_check").fetchone()[0] != "ok":
            raise ValueError(f"{path} failed the integrity check")
        counts = {t: conn.execute(f"select count(*) from {t}").fetchone()[0] for t in ("profiles", "ratings")}
    finally:
        conn.close()
    counts["photo_files"] = sum(1 for _ in (snapshot / "photos").glob("*")) if (snapshot / "photos").exists() else 0
    return counts


def list_snapshots() -> list[Path]:
    return sorted(p for p in backups_dir().iterdir() if (p / DB_NAME).exists())


def restore(snapshot: Path) -> Path:
    """Put a snapshot back as the live data. First takes a photo-less safety snapshot of what is there now
    (returned), so a restore can itself be undone."""
    verify(snapshot)
    safety = create(label="before-restore", photos=False)
    data = config.data_dir()
    _copy_db(snapshot / DB_NAME, data / DB_NAME)
    for name in SMALL:
        if (snapshot / name).exists():
            _copy(snapshot / name, data / name)
    if (snapshot / "photos").exists():
        _copy(snapshot / "photos", data / "photos")  # adds missing files, never deletes
    return safety
