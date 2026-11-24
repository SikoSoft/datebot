from __future__ import annotations

import hashlib
import shutil
import sqlite3
from pathlib import Path

from datebot import config, db

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".heic"}


def store_photo(src: Path) -> str:
    """Copy a photo into the data dir under a content hash, so the library is self-contained and deduped."""
    digest = hashlib.sha256(src.read_bytes()).hexdigest()[:20]
    dest = config.photos_dir() / f"{digest}{src.suffix.lower()}"
    if not dest.exists():
        shutil.copy2(src, dest)
    return str(dest)


def images_in(directory: Path) -> list[Path]:
    return sorted(p for p in directory.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def add_from_files(conn: sqlite3.Connection, files: list[Path], source: str = "manual", **fields) -> int:
    return db.add_profile(conn, source, [store_photo(f) for f in files], **fields)


def add_from_folder(conn: sqlite3.Connection, root: Path, source: str = "manual") -> list[int]:
    """Each subfolder of `root` is one profile (its images are the photos). Loose images in `root` are
    one profile each."""
    ids = []
    for sub in sorted(p for p in root.iterdir() if p.is_dir()):
        files = images_in(sub)
        if files:
            ids.append(add_from_files(conn, files, source, external_id=f"folder:{sub.name}", name=sub.name))
    for img in images_in(root):
        ids.append(add_from_files(conn, [img], source, external_id=f"file:{img.name}"))
    return ids
