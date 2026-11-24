from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

import numpy as np

from datebot import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS profiles (
    id INTEGER PRIMARY KEY,
    source TEXT NOT NULL,              -- tinder | hinge | badoo | manual
    external_id TEXT,                  -- id within the source app, if known
    name TEXT,
    age INTEGER,
    bio TEXT,
    extra TEXT NOT NULL DEFAULT '{}',  -- JSON: prompts, distance, etc.
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (source, external_id)
);

CREATE TABLE IF NOT EXISTS photos (
    id INTEGER PRIMARY KEY,
    profile_id INTEGER NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
    path TEXT NOT NULL,
    position INTEGER NOT NULL DEFAULT 0,
    embedding BLOB,                    -- float32 vector
    embed_model TEXT
);

CREATE TABLE IF NOT EXISTS ratings (
    id INTEGER PRIMARY KEY,
    profile_id INTEGER NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
    liked INTEGER NOT NULL,            -- 1 = would swipe right, 0 = pass
    origin TEXT NOT NULL DEFAULT 'manual',  -- manual | app (captured swipe)
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_photos_profile ON photos(profile_id);
CREATE INDEX IF NOT EXISTS idx_ratings_profile ON ratings(profile_id);
"""


@dataclass
class Profile:
    id: int
    source: str
    external_id: str | None
    name: str | None
    age: int | None
    bio: str | None
    extra: dict


def connect(path: str | None = None) -> sqlite3.Connection:
    conn = sqlite3.connect(path or str(config.db_path()))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    return conn


@contextmanager
def session(path: str | None = None) -> Iterator[sqlite3.Connection]:
    conn = connect(path)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _profile(row: sqlite3.Row) -> Profile:
    return Profile(
        id=row["id"],
        source=row["source"],
        external_id=row["external_id"],
        name=row["name"],
        age=row["age"],
        bio=row["bio"],
        extra=json.loads(row["extra"]),
    )


def add_profile(
    conn: sqlite3.Connection,
    source: str,
    photo_paths: list[str],
    external_id: str | None = None,
    name: str | None = None,
    age: int | None = None,
    bio: str | None = None,
    extra: dict | None = None,
) -> int:
    """Insert a profile with its photos. Re-adding the same (source, external_id) returns the existing id."""
    if external_id is not None:
        row = conn.execute(
            "SELECT id FROM profiles WHERE source = ? AND external_id = ?", (source, external_id)
        ).fetchone()
        if row:
            return row["id"]
    cur = conn.execute(
        "INSERT INTO profiles (source, external_id, name, age, bio, extra) VALUES (?, ?, ?, ?, ?, ?)",
        (source, external_id, name, age, bio, json.dumps(extra or {})),
    )
    pid = cur.lastrowid
    conn.executemany(
        "INSERT INTO photos (profile_id, path, position) VALUES (?, ?, ?)",
        [(pid, p, i) for i, p in enumerate(photo_paths)],
    )
    return pid


def get_profile(conn: sqlite3.Connection, profile_id: int) -> Profile | None:
    row = conn.execute("SELECT * FROM profiles WHERE id = ?", (profile_id,)).fetchone()
    return _profile(row) if row else None


def photo_paths(conn: sqlite3.Connection, profile_id: int) -> list[str]:
    rows = conn.execute("SELECT path FROM photos WHERE profile_id = ? ORDER BY position", (profile_id,))
    return [r["path"] for r in rows]


def add_rating(conn: sqlite3.Connection, profile_id: int, liked: bool, origin: str = "manual") -> None:
    conn.execute(
        "INSERT INTO ratings (profile_id, liked, origin) VALUES (?, ?, ?)", (profile_id, int(liked), origin)
    )


def latest_ratings(conn: sqlite3.Connection) -> dict[int, bool]:
    """Most recent rating per profile (the user may change their mind)."""
    rows = conn.execute(
        "SELECT profile_id, liked FROM ratings WHERE id IN (SELECT MAX(id) FROM ratings GROUP BY profile_id)"
    )
    return {r["profile_id"]: bool(r["liked"]) for r in rows}


def unrated_profile_ids(conn: sqlite3.Connection) -> list[int]:
    rows = conn.execute(
        "SELECT id FROM profiles WHERE id NOT IN (SELECT profile_id FROM ratings) ORDER BY id"
    )
    return [r["id"] for r in rows]


def clear_ratings(conn: sqlite3.Connection) -> int:
    return conn.execute("DELETE FROM ratings").rowcount


def set_embedding(conn: sqlite3.Connection, photo_id: int, vec: np.ndarray, model: str) -> None:
    conn.execute(
        "UPDATE photos SET embedding = ?, embed_model = ? WHERE id = ?",
        (np.asarray(vec, dtype=np.float32).tobytes(), model, photo_id),
    )


def photos_missing_embeddings(conn: sqlite3.Connection, model: str) -> list[tuple[int, str]]:
    rows = conn.execute(
        "SELECT id, path FROM photos WHERE embedding IS NULL OR embed_model != ?", (model,)
    )
    return [(r["id"], r["path"]) for r in rows]


def profile_embeddings(conn: sqlite3.Connection, profile_id: int, model: str) -> np.ndarray:
    """All photo embeddings for a profile as an (n_photos, dim) array."""
    rows = conn.execute(
        "SELECT embedding FROM photos WHERE profile_id = ? AND embed_model = ? AND embedding IS NOT NULL "
        "ORDER BY position",
        (profile_id, model),
    ).fetchall()
    if not rows:
        return np.empty((0, 0), dtype=np.float32)
    return np.stack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
