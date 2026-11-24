"""Hard deal-breaker filters applied before the model.

A profile that fails a filter is never shown for rating and never scored; the model only has to
learn taste, not rules you already know. Missing data (e.g. no age) never rejects a profile.
"""
from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import asdict, dataclass, field

from datebot import config, db

KM_PER_MILE = 1.609344


@dataclass
class Filters:
    min_age: int | None = None
    max_age: int | None = None
    max_distance_km: float | None = None
    blocked_keywords: list[str] = field(default_factory=list)   # reject if any appears in bio/prompts
    required_keywords: list[str] = field(default_factory=list)  # reject unless at least one appears

    def is_empty(self) -> bool:
        return self == Filters()


def _path():
    return config.data_dir() / "filters.json"


def load() -> Filters:
    path = _path()
    if not path.exists():
        return Filters()
    raw = json.loads(path.read_text())
    return Filters(**{k: v for k, v in raw.items() if k in Filters.__dataclass_fields__})


def save(filters: Filters) -> None:
    _path().write_text(json.dumps(asdict(filters), indent=2))


def searchable_text(profile: db.Profile) -> str:
    return db.profile_text(profile).lower()


def _has(text: str, keyword: str) -> bool:
    # Hyphens count as part of a word so "smoker" does not match "non-smoker".
    return re.search(rf"(?<![\w-]){re.escape(keyword.strip().lower())}(?![\w-])", text) is not None


def distance_km(profile: db.Profile) -> float | None:
    if profile.extra.get("distance_km") is not None:
        return float(profile.extra["distance_km"])
    if profile.extra.get("distance_mi") is not None:
        return float(profile.extra["distance_mi"]) * KM_PER_MILE
    return None


def check(profile: db.Profile, f: Filters) -> str | None:
    """Return the reason this profile is rejected, or None if it passes."""
    if profile.age is not None:
        if f.min_age is not None and profile.age < f.min_age:
            return f"age {profile.age} < {f.min_age}"
        if f.max_age is not None and profile.age > f.max_age:
            return f"age {profile.age} > {f.max_age}"
    dist = distance_km(profile)
    if f.max_distance_km is not None and dist is not None and dist > f.max_distance_km:
        return f"{dist:.0f} km away > {f.max_distance_km:g} km"
    text = searchable_text(profile)
    for kw in f.blocked_keywords:
        if kw.strip() and _has(text, kw):
            return f"mentions '{kw}'"
    required = [k for k in f.required_keywords if k.strip()]
    if required and not any(_has(text, k) for k in required):
        return f"mentions none of: {', '.join(required)}"
    return None


def eligible_unrated(conn: sqlite3.Connection, f: Filters | None = None) -> list[int]:
    """Unrated profile ids that pass the filters (defaults to the saved filters)."""
    f = load() if f is None else f
    ids = db.unrated_profile_ids(conn)
    if f.is_empty():
        return ids
    return [pid for pid in ids if check(db.get_profile(conn, pid), f) is None]
