"""Everything saved about a profile besides its photos and bio, as labelled lines for the rating/review/browse cards.

The point is parity with the app: you should see roughly what the dating app showed you when you swiped.
"""
from __future__ import annotations

from datebot import db, filters


def profile_details(profile: db.Profile) -> list[dict]:
    """[{"label", "lines": [str]}] in display order; sections with nothing saved are left out."""
    x = profile.extra
    out: list[dict] = []

    def add(label: str, lines) -> None:
        lines = [s.strip() for s in lines if isinstance(s, str) and s.strip()]
        if lines:
            out.append({"label": label, "lines": lines})

    where = []
    if isinstance(x.get("city"), str):
        where.append(x["city"])
    dist = filters.distance_km(profile)
    if dist is not None:
        where.append("under 1 km away" if dist < 1 else f"{dist:.0f} km away")
    add("Location", [" · ".join(where)] if where else [])
    height = filters.height_cm(profile)
    add("Height", [f"{height} cm"] if height else [])
    add("Work", db._strings(x.get("jobs")))
    add("Education", db._strings(x.get("schools")))
    add("About", db._strings(x.get("descriptors")))
    add("Prompts", db._strings(x.get("prompts")))
    add("Interests", [", ".join(db._strings(x.get("interests")))])
    return out
