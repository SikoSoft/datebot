"""Parsing of Tinder web API traffic into datebot profiles and swipes.

NOTE: Tinder has no public API. The endpoint paths (/v2/recs/core, /like/{id}, /pass/{id}) come from
community documentation, but the exact response field names below are best-effort and may drift. Every
accessor is defensive (missing fields are skipped, never fatal). Run `datebot capture tinder --dump DIR`
to save raw responses if something stops parsing.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime
from urllib.parse import urlparse

HOSTS = ("gotinder.com", "tinder.com")
RECS_PATH_RE = re.compile(r"^/(?:v\d+/)?recs/core/?$")
SWIPE_PATH_RE = re.compile(r"^/(?:v\d+/)?(like|pass)/([^/]+)(/super)?/?$")


@dataclass
class CapturedProfile:
    external_id: str
    name: str | None = None
    age: int | None = None
    bio: str | None = None
    photo_urls: list[str] = field(default_factory=list)
    extra: dict = field(default_factory=dict)


@dataclass
class Swipe:
    user_id: str
    liked: bool  # likes and super likes are True; passes are False


def is_tinder_url(url: str, hosts: tuple[str, ...] = HOSTS) -> bool:
    host = urlparse(url).hostname or ""
    return any(host == h or host.endswith("." + h) for h in hosts)


def is_recs_url(url: str) -> bool:
    return RECS_PATH_RE.match(urlparse(url).path) is not None


def parse_swipe(url: str) -> Swipe | None:
    """Recognize a like / super like / pass request from its URL."""
    m = SWIPE_PATH_RE.match(urlparse(url).path)
    return Swipe(user_id=m.group(2), liked=m.group(1) == "like") if m else None


def age_from_birth_date(value, today: date | None = None) -> int | None:
    if not isinstance(value, str) or len(value) < 10:
        return None
    try:
        born = datetime.strptime(value[:10], "%Y-%m-%d").date()
    except ValueError:
        return None
    today = today or date.today()
    return today.year - born.year - ((today.month, today.day) < (born.month, born.day))


def _names(items) -> list[str]:
    """Pull display names out of a list of strings or {'name': ...} dicts."""
    out = []
    for it in items if isinstance(items, list) else []:
        name = it if isinstance(it, str) else (it.get("name") if isinstance(it, dict) else None)
        if isinstance(name, str) and name.strip():
            out.append(name.strip())
    return out


def _photo_url(photo) -> str | None:
    """Prefer the widest processed rendition; fall back to the plain url."""
    if not isinstance(photo, dict):
        return None
    files = [f for f in photo.get("processedFiles") or [] if isinstance(f, dict) and f.get("url")]
    if files:
        return max(files, key=lambda f: f.get("width") or 0)["url"]
    return photo.get("url") if isinstance(photo.get("url"), str) else None


def _descriptors(user: dict) -> list[str]:
    out = []
    for d in user.get("selected_descriptors") or []:
        if not isinstance(d, dict) or not d.get("name"):
            continue
        choices = _names(d.get("choice_selections"))
        out.append(f"{d['name']}: {', '.join(choices)}" if choices else str(d["name"]))
    return out


def _profile(item: dict) -> CapturedProfile | None:
    user = item.get("user") if isinstance(item.get("user"), dict) else None
    if not user or not user.get("_id"):
        return None  # ads, boosts and other non-user cards
    photos = [u for u in (_photo_url(p) for p in user.get("photos") or []) if u]
    jobs = []
    for j in user.get("jobs") or []:
        if isinstance(j, dict):
            jobs += _names([j.get("title")]) + _names([j.get("company")])
    extra = {
        "distance_mi": item.get("distance_mi", user.get("distance_mi")),
        "city": (user.get("city") or {}).get("name") if isinstance(user.get("city"), dict) else None,
        "jobs": jobs,
        "schools": _names(user.get("schools")),
        "interests": _names(user.get("user_interests")),
        "descriptors": _descriptors(user),
    }
    return CapturedProfile(
        external_id=str(user["_id"]),
        name=user.get("name") if isinstance(user.get("name"), str) else None,
        age=age_from_birth_date(user.get("birth_date")),
        bio=(user.get("bio") or "").strip() or None,
        photo_urls=photos,
        extra={k: v for k, v in extra.items() if v not in (None, [], "")},
    )


def parse_recs(payload) -> list[CapturedProfile]:
    """Parse a /v2/recs/core response. Returns [] for anything unrecognized."""
    try:
        results = payload["data"]["results"]
    except (KeyError, TypeError):
        return []
    out = []
    for item in results if isinstance(results, list) else []:
        if isinstance(item, dict) and (p := _profile(item)):
            out.append(p)
    return out
