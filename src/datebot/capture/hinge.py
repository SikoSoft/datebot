"""Parsing of Hinge phone traffic (seen through a mitmproxy on this Mac) into datebot profiles and swipes.

Format verified from real captured traffic (Nov 2026), all on prod-api.hingeaws.net, JSON, no certificate pinning:
  * POST /rec/v2              the feed: ids + rating tokens only. Not used; the profile calls below carry the data.
  * POST /user/v3/public      request {ids, viewToken}; response [{userId, profile: {age, firstName, height (cm),
                              location.name, works, jobTitle, educations, hometown, ...}}].
  * POST /content/v2/public   same ids; response [{userId, content: {photos: [{url, ...}], answers: [{position,
                              questionId, type, response}]}}]. Photos are in display order.
  * POST /prompts             {prompts: [{id, prompt}]}: question text for the answers' questionId.
  * POST /rate/v2/initiate    your swipe: {subjectId, rating}. rating "note" = a like (with a comment; each one
                              used up a daily like) and "skip" = a pass, both confirmed on the user's own session.
The two profile responses arrive separately, so they are joined per userId (`HingeSession`).
Other rating values (a like without a comment, roses, ...) have not been seen, so they are NOT guessed: they are
ignored and counted in `HingeSession.unknown_ratings` so they can be checked against fresh traffic.
Numeric profile fields (drinking, smoking, dating intention, ...) are unmapped ids and are left out.
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from datebot.capture.tinder import CapturedProfile, Swipe

HOST = "prod-api.hingeaws.net"
PATH_PROMPTS = "/prompts"
PATH_USER = "/user/v3/public"
PATH_CONTENT = "/content/v2/public"
PATH_RATE = "/rate/v2/initiate"
RATING_LIKE, RATING_PASS = "note", "skip"
MAX_PHOTOS = 6
MAX_PENDING = 500  # half-assembled profiles kept while waiting for their other response


def _loads(text):
    try:
        return json.loads(text) if isinstance(text, (str, bytes)) else text
    except ValueError:
        return None


def _text(v) -> str | None:
    return v.strip() if isinstance(v, str) and v.strip() else None


def parse_prompts(payload) -> dict[str, str]:
    """POST /prompts response -> {questionId: question text}."""
    payload = _loads(payload)
    items = payload.get("prompts") if isinstance(payload, dict) else None
    return {p["id"]: t for p in items or [] if isinstance(p, dict) and isinstance(p.get("id"), str)
            and (t := _text(p.get("prompt")))}


def _items(payload, key: str) -> dict[str, dict]:
    payload = _loads(payload)
    out = {}
    for item in payload if isinstance(payload, list) else []:
        if isinstance(item, dict) and item.get("userId") and isinstance(item.get(key), dict):
            out[str(item["userId"])] = item[key]
    return out


def parse_users(payload) -> dict[str, dict]:
    """POST /user/v3/public response -> {userId: profile dict}."""
    return _items(payload, "profile")


def parse_contents(payload) -> dict[str, dict]:
    """POST /content/v2/public response -> {userId: content dict}."""
    return _items(payload, "content")


def build_profile(user_id: str, profile: dict, content: dict, prompts: dict[str, str]) -> CapturedProfile:
    photos = [p["url"] for p in content.get("photos") or [] if isinstance(p, dict) and _text(p.get("url"))]
    answers = [a for a in content.get("answers") or [] if isinstance(a, dict) and a.get("type") == "text"
               and _text(a.get("response"))]
    lines = []
    for a in sorted(answers, key=lambda a: a.get("position") if isinstance(a.get("position"), int) else 0):
        question = prompts.get(a.get("questionId"))
        lines.append(f"{question}: {a['response'].strip()}" if question else a["response"].strip())
    descriptors = []
    for label, value in (("Job title", _text(profile.get("jobTitle"))), ("Work", _text(profile.get("works"))),
                         ("Hometown", _text(profile.get("hometown")))):
        if value:
            descriptors.append(f"{label}: {value}")
    schools = [s for s in map(_text, profile.get("educations") or []) if s]
    if schools:
        descriptors.append("Education: " + ", ".join(schools))
    location = profile.get("location")
    height = profile.get("height")
    age = profile.get("age")
    extra = {
        "city": _text(location.get("name")) if isinstance(location, dict) else None,
        "prompts": lines,
        "descriptors": descriptors,
        # Height is a rule input, not taste: kept out of the text descriptors so it is never embedded.
        "height_cm": height if isinstance(height, int) and height > 0 else None,
    }
    return CapturedProfile(
        external_id=user_id,
        name=_text(profile.get("firstName")),
        age=age if isinstance(age, int) and age > 0 else None,
        bio=None,  # Hinge has no free-text bio; its prompts go in extra["prompts"]
        photo_urls=photos[:MAX_PHOTOS],
        extra={k: v for k, v in extra.items() if v not in (None, [], "")},
    )


@dataclass
class Event:
    kind: str  # "profile" or "swipe"
    profile: CapturedProfile | None = None
    swipe: Swipe | None = None
    at: float | None = None  # epoch seconds, stamped by the live capture (a swipe's is when the request started)


class HingeSession:
    """Feed it Hinge API exchanges in order; it yields profiles (once both halves arrived) and your swipes."""

    def __init__(self, prompts: dict[str, str] | None = None):
        self.prompts: dict[str, str] = dict(prompts or {})
        self.prompts_changed = False
        self.unknown_ratings: Counter = Counter()
        self._users: dict[str, dict] = {}
        self._contents: dict[str, dict] = {}

    def handle(self, path: str, status: int, request_body, response_body) -> list[Event]:
        if status != 200:
            return []
        if path == PATH_PROMPTS:
            found = parse_prompts(response_body)
            if found:
                self.prompts.update(found)
                self.prompts_changed = True
            return []
        if path == PATH_USER:
            self._users.update(parse_users(response_body))
        elif path == PATH_CONTENT:
            self._contents.update(parse_contents(response_body))
        elif path == PATH_RATE:
            return self._swipe(request_body)
        else:
            return []
        events = []
        for uid in [u for u in self._users if u in self._contents]:
            profile = build_profile(uid, self._users.pop(uid), self._contents.pop(uid), self.prompts)
            events.append(Event("profile", profile=profile))
        for pending in (self._users, self._contents):
            while len(pending) > MAX_PENDING:
                pending.pop(next(iter(pending)))
        return events

    def _swipe(self, request_body) -> list[Event]:
        req = _loads(request_body)
        if not isinstance(req, dict) or not req.get("subjectId"):
            return []
        rating = req.get("rating")
        if rating not in (RATING_LIKE, RATING_PASS):
            self.unknown_ratings[str(rating)] += 1
            return []
        return [Event("swipe", swipe=Swipe(str(req["subjectId"]), rating == RATING_LIKE))]


def replay(dump_dir: Path, session: HingeSession | None = None) -> Iterable[Event]:
    """Run a hinge_probe dump (numbered JSON exchanges) back through the parser, in capture order."""
    session = session or HingeSession()
    for f in sorted(Path(dump_dir).glob("[0-9]*.json")):
        x = json.loads(f.read_text())
        url = x.get("url", "")
        if f"//{HOST}/" not in url:
            continue
        path = url.split(HOST, 1)[1].split("?")[0]
        yield from session.handle(path, x.get("status", 0), x.get("request_body"), x.get("response_body"))
