"""Parsing of Badoo web API traffic into datebot profiles and swipes.

Format verified from real captured traffic (Nov 2026): everything goes through POST /mwebapi.phtml?<NAME>
with a JSON envelope {"body": [{"message_type": N, ...}]}.
  * SERVER_GET_ENCOUNTERS: the response body holds message_type 84 -> client_encounters.results[].user.
  * SERVER_ENCOUNTERS_VOTE: the request holds message_type 80 -> server_encounters_vote {person_id, vote}.
UNVERIFIED: the vote codes. 2 = like and 3 = pass is assumed (the numbers match Bumble's documented
protocol); confirm against your own swipes. Any other code is ignored rather than guessed.
Badoo exposes no distance in this feed, only a city, so the distance filter cannot apply.
"""
from __future__ import annotations

import json
from urllib.parse import urlparse

from datebot.capture.tinder import CapturedProfile, Swipe

HOSTS = ("badoo.com",)
API_PATH = "/mwebapi.phtml"
MSG_ENCOUNTERS_RESPONSE = 84
MSG_VOTE_REQUEST = 80
VOTE_LIKE, VOTE_PASS = 2, 3
MAX_PHOTOS = 6

PROMPT_TYPE = 26  # profile_fields of this type are "question -> answer" prompts
SKIP_FIELDS = {"location", "aboutme_text", "languages", "education_level_array"}  # bio is read separately; the rest is noise
LABELLED_FIELDS = {"relationship", "sexuality", "children", "smoking", "drinking", "pet", "belief", "zodiac",
                   "personality", "work", "education", "height"}


def is_badoo_url(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return any(host == h or host.endswith("." + h) for h in HOSTS)


def api_name(url: str) -> str | None:
    """The call name of an API URL, e.g. SERVER_GET_ENCOUNTERS, or None if it is not the API."""
    p = urlparse(url)
    if p.path != API_PATH or not is_badoo_url(url):
        return None
    return p.query.split("&")[0] or None


def _bodies(payload, message_type: int):
    try:
        items = payload["body"]
    except (KeyError, TypeError):
        return
    for item in items if isinstance(items, list) else []:
        if isinstance(item, dict) and item.get("message_type") == message_type:
            yield item


def _loads(text):
    try:
        return json.loads(text) if isinstance(text, (str, bytes)) else text
    except ValueError:
        return None


def _photo_urls(user: dict) -> list[str]:
    urls: list[str] = []
    for album in user.get("albums") or []:
        for photo in (album.get("photos") or []) if isinstance(album, dict) else []:
            url = photo.get("large_url") or photo.get("preview_url") if isinstance(photo, dict) else None
            if isinstance(url, str) and url:
                urls.append("https:" + url if url.startswith("//") else url)
    return urls[:MAX_PHOTOS]


def _text(v) -> str | None:
    return v.strip() if isinstance(v, str) and v.strip() else None


def _profile(item: dict) -> CapturedProfile | None:
    user = item.get("user") if isinstance(item.get("user"), dict) else None
    if not user or not user.get("user_id"):
        return None
    fields = [f for f in user.get("profile_fields") or [] if isinstance(f, dict)]
    bio = next((_text(f.get("display_value")) for f in fields if f.get("id") == "aboutme_text"), None)
    if bio is None:
        bio = next((t for t in map(_text, user.get("displayed_about_me") or []) if t), None)
    prompts, descriptors = [], []
    for f in fields:
        value, label = _text(f.get("display_value")), _text(f.get("name"))
        if not value or f.get("id") in SKIP_FIELDS:
            continue
        if f.get("type") == PROMPT_TYPE:
            prompts.append(f"{label}: {value}" if label else value)
        elif f.get("id") in LABELLED_FIELDS:
            descriptors.append(f"{label}: {value}" if label else value)
    tiw = user.get("tiw_idea")
    if isinstance(tiw, dict) and (phrase := _text(tiw.get("tiw_phrase"))):
        descriptors.append(f"Looking for: {phrase}")
    city = user.get("city")
    extra = {
        "city": _text(city.get("name")) if isinstance(city, dict) else None,
        "interests": [n for n in (_text(i.get("name")) for i in user.get("interests") or [] if isinstance(i, dict)) if n],
        "prompts": prompts,
        "descriptors": descriptors,
    }
    age = user.get("age")
    return CapturedProfile(
        external_id=str(user["user_id"]),
        name=_text(user.get("name")),
        age=age if isinstance(age, int) and age > 0 else None,
        bio=bio,
        photo_urls=_photo_urls(user),
        extra={k: v for k, v in extra.items() if v not in (None, [], "")},
    )


def parse_encounters(payload) -> list[CapturedProfile]:
    """Parse a SERVER_GET_ENCOUNTERS response (dict or JSON text). Returns [] for anything unrecognized."""
    payload = _loads(payload)
    out = []
    for body in _bodies(payload, MSG_ENCOUNTERS_RESPONSE):
        results = (body.get("client_encounters") or {}).get("results")
        for item in results if isinstance(results, list) else []:
            if isinstance(item, dict) and (p := _profile(item)):
                out.append(p)
    return out


def parse_vote(request_body) -> Swipe | None:
    """Parse a SERVER_ENCOUNTERS_VOTE request into your swipe. None for other calls or unknown vote codes."""
    payload = _loads(request_body)
    for body in _bodies(payload, MSG_VOTE_REQUEST):
        vote = body.get("server_encounters_vote")
        if isinstance(vote, dict) and vote.get("person_id") and vote.get("vote") in (VOTE_LIKE, VOTE_PASS):
            return Swipe(str(vote["person_id"]), vote["vote"] == VOTE_LIKE)
    return None
