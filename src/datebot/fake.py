"""Gut-feeling "is this a scam / fake profile?" score, 0-100, so the insights can leave those profiles out.

This is a transparent additive heuristic, not a trained model: each signal adds points and the reasons are kept
so you can audit why a profile was flagged. It only looks at stored profile data (text, interests, photo count)
and, when available, the CLIP visual labels from `attributes.py`. It never touches ratings and never hides a
profile from rating or training: it only decides what the /insights statistics ignore.

Signals come from the usual tells: no or tiny bio, crypto / investing talk, moving the chat to Telegram or
WhatsApp, a single photo, studio-polished or heavily filtered photos, and a pile of stereotypically male-coded
interests on a profile that is supposed to be a woman's.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from datebot import db

# Strong scam tells: any one of these is a big push.
SCAM_RE = re.compile(
    r"\b(crypto\w*|bitcoin|btc|ethereum|forex|binance|nfts?|trad(e|er|ing)s?|invest\w*|stocks?|"
    r"financial freedom|passive income|portfolio|broker\w*|telegram|whatsapp|cash ?app|onlyfans|"
    r"link in bio|sugar (daddy|baby)|verify|kryptovaluta\w*|aktier?|investera\w*)\b", re.IGNORECASE)
HANDLE_RE = re.compile(r"@\w{3,}|\b(insta(gram)?|snap(chat)?|tiktok)\b", re.IGNORECASE)
# Male-coded interests; weak on their own, a cluster of them is suspicious.
MALE_RE = re.compile(
    r"\b(cars?|motorcycles?|motorbikes?|fishing|hunting|whisk(e)?y|cigars?|poker|gaming|esports?|football|soccer|"
    r"mma|boxing|crossfit|bodybuilding|lifting|gym|grill\w*|bbq|bil(ar)?|jakt|fiske|fotboll)\b", re.IGNORECASE)

POINTS = {
    "no_text": 25, "tiny_text": 12, "scam_words": 45, "handle": 10, "one_photo": 10,
    "male_per_hit": 5, "male_cap": 15, "professional": 15, "filtered": 10, "swim_or_cleavage": 5, "visual_cap": 25,
}


@dataclass
class FakeScore:
    score: int
    reasons: list[str] = field(default_factory=list)


def score(profile: db.Profile, photo_count: int = 0, visual: dict[str, str | None] | None = None) -> FakeScore:
    """Score one profile. `visual` is its attribute labels (see attributes.profile_attributes) or None if unknown."""
    parts = db.profile_text_parts(profile)
    text = " ".join(parts)
    words = len(re.findall(r"\w+", text))
    total, reasons = 0, []

    def add(points: int, why: str):
        nonlocal total
        total += points
        reasons.append(f"{why} (+{points})")

    if words == 0:
        add(POINTS["no_text"], "no profile text")
    elif words < 8:
        add(POINTS["tiny_text"], "almost no profile text")
    hit = SCAM_RE.search(text)
    if hit:
        add(POINTS["scam_words"], f"mentions '{hit.group(0).lower()}'")
    if HANDLE_RE.search(text):
        add(POINTS["handle"], "pushes an Instagram/Snapchat/@handle")
    if photo_count == 1:
        add(POINTS["one_photo"], "only one photo")
    male = {m.group(0).lower() for m in MALE_RE.finditer(text)}
    if male:
        add(min(POINTS["male_cap"], POINTS["male_per_hit"] * len(male)), f"male-coded interests ({', '.join(sorted(male))})")
    if visual:
        vis = 0
        if visual.get("professional") == "yes":
            vis += POINTS["professional"]
            reasons.append(f"studio-polished photos (+{POINTS['professional']})")
        if visual.get("filtered") == "yes":
            vis += POINTS["filtered"]
            reasons.append(f"heavy filter (+{POINTS['filtered']})")
        if visual.get("swimwear") == "yes" or visual.get("cleavage") == "yes":
            vis += POINTS["swim_or_cleavage"]
            reasons.append(f"swimwear/cleavage shots (+{POINTS['swim_or_cleavage']})")
        total += min(vis, POINTS["visual_cap"])
    return FakeScore(min(100, total), reasons)
