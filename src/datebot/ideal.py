"""The "ideal" page: your like rates (insights.py) boiled down to one preferred value per attribute.

For every attribute card the value you like most is picked, but only when it is backed by enough profiles and
clearly beats your average ("clear": the 95% interval is above it; "lean": above it by at least one standard error).
Attributes where nothing stands out are reported as undecided instead of being guessed. Values you clearly dislike
become the "avoid" list. From the picks we build (a) a plain-language description, (b) a text prompt you could
paste into an image generator (nothing is sent anywhere from here), and (c) a ranking of profiles you have already
seen by how many ideal traits they have and by the trained model's score, with photos.

Everything inherits the limits of the insights: the visual labels are noisy CLIP guesses and the picks are only as
good as the number of ratings behind them. Only your own decisions count (manual + app, never the bot's swipes).
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

from datebot import attributes, db, insights, model

MIN_N = 8  # profiles a value needs before it can be a pick
LEAN_Z = 1.0
MAX_ALSO = 2
MAX_PERSONALITY = 10
CANDIDATES = 60  # profiles returned per ranking
MIN_MATCH_TRAITS = 3  # fewer detectable traits than this and a profile gets no match percentage
LOOK_SECTIONS = ("Looks", "Body", "Details", "Style")
IGNORED_LABELS = {attributes.UNCLEAR, "not visible"}
SHOWS = "shows it"
LOAD = object()  # build(trained=LOAD): load the saved model; None means "no model"

# How a picked value reads inside an image prompt, by attribute key. Missing key = not used in the prompt.
# Breast size and cleavage are deliberately left out: the weakest detectors, and generators refuse them anyway.
PROMPT_PARTS: dict[str, str] = {
    "ethnicity": "{} woman",
    "hair_colour": "{} hair",
    "hair_length": "{} hair",
    "hair_texture": "{} hair",
    "skin_tone": "{} skin",
    "makeup": "{} makeup",
    "expression": "{} expression",
    "body_type": "{} build",
    "style": "{} style outfit",
}
PROMPT_FLAGS = {"glasses": "wearing glasses", "freckles": "freckles", "tattoos": "visible tattoos",
                "bangs": "bangs", "piercings": "a small facial piercing", "hat": "wearing a hat"}
PROMPT_WORDS = {"white / European": "white European", "petite / slim": "slim petite", "none / natural": "no",
                "light": "light", "up": "tied-up", "medium/tan": "olive-tan", "grey/white": "grey"}


def _strength(r: dict, baseline: float) -> str | None:
    if r["n"] < MIN_N:
        return None
    if r["lo"] > baseline:
        return "clear"
    if r["z"] >= LEAN_Z and r["rate"] > baseline:
        return "lean"
    return None


def _row(r: dict, baseline: float) -> dict:
    return {"label": r["label"], "rate": r["rate"], "n": r["n"], "lo": r["lo"], "hi": r["hi"],
            "strength": _strength(r, baseline)}


def pick_for_card(card: dict, baseline: float) -> dict | None:
    """{'pick', 'also', 'avoid'} for one card, or None when nothing in it stands out in either direction."""
    rows = [r for r in card["rows"] if r["label"] not in IGNORED_LABELS]
    if card["visual"] and rows and all(r["label"] in (SHOWS, "doesn't stand out") for r in rows):  # a flag
        shows = next((r for r in rows if r["label"] == SHOWS), None)
        if shows is None or shows["n"] < MIN_N:
            return None
        s = _strength(shows, baseline)
        if s:
            return {"pick": _row(shows, baseline), "also": [], "avoid": []}
        if shows["n"] >= MIN_N and shows["hi"] < baseline:
            return {"pick": None, "also": [], "avoid": [_row(shows, baseline)]}
        return None
    good = sorted((r for r in rows if _strength(r, baseline)), key=lambda r: -r["rate"])
    bad = sorted((r for r in rows if r["n"] >= MIN_N and r["hi"] < baseline), key=lambda r: r["rate"])
    if not good and not bad:
        return None
    return {"pick": _row(good[0], baseline) if good else None,
            "also": [_row(r, baseline) for r in good[1:1 + MAX_ALSO]],
            "avoid": [_row(r, baseline) for r in bad]}


def _card_key(card_id: str) -> str:
    return card_id.removeprefix("vis-")


def _entry(card: dict, found: dict) -> dict:
    return {"card": card["id"], "key": _card_key(card["id"]), "title": card["title"], "section": card["section"],
            "visual": card["visual"], "flag": found["pick"] is not None and found["pick"]["label"] == SHOWS
            or any(a["label"] == SHOWS for a in found["avoid"]), **found}


def describe(entries: list[dict]) -> str:
    """One readable sentence of the picked looks."""
    parts = []
    for e in entries:
        p = e["pick"]
        if not p:
            continue
        label = p["label"]
        parts.append(e["title"].lower() if label == SHOWS else f"{label} {e['title'].lower()}")
    return ", ".join(parts)


def image_prompt(entries: list[dict]) -> tuple[str, str]:
    """(prompt, negative prompt) for an image generator, from the visual picks that read naturally in one."""
    bits, flags, negatives = [], [], []
    by_key = {e["key"]: e for e in entries}
    for key, template in PROMPT_PARTS.items():
        e = by_key.get(key)
        if e and e["pick"]:
            label = e["pick"]["label"]
            bits.append(template.format(PROMPT_WORDS.get(label, label)))
    for key, phrase in PROMPT_FLAGS.items():
        e = by_key.get(key)
        if e and e["pick"] and e["pick"]["label"] == SHOWS:
            flags.append(phrase)
        elif e and e["avoid"]:
            negatives.append(phrase)
    for key in ("hair_colour", "hair_length", "hair_texture", "makeup", "style", "body_type"):
        e = by_key.get(key)
        if e:
            negatives += [f"{a['label']} {e['title'].lower()}" for a in e["avoid"]]
    if not bits and not flags:
        return "", ""
    prompt = "Candid natural-light portrait photo of " + ", ".join(bits + flags) + ", fully clothed, relaxed, looking at the camera"
    return prompt, ", ".join(negatives)


def _matches(entry: dict, label: str | None) -> bool | None:
    """Does a profile's visual label satisfy this pick? None = the detector could not say."""
    if label is None or label == attributes.UNCLEAR:
        return None
    pick = entry["pick"]
    if pick["label"] == SHOWS:
        return label == "yes"
    return label in {pick["label"], *(a["label"] for a in entry["also"])}


def build(conn: sqlite3.Connection, source: str | None = None, fake_threshold: int | None = None,
          model_name: str = "default", labels: dict | None = None, trained=LOAD) -> dict:
    """Everything the /ideal page shows. `labels` (and `trained`) can be passed in by tests."""
    if labels is None:
        try:
            labels = insights.visual_labels(conn)
        except ImportError:
            labels = {}
    data = insights.build(conn, source, visual=lambda c: labels, fake_threshold=fake_threshold)
    baseline = data["summary"]["baseline"]
    cards = {c["id"]: c for c in data["cards"]}

    looks, basics, undecided = [], [], []
    for card in data["cards"]:
        if card["visual"] and card["section"] in LOOK_SECTIONS:
            found = pick_for_card(card, baseline)
            (looks if found else undecided).append(_entry(card, found) if found else card["title"])
    for card_id in ("age", "height"):
        if card_id in cards:
            found = pick_for_card(cards[card_id], baseline)
            if found:
                basics.append(_entry(cards[card_id], found))
            else:
                undecided.append(cards[card_id]["title"])
    looks.sort(key=lambda e: (e["pick"] is None, -(e["pick"]["lo"] if e["pick"] else 0)))

    likes, avoids = [], []
    for card in data["cards"]:
        if card["visual"] or card["section"] not in ("Profile text", "Lifestyle") or card["id"] in ("words", "interest_count"):
            continue
        for r in card["rows"]:
            row = {"title": card["title"], "label": r["label"], "rate": r["rate"], "n": r["n"]}
            if r["n"] >= MIN_N and r["lo"] > baseline:
                likes.append({**row, "z": r["z"]})
            elif r["n"] >= MIN_N and r["hi"] < baseline:
                avoids.append({**row, "z": r["z"]})
    likes.sort(key=lambda r: -r["z"])
    avoids.sort(key=lambda r: r["z"])

    prompt, negative = image_prompt(looks)
    result = {
        "summary": {**{k: data["summary"][k] for k in ("rated", "likes", "baseline", "visual_covered", "visual_note")},
                    "min_n": MIN_N},
        "looks": looks, "basics": basics, "undecided": undecided,
        "personality": {"likes": likes[:MAX_PERSONALITY], "avoids": avoids[:MAX_PERSONALITY]},
        "description": describe(looks), "image_prompt": prompt, "negative_prompt": negative,
        "matches": [], "model_active": False,
    }
    if trained is LOAD:
        trained = model.load(model_name)
    result["model_active"] = trained is not None
    result["matches"] = rank_profiles(conn, looks, labels, trained, set(data["summary"]["fake"].get("ids", [])), source)
    return result


def rank_profiles(conn, looks: list[dict], labels: dict, trained, excluded: set[int], source: str | None) -> list[dict]:
    picks = [e for e in looks if e["pick"]]
    human = db.latest_ratings(conn)
    auto = db.auto_swiped_ids(conn)
    marks = db.fake_marks(conn)
    items = []
    for (pid,) in conn.execute("SELECT id FROM profiles ORDER BY id").fetchall():
        p = db.get_profile(conn, pid)
        if pid in excluded or marks.get(pid) is True or (source and p.source != source):
            continue
        hit, miss = [], []
        for e in picks:
            ok = _matches(e, labels.get(pid, {}).get(e["key"]))
            if ok is True:
                hit.append(f"{e['title']}: {e['pick']['label']}" if e["pick"]["label"] != SHOWS else e["title"])
            elif ok is False:
                miss.append(e["title"])
        known = len(hit) + len(miss)
        items.append({"id": pid, "name": p.name, "age": p.age, "source": p.source, "yours": human.get(pid),
                      "bot": auto.get(pid) if pid not in human else None,
                      "score": model.score_profile(conn, trained, pid) if trained else None,
                      "match": len(hit) / known if known >= MIN_MATCH_TRAITS else None,
                      "matched": hit, "missed": miss, "traits_known": known})

    def top(key):
        return sorted((i for i in items if i[key] is not None), key=lambda i: (-i[key], -(i["score"] or 0)))[:CANDIDATES]

    chosen = {i["id"]: i for i in top("score") + top("match")}
    for i in chosen.values():
        i["photos"] = [f"/photos/{Path(x).name}" for x in db.photo_paths(conn, i["id"])][:4]
    return list(chosen.values())
