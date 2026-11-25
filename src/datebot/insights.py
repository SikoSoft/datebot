"""Descriptive statistics about your taste, for the /insights page.

Every stat is the same shape: a *card* (one attribute) made of *rows* (one value of it), each row giving how many
rated profiles had that value and what share of them you liked. The baseline is your overall like rate, so a row
is interesting when it sits clearly above or below it. Cards come from stored profile data (age, height, distance,
descriptors, interests, prompts, text) and from the visual labels in `attributes.py`.

Profiles scoring above the chosen fake-profile threshold (`fake.py`) are left out of every stat.
Only your own decisions count (`db.latest_ratings`: manual + app, never the bot's swipes). Nothing is hidden by
the hard filters: you want to see the whole picture, including what you filtered out.
"""
from __future__ import annotations

import math
import re
import sqlite3
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Callable, Iterable

from datebot import attributes, db, embed, fake, filters

MIN_HIGHLIGHT_N = 8
HIGHLIGHTS = 12
MAX_TAG_ROWS = 30
MIN_TAG_N = 4
MAX_DESCRIPTOR_VALUES = 15
MIN_DESCRIPTOR_PROFILES = 10

EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿]")
HANDLE = re.compile(r"@\w{3,}|\b(insta(gram)?|snap(chat)?|tiktok)\b", re.IGNORECASE)

# Themes mentioned in bios/prompts: label -> regex (English plus the Swedish/Danish this data mostly comes in).
THEMES: dict[str, str] = {
    "dogs": r"\b(dogs?|puppy|hund(ar)?)\b",
    "cats": r"\b(cats?|kitten|katt(er)?|kat(te)?)\b",
    "travel": r"\b(travel(l?ing)?|trips?|resa|resor|rejse[rn]?|adventures?|backpack\w*)\b",
    "gym / training": r"\b(gym|workout|training|crossfit|lifting|tr[aä]ning|fitness)\b",
    "hiking / outdoors": r"\b(hik(e|ing)|camping|outdoors?|nature|vandr\w+|friluft\w*|natur)\b",
    "coffee": r"\b(coffee|kaffe|latte)\b",
    "wine / drinks": r"\b(wine|vin|beer|[oö]l|cocktails?|drinks?|gin)\b",
    "food / cooking": r"\b(food|cooking|cook|baking|foodie|mat|matlagning|madlavning|restaurants?)\b",
    "humour": r"\b(humou?r|funny|laugh\w*|sarcas\w*|joke\w*|sk[aä]mt\w*|humor)\b",
    "music / concerts": r"\b(music|concerts?|festivals?|musik|koncert\w*)\b",
    "books / reading": r"\b(books?|reading|b[oö]cker|l[aä]sa|bøger)\b",
    "movies / series": r"\b(movies?|netflix|series|film\w*|tv)\b",
    "yoga / wellness": r"\b(yoga|meditation|wellness|pilates)\b",
    "astrology": r"\b(astrology|zodiac|horoscope|stjärntecken)\b",
    "kids": r"\b(kids?|children|mum|mom|mamma|barn|b[oö]rn)\b",
    "looking for something serious": r"\b(serious|relationship|long[- ]term|partner|kærlighed|k[aä]rlek|f[oö]rh[aå]llande)\b",
    "casual / fun": r"\b(casual|fun|no strings|spontaneous|hook ?ups?)\b",
    "no drama / honesty": r"\b(no drama|honest\w*|loyal\w*|[aä]rlig\w*|ærlig\w*)\b",
    "swipe left if…": r"\b(swipe left|don'?t swipe|dealbreaker|red flags?)\b",
    "career / work": r"\b(career|work(ing)?|job|business|entrepreneur|arbete|jobb|arbejde)\b",
    "family / friends": r"\b(family|friends|familj|v[aä]nner|venner)\b",
}
THEME_RES = {k: re.compile(v, re.IGNORECASE) for k, v in THEMES.items()}


@dataclass
class Row:
    label: str
    n: int
    likes: int
    rate: float
    lo: float  # 95% Wilson interval for the like rate
    hi: float
    z: float  # how many standard errors the rate sits from your baseline

    def to_dict(self) -> dict:
        return {"label": self.label, "n": self.n, "likes": self.likes, "rate": self.rate,
                "lo": self.lo, "hi": self.hi, "z": self.z}


@dataclass
class Card:
    id: str
    title: str
    section: str
    rows: list[Row]
    note: str = ""
    visual: bool = False  # a CLIP guess rather than data from the profile
    ordered: bool = False  # rows have a natural order (age, height) and are not sorted by size
    covered: int = 0  # rated profiles this card could say something about
    total: int = 0
    skip_in_highlights: tuple[str, ...] = field(default=())  # row labels that only mirror another row

    def to_dict(self) -> dict:
        return {"id": self.id, "title": self.title, "section": self.section, "note": self.note,
                "visual": self.visual, "ordered": self.ordered, "covered": self.covered, "total": self.total,
                "rows": [r.to_dict() for r in self.rows]}


def wilson(likes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = likes / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, centre - half), min(1.0, centre + half)


def make_row(label: str, likes: int, n: int, baseline: float) -> Row:
    lo, hi = wilson(likes, n)
    se = math.sqrt(max(baseline * (1 - baseline), 1e-9) / n) if n else 1.0
    return Row(label, n, likes, likes / n if n else 0.0, lo, hi, (likes / n - baseline) / se if n else 0.0)


def _fold(label: str) -> str:
    """Spelling-insensitive key: 'Malmö', 'Malmo' and ' malmo ' are the same place."""
    decomposed = unicodedata.normalize("NFKD", label)
    return "".join(c for c in decomposed if not unicodedata.combining(c)).casefold().strip()


def merge_variants(groups: dict[str, list[bool]]) -> dict[str, list[bool]]:
    """Merge values that differ only by accents/case/spacing, keeping the spelling with the most profiles."""
    by_key: dict[str, list[str]] = defaultdict(list)
    for label in groups:
        by_key[_fold(label)].append(label)
    merged: dict[str, list[bool]] = {}
    for labels in by_key.values():
        best = max(labels, key=lambda l: (len(groups[l]), l))
        merged[best] = [x for l in labels for x in groups[l]]
    return merged


def _tally(groups: dict[str, list[bool]], baseline: float, order: list[str] | None = None) -> list[Row]:
    rows = [make_row(k, sum(v), len(v), baseline) for k, v in groups.items() if v]
    if order is not None:
        rows.sort(key=lambda r: order.index(r.label))
    else:
        rows.sort(key=lambda r: -r.n)
    return rows


# --- bucketing -------------------------------------------------------------------------------------------


def bucket(value: float, edges: list[float], unit: str = "") -> str:
    """'<5', '5–10', ... '100+' for a value against ascending edges."""
    if value < edges[0]:
        return f"<{edges[0]:g}{unit}"
    for lo, hi in zip(edges, edges[1:]):
        if value < hi:
            return f"{lo:g}–{hi:g}{unit}"
    return f"{edges[-1]:g}+{unit}"


def _bucket_order(edges: list[float], unit: str = "") -> list[str]:
    return [bucket(edges[0] - 1, edges, unit)] + [bucket(e, edges, unit) for e in edges]


def step_bucket(value: int, step: int) -> str:
    lo = value // step * step
    return f"{lo}–{lo + step - 1}"


# --- extractors: profile -> value (None = unknown, the profile is simply left out of that card) ----------------


def descriptor_pairs(profile: db.Profile) -> dict[str, str]:
    """'Key: value' descriptor lines as {key: value} (first occurrence wins)."""
    out: dict[str, str] = {}
    for line in db._strings(profile.extra.get("descriptors")):
        key, sep, value = line.partition(":")
        if sep and key.strip() and value.strip():
            out.setdefault(key.strip(), value.strip())
    return out


def prompt_questions(profile: db.Profile) -> list[str]:
    """The prompt questions they chose to answer ('My simple pleasures', ...), when stored as 'Question: answer'."""
    out = []
    for line in db._strings(profile.extra.get("prompts")):
        q, sep, _ = line.partition(":")
        if sep and 3 <= len(q.strip()) <= 70:
            out.append(q.strip())
    return out


def word_count(profile: db.Profile) -> int:
    return len(re.findall(r"\w+", " ".join(db.profile_text_parts(profile))))


def theme_hits(text: str) -> list[str]:
    return [k for k, rx in THEME_RES.items() if rx.search(text)]


# --- the builder -----------------------------------------------------------------------------------------


def _rated_rows(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT r.profile_id, r.liked, r.created_at FROM ratings r WHERE r.id IN ("
        "  SELECT MAX(id) FROM ratings WHERE origin IN ('manual', 'app') GROUP BY profile_id) ORDER BY r.id"
    ).fetchall()


def visual_labels(conn: sqlite3.Connection) -> dict[int, dict[str, str | None]]:
    """Visual labels from the stored photo embeddings (separate function so tests can replace the CLIP part)."""
    from datebot import embed

    return attributes.profile_attributes(conn, embed.EMBED_ID)


def build(conn: sqlite3.Connection, source: str | None = None,
          visual: Callable[[sqlite3.Connection], dict] | None = visual_labels,
          fake_threshold: int | None = None) -> dict:
    """All cards. Profiles whose `fake.score` is >= `fake_threshold` are left out of every statistic (None = keep all)."""
    rated = _rated_rows(conn)
    labels: dict = {}
    visual_note = ""
    if visual is not None and rated:
        try:
            labels = visual(conn)
        except ImportError:
            visual_note = "Visual stats need the embed extra: uv sync --extra embed"
    profiles: dict[int, db.Profile] = {}
    liked: dict[int, bool] = {}
    when: dict[int, str] = {}
    for r in rated:
        p = db.get_profile(conn, r["profile_id"])
        if p is None or (source and p.source != source):
            continue
        profiles[p.id] = p
        liked[p.id] = bool(r["liked"])
        when[p.id] = r["created_at"]
    photo_counts = {r["profile_id"]: r["c"] for r in conn.execute("SELECT profile_id, COUNT(*) c FROM photos GROUP BY profile_id")}
    excluded = []
    if fake_threshold is not None:
        marks = db.fake_marks(conn)  # your verdict beats the heuristic, both ways
        for pid in list(profiles):
            fs = fake.score(profiles[pid], photo_counts.get(pid, 0), labels.get(pid))
            manual = marks.get(pid)
            if manual is True or (manual is None and fs.score >= fake_threshold):
                excluded.append({"id": pid, "name": profiles[pid].name, "source": profiles[pid].source,
                                 "score": 100 if manual else fs.score, "liked": liked[pid],
                                 "reasons": ["marked fake by you"] if manual else fs.reasons, "manual": bool(manual)})
                del profiles[pid], liked[pid], when[pid]
    excluded.sort(key=lambda e: (not e["manual"], -e["score"]))
    n = len(profiles)
    likes = sum(liked.values())
    baseline = likes / n if n else 0.0

    cards: list[Card] = []

    def add(card_id, title, section, values: dict[int, str | None], *, order=None, note="", ordered=False,
            visual_card=False, skip=(), limit: int | None = None):
        groups: dict[str, list[bool]] = defaultdict(list)
        for pid, v in values.items():
            if v is not None:
                groups[v].append(liked[pid])
        if order is None:  # fixed orders are our own labels, not free text
            groups = merge_variants(groups)
        rows = _tally(groups, baseline, order)
        if limit:
            rows = rows[:limit]
        if len(rows) >= (1 if visual_card else 2):
            cards.append(Card(card_id, title, section, rows, note, visual_card, ordered or order is not None,
                              sum(len(v) for v in groups.values()), n, skip))

    def add_tags(card_id, title, section, tags: dict[int, Iterable[str]], note="", min_n=MIN_TAG_N, limit=MAX_TAG_ROWS):
        """Cards where each row is 'profiles that have this tag' (a profile can be in many rows)."""
        groups: dict[str, list[bool]] = defaultdict(list)
        covered = 0
        for pid, ts in tags.items():
            ts = set(ts)
            covered += bool(ts)
            for t in ts:
                groups[t].append(liked[pid])
        groups = merge_variants(groups)
        rows = [make_row(k, sum(v), len(v), baseline) for k, v in groups.items() if len(v) >= min_n]
        rows.sort(key=lambda r: -r.n)
        if rows:
            cards.append(Card(card_id, title, section, rows[:limit], note, covered=covered, total=n))

    ids = list(profiles)

    # Overview
    add("source", "By app", "Overview", {i: profiles[i].source for i in ids})
    months = {i: when[i][:7] for i in ids if when.get(i)}
    add("month", "By month rated", "Overview", months, ordered=True, order=sorted(set(months.values())),
        note="Does your taste drift? Dates are when the rating was stored (a captured swipe is stored at capture time).")

    # Basics
    ages = {i: profiles[i].age for i in ids if profiles[i].age}
    add("age", "Age", "Basics", {i: step_bucket(a, 5) for i, a in ages.items()}, ordered=True,
        order=sorted({step_bucket(a, 5) for a in ages.values()}, key=lambda s: int(s.split("–")[0])))
    heights = {i: filters.height_cm(profiles[i]) for i in ids}
    heights = {i: h for i, h in heights.items() if h}
    add("height", "Stated height (cm)", "Basics", {i: step_bucket(h, 5) for i, h in heights.items()}, ordered=True,
        order=sorted({step_bucket(h, 5) for h in heights.values()}, key=lambda s: int(s.split("–")[0])))
    dist_edges = [5, 10, 25, 50, 100]
    dists = {i: filters.distance_km(profiles[i]) for i in ids}
    dists = {i: d for i, d in dists.items() if d is not None}
    add("distance", "Distance (km)", "Basics", {i: bucket(d, dist_edges) for i, d in dists.items()},
        order=_bucket_order(dist_edges))
    add("city", "City", "Basics", {i: profiles[i].extra.get("city") for i in ids}, limit=12)
    add("photos", "Number of photos", "Basics", {i: ("8+" if photo_counts.get(i, 0) >= 8 else str(photo_counts.get(i, 0))) for i in ids},
        ordered=True, order=[str(x) for x in range(0, 8)] + ["8+"])

    # Profile text
    words = {i: word_count(profiles[i]) for i in ids}
    w_edges = [1, 20, 60, 120]
    add("words", "Amount of profile text (words)", "Profile text",
        {i: ("none" if w == 0 else bucket(w, w_edges)) for i, w in words.items()},
        order=["none"] + _bucket_order(w_edges)[1:])
    texts = {i: " ".join(db.profile_text_parts(profiles[i])) for i in ids}
    add("emoji", "Uses emoji in text", "Profile text",
        {i: ("yes" if EMOJI.search(t) else "no") for i, t in texts.items() if t}, order=["yes", "no"])
    add("handle", "Mentions Instagram / Snapchat / @handle", "Profile text",
        {i: ("yes" if HANDLE.search(t) else "no") for i, t in texts.items() if t}, order=["yes", "no"])
    add("interest_count", "Number of interests listed", "Profile text",
        {i: bucket(len(db._strings(profiles[i].extra.get("interests"))), [1, 5, 9]) for i in ids if "interests" in profiles[i].extra},
        order=_bucket_order([1, 5, 9]))
    add_tags("themes", "Themes mentioned in text", "Profile text", {i: theme_hits(t) for i, t in texts.items() if t},
             note="Keyword match on bio/prompts/interests (English, Swedish, Danish).", min_n=3)
    add_tags("prompts", "Prompts they chose to answer", "Profile text",
             {i: prompt_questions(profiles[i]) for i in ids if "prompts" in profiles[i].extra})
    add_tags("interests", "Interests", "Profile text",
             {i: db._strings(profiles[i].extra.get("interests")) for i in ids if "interests" in profiles[i].extra})

    # Lifestyle descriptors ("Workout: Often", "Children: ...", ...): one card per key
    pairs = {i: descriptor_pairs(profiles[i]) for i in ids}
    by_key: dict[str, dict[int, str]] = defaultdict(dict)
    for i, d in pairs.items():
        for k, v in d.items():
            by_key[k][i] = v
    for key, vals in sorted(by_key.items(), key=lambda kv: -len(kv[1])):
        if len(vals) >= MIN_DESCRIPTOR_PROFILES and len(set(vals.values())) <= MAX_DESCRIPTOR_VALUES:
            add(f"desc-{re.sub(r'[^a-z0-9]+', '-', key.lower())}", key, "Lifestyle", vals)

    # Visual labels
    covered_visual = 0
    if visual is not None and n:
        covered_visual = sum(1 for i in ids if i in labels)
        if labels and covered_visual < n:
            visual_note = f"{n - covered_visual} rated profiles have no embedded photos yet (run: datebot embed)."
        for g in attributes.GROUPS:
            values = {i: labels[i].get(g.key) for i in ids if i in labels}
            if g.kind == "flag":
                values = {i: ("shows it" if v == "yes" else "doesn't stand out") for i, v in values.items() if v}
                add(f"vis-{g.key}", g.title, g.section if g.section != "Photos" else "Photo content", values,
                    note=g.note, visual_card=True, order=["shows it", "doesn't stand out"], skip=("doesn't stand out",))
            else:
                add(f"vis-{g.key}", g.title, g.section, values, note=g.note, visual_card=True, skip=(attributes.UNCLEAR,))

    return {
        "summary": {"rated": n, "likes": likes, "passes": n - likes, "baseline": baseline,
                    "sources": dict(Counter(p.source for p in profiles.values())),
                    "embeddings": db.embedding_status(conn, embed.EMBED_ID),
                    "visual_covered": covered_visual, "visual_note": visual_note, "source": source,
                    "fake": {"threshold": fake_threshold, "excluded": len(excluded),
                             "excluded_likes": sum(e["liked"] for e in excluded), "examples": excluded[:40]}},
        "highlights": highlights(cards, baseline),
        "cards": [c.to_dict() for c in cards],
    }


def highlights(cards: list[Card], baseline: float) -> list[dict]:
    """The rows that differ most from your baseline, among rows large enough to mean something.

    There are dozens of rows, so a few will look striking by chance: lean on the ones with a large n that
    point the same way as related rows."""
    found = []
    for c in cards:
        for r in c.rows:
            if r.n < MIN_HIGHLIGHT_N or r.label in c.skip_in_highlights:
                continue
            if r.lo <= baseline <= r.hi:
                continue
            found.append({"card": c.id, "title": c.title, "label": r.label, "n": r.n, "rate": r.rate,
                          "baseline": baseline, "z": r.z, "visual": c.visual})
    found.sort(key=lambda h: -abs(h["z"]))
    return found[:HIGHLIGHTS]
