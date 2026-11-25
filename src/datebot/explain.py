"""Answers "what does my taste seem to be about?" by asking CLIP about attributes after the fact.

Nothing here is labelled up front. CLIP puts photos and text in the same space, so the similarity between a
photo's embedding and the text "a person wearing glasses" is a (noisy) glasses detector. For each attribute we
score every photo, collapse that to one number per profile, and measure how well it separates your likes from
your passes. The detector is zero-shot and imperfect, so check it with `--examples`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import rankdata

MIN_PER_CLASS = 3
PERMUTATIONS = 2000
FDR = 0.10  # Benjamini-Hochberg false-discovery rate for flagging attributes as real signal


@dataclass(frozen=True)
class Attribute:
    name: str  # short label shown in the report
    prompt: str  # what CLIP is asked about
    group: str
    agg: str = "max"  # "max": any photo shows it; "mean": how much of the profile is like this


ATTRIBUTES: list[Attribute] = [
    # styling
    Attribute("glasses", "a photo of a person wearing glasses", "styling"),
    Attribute("sunglasses", "a photo of a person wearing sunglasses", "styling"),
    Attribute("tattoos", "a photo of a person with tattoos", "styling"),
    Attribute("heavy makeup", "a photo of a woman with heavy makeup", "styling"),
    Attribute("natural look", "a photo of a woman with no makeup, natural look", "styling"),
    Attribute("long hair", "a photo of a person with long hair", "styling"),
    Attribute("short hair", "a photo of a person with short hair", "styling"),
    Attribute("blonde", "a photo of a woman with blonde hair", "styling"),
    Attribute("dark hair", "a photo of a woman with dark black hair", "styling"),
    Attribute("red/coloured hair", "a photo of a woman with red or dyed hair", "styling"),
    Attribute("smiling", "a photo of a person smiling broadly", "styling"),
    # photo type
    Attribute("gym (share)", "a photo of a person at the gym", "photo type", "mean"),
    Attribute("gym (any photo)", "a photo of a person working out at the gym", "photo type"),
    Attribute("shirtless/swimwear", "a photo of a person in a bikini or swimsuit at the beach", "photo type"),
    Attribute("mirror selfie", "a mirror selfie", "photo type", "mean"),
    Attribute("selfies", "a close-up selfie", "photo type", "mean"),
    Attribute("group photo", "a group photo of several friends", "photo type", "mean"),
    Attribute("outdoors/nature", "a photo of a person hiking outdoors in nature", "photo type", "mean"),
    Attribute("travel", "a photo of a person travelling abroad, tourist photo", "photo type", "mean"),
    Attribute("with pet", "a photo of a person with a dog or cat", "photo type"),
    Attribute("party/nightlife", "a photo of a person at a party or nightclub", "photo type", "mean"),
    Attribute("dressed up", "a photo of a person in an elegant dress or suit", "photo type"),
    Attribute("casual", "a photo of a person in casual clothes, t-shirt and jeans", "photo type", "mean"),
    # body
    Attribute("cleavage", "a photo of a woman with cleavage in a low-cut top", "body"),
    Attribute("large breasts", "a photo of a woman with large breasts", "body"),
    Attribute("slim", "a photo of a slim woman", "body"),
    Attribute("athletic", "a photo of a fit athletic woman", "body"),
    Attribute("curvy", "a photo of a curvy woman", "body"),
    Attribute("plus size", "a photo of a plus-size woman", "body"),
    Attribute("tall", "a photo of a very tall person", "body"),
    # appearance / ethnicity (CLIP is least reliable here: use --examples to check what it actually finds)
    Attribute("East Asian", "a photo of an East Asian woman", "ethnicity"),
    Attribute("Southeast Asian", "a photo of a Southeast Asian woman", "ethnicity"),
    Attribute("South Asian", "a photo of a South Asian woman", "ethnicity"),
    Attribute("Black", "a photo of a Black woman", "ethnicity"),
    Attribute("White", "a photo of a White woman", "ethnicity"),
    Attribute("Latina", "a photo of a Latina woman", "ethnicity"),
    Attribute("Middle Eastern", "a photo of a Middle Eastern woman", "ethnicity"),
]

GROUPS = sorted({a.group for a in ATTRIBUTES})


@dataclass
class Result:
    attribute: Attribute
    auc: float  # 0.5 = no relation; above = you like profiles that show it more; below = you pass on them
    like_rate_high: float  # like rate among the profiles that show it most (top third)
    like_rate_low: float  # ... and among those that show it least (bottom third)
    n_third: int
    p: float  # permutation p-value for "AUC differs from 0.5"
    q: float = 1.0  # Benjamini-Hochberg adjusted p across the attributes tested together
    examples: list[str] | None = None  # photo paths where it scored highest

    @property
    def significant(self) -> bool:
        return self.q <= FDR

    @property
    def verdict(self) -> str:
        if not self.significant:
            return "no clear signal"
        return "you like this" if self.auc > 0.5 else "you tend to pass on this"


def auc_from_ranks(ranks: np.ndarray, y: np.ndarray) -> float:
    n1 = int(y.sum())
    n0 = len(y) - n1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def permutation_p(values: np.ndarray, y: np.ndarray, rng: np.random.Generator, n: int = PERMUTATIONS) -> float:
    """Two-sided p-value for AUC != 0.5 by shuffling the labels."""
    ranks = rankdata(values)
    observed = abs(auc_from_ranks(ranks, y) - 0.5)
    n1 = int(y.sum())
    n0 = len(y) - n1
    hits = 0
    for _ in range(n):
        perm = rng.permutation(y)
        if abs((ranks[perm == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) - 0.5) >= observed - 1e-12:
            hits += 1
    return (hits + 1) / (n + 1)


def benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    """Adjusted p-values (q-values): the share of flagged attributes expected to be flukes."""
    order = np.argsort(p)
    scaled = p[order] * len(p) / (np.arange(len(p)) + 1)
    q = np.minimum.accumulate(scaled[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.minimum(q, 1.0)
    return out


def photo_scores(photo_emb: np.ndarray, text_emb: np.ndarray) -> np.ndarray:
    """(n_photos, n_attributes) z-scored similarities: how much more than a typical photo here shows each attribute."""
    sims = photo_emb @ text_emb.T
    return (sims - sims.mean(axis=0)) / (sims.std(axis=0) + 1e-9)


def analyse(
    profile_ids: list[int],
    photo_paths: list[str],
    photo_emb: np.ndarray,
    text_emb: np.ndarray,
    attributes: list[Attribute],
    ratings: dict[int, bool],
    examples: int = 0,
    seed: int = 0,
) -> list[Result]:
    """Rank `attributes` by how strongly they separate your likes from your passes."""
    pids = np.asarray(profile_ids)
    rated = [pid for pid in dict.fromkeys(profile_ids) if pid in ratings]
    y = np.array([int(ratings[pid]) for pid in rated])
    if min(y.sum(), len(y) - y.sum()) < MIN_PER_CLASS:
        raise ValueError(
            f"Need at least {MIN_PER_CLASS} likes and {MIN_PER_CLASS} passes with embeddings to explain "
            f"(have {int(y.sum())} likes, {len(y) - int(y.sum())} passes)."
        )
    z = photo_scores(photo_emb, text_emb)
    rng = np.random.default_rng(seed)
    third = max(1, len(y) // 3)
    results = []
    for j, attr in enumerate(attributes):
        reduce = np.max if attr.agg == "max" else np.mean
        v = np.array([reduce(z[pids == pid, j]) for pid in rated])
        order = np.argsort(v)
        low, high = order[:third], order[-third:]
        ex = None
        if examples:
            ex = [photo_paths[i] for i in np.argsort(-z[:, j])[:examples]]
        results.append(Result(
            attr, auc_from_ranks(rankdata(v), y), float(y[high].mean()), float(y[low].mean()), third,
            permutation_p(v, y, rng), examples=ex,
        ))
    for r, q in zip(results, benjamini_hochberg(np.array([r.p for r in results]))):
        r.q = float(q)
    results.sort(key=lambda r: (-abs(r.auc - 0.5)))
    return results


def format_report(results: list[Result], n_likes: int, n_passes: int) -> str:
    lines = [f"Ratings used: {n_likes + n_passes} ({n_likes} likes, {n_passes} passes). "
             f"{len(results)} attributes tested, strongest first.", ""]
    if min(n_likes, n_passes) < 30:
        lines += ["WARNING: fewer than 30 likes or passes. With this little data most differences are chance;",
                  "only trust attributes flagged '*', and even those loosely.", ""]
    lines.append(f"{'attribute':<20} {'group':<11} {'AUC':>5}  {'liked if high / low':>20}  {'q':>5}  verdict")
    for r in results:
        rates = f"{r.like_rate_high:4.0%} / {r.like_rate_low:4.0%} (n={r.n_third})"
        lines.append(f"{r.attribute.name:<20} {r.attribute.group:<11} {r.auc:5.2f}  {rates:>20}  {r.q:5.2f}  "
                     f"{'* ' if r.significant else '  '}{r.verdict}")
        for path in r.examples or []:
            lines.append(f"    e.g. {path}")
    lines += [
        "",
        "AUC 0.5 = unrelated; above = you like profiles showing it more; below = you pass on them more.",
        f"'high / low' = your like rate on the third of profiles that show it most / least. '*' = survives a "
        f"{FDR:.0%} false-discovery correction across all attributes tested (q is the adjusted p-value).",
        "Detectors are zero-shot CLIP: look at --examples to check each one finds what its name says.",
        "An attribute can also stand in for something it correlates with (e.g. 'glasses' for a certain style).",
    ]
    return "\n".join(lines)
