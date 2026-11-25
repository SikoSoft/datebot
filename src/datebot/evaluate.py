"""Answers "is the model actually learning my taste?" using only your own ratings.

Everything here is cross-validated or held out, so scores never come from profiles the model was
trained on. Numbers from small rating sets are noisy; the report says so.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.base import clone
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from datebot import model

MIN_PER_CLASS = model.MIN_PER_CLASS
MIN_CURVE_TRAIN = 10  # smallest training-set size shown on the learning curve


def _minority(y: np.ndarray) -> int:
    return int(min(y.sum(), len(y) - y.sum()))


def check_enough(y: np.ndarray) -> None:
    if _minority(y) < MIN_PER_CLASS:
        raise ValueError(
            f"Need at least {MIN_PER_CLASS} likes and {MIN_PER_CLASS} passes to evaluate "
            f"(have {int(y.sum())} likes, {len(y) - int(y.sum())} passes)."
        )


def safe_auc(y_true: np.ndarray, scores: np.ndarray) -> float | None:
    """ROC AUC, or None when only one class is present (undefined)."""
    return float(roc_auc_score(y_true, scores)) if len(set(y_true.tolist())) == 2 else None


def oof_scores(X: np.ndarray, y: np.ndarray, repeats: int = 3, folds: int = 5, seed: int = 0) -> np.ndarray:
    """Out-of-fold P(like) per profile, averaged over `repeats` differently-shuffled CV runs."""
    check_enough(y)
    folds = max(2, min(folds, _minority(y)))
    total = np.zeros(len(y))
    for r in range(repeats):
        for train, test in StratifiedKFold(folds, shuffle=True, random_state=seed + r).split(X, y):
            clf = clone(model.make_classifier()).fit(X[train], y[train])
            total[test] += clf.predict_proba(X[test])[:, 1]
    return total / repeats


@dataclass
class Headline:
    auc: float | None
    accuracy: float
    balanced_accuracy: float
    majority_baseline: float  # accuracy of always guessing the more common label


def headline(y: np.ndarray, oof: np.ndarray) -> Headline:
    pred = (oof >= 0.5).astype(int)
    majority = max(y.mean(), 1 - y.mean())
    return Headline(safe_auc(y, oof), float(accuracy_score(y, pred)),
                    float(balanced_accuracy_score(y, pred)), float(majority))


def feature_blocks(X: np.ndarray) -> dict[str, np.ndarray | None]:
    """Split features laid out as [photo mean, photo max, lead photo, text mean, text max, has_text] into ablation sets.

    'text' is None when no rated profile has any text, since that comparison would be meaningless.
    """
    d = (X.shape[1] - 1) // 5
    photos, text_cols = X[:, : 3 * d], X[:, 3 * d :]
    has_text = X[:, -1] > 0
    return {"photos only": photos, "text only": text_cols if has_text.any() else None, "photos + text": X}


def ablation(X: np.ndarray, y: np.ndarray, **kw) -> dict[str, float | None]:
    """Cross-validated AUC using photos only, text only, and both."""
    out: dict[str, float | None] = {}
    for name, block in feature_blocks(X).items():
        out[name] = None if block is None else safe_auc(y, oof_scores(block, y, **kw))
    return out


@dataclass
class CurvePoint:
    n_train: int
    auc_mean: float | None
    auc_std: float | None


def _fit_auc(X, y, train, test) -> float | None:
    if min(y[train].sum(), len(train) - y[train].sum()) < 2:
        return None
    clf = clone(model.make_classifier()).fit(X[train], y[train])
    return safe_auc(y[test], clf.predict_proba(X[test])[:, 1])


def learning_curve(X: np.ndarray, y: np.ndarray, points: int = 6, repeats: int = 8, seed: int = 0) -> list[CurvePoint]:
    """Held-out AUC when training on n ratings, for growing n. A curve that is still rising means more
    ratings will help; a flat one means more of the same data will not."""
    check_enough(y)
    from sklearn.model_selection import train_test_split

    n_total = len(y)
    lo, hi = MIN_CURVE_TRAIN, int(n_total * 0.8)
    if hi < lo + 5:  # not enough ratings for a curve that means anything
        return []
    sizes = sorted({int(s) for s in np.linspace(lo, hi, points)})
    idx = np.arange(n_total)
    curve = []
    for n in sizes:
        aucs = []
        for r in range(repeats):
            try:
                train, test = train_test_split(idx, train_size=n, stratify=y, random_state=seed + r)
            except ValueError:
                continue
            if (a := _fit_auc(X, y, train, test)) is not None:
                aucs.append(a)
        curve.append(CurvePoint(n, float(np.mean(aucs)) if aucs else None, float(np.std(aucs)) if aucs else None))
    return curve


@dataclass
class Drift:
    n_train: int
    n_test: int
    future_auc: float | None   # train on your oldest ratings, test on your newest
    random_auc: float | None   # same sizes, but with a random split instead of a time split

    @property
    def drifting(self) -> bool:
        return self.future_auc is not None and self.random_auc is not None and self.future_auc < self.random_auc - 0.1


def drift_check(X: np.ndarray, y: np.ndarray, train_frac: float = 0.7, repeats: int = 8, seed: int = 0) -> Drift | None:
    """Rows must be in chronological order (model.training_data guarantees this).

    If newer ratings are much harder to predict than a random split of the same size, your taste has
    probably shifted (or you rated differently over time) and old ratings may be misleading the model.
    """
    from sklearn.model_selection import train_test_split

    n = len(y)
    cut = int(n * train_frac)
    train, test = np.arange(cut), np.arange(cut, n)
    if len(test) < 5 or min(y[train].sum(), cut - y[train].sum()) < MIN_PER_CLASS:
        return None
    future = _fit_auc(X, y, train, test)
    rand = []
    for r in range(repeats):
        try:
            tr, te = train_test_split(np.arange(n), test_size=len(test), stratify=y, random_state=seed + r)
        except ValueError:
            continue
        if (a := _fit_auc(X, y, tr, te)) is not None:
            rand.append(a)
    return Drift(cut, len(test), future, float(np.mean(rand)) if rand else None)


@dataclass
class Thresholds:
    """Score cut-offs (0-100) for auto-swiping, estimated from out-of-fold scores."""
    n: int
    like_at: float | None        # auto-like at or above this score
    like_precision: float | None  # of profiles scoring >= like_at, the share you actually liked
    like_count: int
    pass_at: float | None        # auto-pass at or below this score
    pass_precision: float | None  # of profiles scoring <= pass_at, the share you actually passed
    pass_count: int
    likes_lost: int              # your real likes that would have been auto-passed
    total_likes: int

    @property
    def coverage(self) -> float:
        """Share of profiles decided automatically (the rest go to your review queue)."""
        return (self.like_count + self.pass_count) / self.n if self.n else 0.0


def suggest_thresholds(
    y: np.ndarray, oof: np.ndarray, like_precision: float = 0.9, pass_precision: float = 0.95, min_support: int = 10
) -> Thresholds:
    """Cut-offs that hit both precision targets on at least `min_support` profiles each, chosen together
    (so the bands cannot overlap) to auto-decide as many profiles as possible.

    These are estimated on the same ratings used to pick them, so treat them as optimistic and
    re-check as you rate more.
    """
    scores = oof * 100
    n = len(y)
    desc = np.argsort(-scores, kind="stable")
    asc = desc[::-1]

    def valid_sizes(order: np.ndarray, target_class: int, target: float) -> np.ndarray:
        """Band sizes k (top-k of `order`) whose precision for `target_class` meets the target; 0 = no band."""
        k = np.arange(1, n + 1)
        ok = (np.cumsum(y[order] == target_class) / k >= target) & (k >= min_support)
        return np.concatenate([[0], k[ok]])

    likes_k, passes_k = valid_sizes(desc, 1, like_precision), valid_sizes(asc, 0, pass_precision)
    best_a, best_b = 0, 0
    for a in likes_k[::-1]:
        b = int(passes_k[np.searchsorted(passes_k, n - a, side="right") - 1])  # largest valid b with a + b <= n
        if a + b > best_a + best_b:
            best_a, best_b = int(a), b
    like_at = float(scores[desc][best_a - 1]) if best_a else None
    pass_at = float(scores[asc][best_b - 1]) if best_b else None
    if like_at is not None and pass_at is not None and pass_at >= like_at:
        pass_at = None  # tied scores at the boundary: keep the bands disjoint

    like_mask = scores >= like_at if like_at is not None else np.zeros(n, dtype=bool)
    pass_mask = scores <= pass_at if pass_at is not None else np.zeros(n, dtype=bool)
    return Thresholds(
        n=n,
        like_at=like_at,
        like_precision=float(y[like_mask].mean()) if like_mask.any() else None,
        like_count=int(like_mask.sum()),
        pass_at=pass_at,
        pass_precision=float(1 - y[pass_mask].mean()) if pass_mask.any() else None,
        pass_count=int(pass_mask.sum()),
        likes_lost=int(y[pass_mask].sum()),
        total_likes=int(y.sum()),
    )


@dataclass
class Surprise:
    profile_id: int
    liked: bool
    score: float  # out-of-fold score 0-100


def surprises(ids: list[int], y: np.ndarray, oof: np.ndarray, k: int = 5) -> list[Surprise]:
    """Your likes the model scored lowest and your passes it scored highest: where it is most wrong.
    Worth a look: they are either blind spots in the model or ratings you may want to revisit."""
    scores = oof * 100
    liked = [i for i in np.argsort(scores) if y[i] == 1][:k]
    passed = [i for i in np.argsort(-scores) if y[i] == 0][:k]
    return [Surprise(ids[i], bool(y[i]), float(scores[i])) for i in [*liked, *passed]]


@dataclass
class SourceStat:
    source: str
    n: int
    likes: int
    auc: float | None  # how well the pooled model ranks this source's profiles (cross-validated)
    accuracy: float


@dataclass
class Report:
    n: int
    likes: int
    passes: int
    headline: Headline
    ablation: dict[str, float | None]
    drift: Drift | None
    curve: list[CurvePoint]
    thresholds: Thresholds
    surprises: list[Surprise]
    warnings: list[str]
    by_source: list[SourceStat]


def verdict(auc: float | None) -> str:
    if auc is None:
        return "undefined"
    for bound, word in ((0.9, "strong"), (0.8, "good"), (0.7, "decent"), (0.6, "weak")):
        if auc >= bound:
            return word
    return "barely better than guessing"


def by_source(conn, ids: list[int], y: np.ndarray, oof: np.ndarray) -> list[SourceStat]:
    """Cross-validated quality per app, from the pooled model's out-of-fold scores. Biggest source first."""
    marks = ",".join("?" * len(ids))
    src = {r["id"]: r["source"] for r in conn.execute(f"SELECT id, source FROM profiles WHERE id IN ({marks})", ids)}
    names = np.array([src[i] for i in ids])
    out = []
    for name in sorted(set(names), key=lambda n: (-(names == n).sum(), n)):
        m = names == name
        out.append(SourceStat(name, int(m.sum()), int(y[m].sum()), safe_auc(y[m], oof[m]),
                              float(accuracy_score(y[m], (oof[m] >= 0.5).astype(int)))))
    return out


def run(conn, embed_model: str, like_precision: float = 0.9, pass_precision: float = 0.95) -> Report:
    """Evaluate the current ratings. Raises ValueError if there are too few to say anything."""
    ids, X, y = model.training_data(conn, embed_model)
    if len(y) == 0:
        raise ValueError("No rated profiles with embeddings yet. Rate some profiles, then run: datebot embed")
    check_enough(y)
    oof = oof_scores(X, y)
    head = headline(y, oof)
    drift = drift_check(X, y)
    warnings = []
    if len(y) < 50:
        warnings.append(f"Only {len(y)} ratings: treat every number below as rough (aim for 100+).")
    if int(y.sum()) < 10:
        warnings.append(f"Only {int(y.sum())} likes: the model has very little to learn your taste from.")
    if head.auc is not None and head.auc < 0.6:
        warnings.append("The model is barely beating chance. Rate more, or your taste may not show in photos/bios.")
    if drift is not None and drift.drifting:
        warnings.append("Your newest ratings are much harder to predict than older ones: your taste may have "
                        "shifted. Consider 'datebot reset-ratings' or rating more recent profiles.")
    return Report(
        n=len(y), likes=int(y.sum()), passes=len(y) - int(y.sum()), headline=head,
        ablation=ablation(X, y), drift=drift, curve=learning_curve(X, y),
        thresholds=suggest_thresholds(y, oof, like_precision, pass_precision),
        surprises=surprises(ids, y, oof), warnings=warnings, by_source=by_source(conn, ids, y, oof),
    )


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.0%}"


def _f(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.2f}"


def format_report(r: Report, describe=lambda pid: f"#{pid}") -> str:
    """Plain-text report. `describe(profile_id)` renders a profile for the surprise list."""
    h = r.headline
    lines = [
        f"Ratings: {r.n} ({r.likes} likes, {r.passes} passes; you like {r.likes / r.n:.0%} of profiles)",
        "",
        "How well does it predict profiles it has not seen? (cross-validated)",
        f"  ROC AUC            {_f(h.auc)}   {verdict(h.auc)}   (0.50 = coin flip, 1.00 = perfect)",
        f"  accuracy           {_pct(h.accuracy)}   (always guessing the common answer: {_pct(h.majority_baseline)})",
        f"  balanced accuracy  {_pct(h.balanced_accuracy)}   (50% = chance)",
        "",
        "What is it learning from? (AUC)",
        *[f"  {name:<14} {_f(auc)}" + ("   (no bios captured yet)" if auc is None else "")
          for name, auc in r.ablation.items()],
    ]
    if r.by_source:
        lines += ["", "By source (one pooled model; each app's profiles scored out-of-fold)"]
        for st in r.by_source:
            note = "   (too few of one answer for an AUC)" if st.auc is None else ""
            lines.append(f"  {st.source:<8} n={st.n:<4} likes {st.likes / st.n:.0%}  AUC {_f(st.auc)}  "
                         f"accuracy {_pct(st.accuracy)}{note}")
    lines += ["", "Does more rating help? (AUC when trained on n ratings, tested on the rest)"]
    if r.curve:
        lines += [f"  n={p.n_train:<4} {_f(p.auc_mean)}  ±{_f(p.auc_std)}" for p in r.curve]
        first, last = next((p for p in r.curve if p.auc_mean is not None), None), next(
            (p for p in reversed(r.curve) if p.auc_mean is not None), None)
        if first and last and last.n_train > first.n_train:
            gain = last.auc_mean - first.auc_mean
            lines.append("  -> still improving: keep rating" if gain > 0.03 else "  -> flattening out")
    else:
        lines.append("  (needs about 20+ ratings)")
    if r.drift:
        lines += ["", "Has your taste shifted? (train on your oldest ratings, test on your newest)",
                  f"  newest {r.drift.n_test} ratings: AUC {_f(r.drift.future_auc)}  "
                  f"vs random split {_f(r.drift.random_auc)}" + ("   <- drifting" if r.drift.drifting else "")]
    t = r.thresholds
    lines += ["", f"Auto-swipe bands (to act automatically when sure, review the rest) - estimated, optimistic:"]
    if t.like_at is None:
        lines.append("  auto-like: not reachable at the target precision yet")
    else:
        lines.append(f"  auto-like at score >= {t.like_at:.0f}: {t.like_count} profiles, {_pct(t.like_precision)} you liked")
    if t.pass_at is None:
        lines.append("  auto-pass: not reachable at the target precision yet")
    else:
        lines.append(f"  auto-pass at score <= {t.pass_at:.0f}: {t.pass_count} profiles, {_pct(t.pass_precision)} "
                     f"you passed; would have lost {t.likes_lost} of your {t.total_likes} likes")
    lines.append(f"  decided automatically: {_pct(t.coverage)} of profiles; the rest go to your review queue")
    if r.surprises:
        lines += ["", "Biggest surprises (model most wrong; blind spots, or ratings to reconsider)"]
        lines += [f"  you {'liked' if s.liked else 'passed'}, scored {s.score:.0f}: {describe(s.profile_id)}"
                  for s in r.surprises]
    if r.warnings:
        lines += ["", *[f"! {w}" for w in r.warnings]]
    return "\n".join(lines)
