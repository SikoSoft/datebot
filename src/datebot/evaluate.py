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
    """Split features laid out as [photo mean, photo max, text, has_text] into ablation sets.

    'text' is None when no rated profile has any text, since that comparison would be meaningless.
    """
    d = (X.shape[1] - 1) // 3
    photos, text_cols = X[:, : 2 * d], X[:, 2 * d :]
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
