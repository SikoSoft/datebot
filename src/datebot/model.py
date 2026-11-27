from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
from sklearn.model_selection import StratifiedKFold, cross_val_score

from datebot import config, db, filters as hard_filters

MIN_PER_CLASS = 3  # need at least this many likes and passes to train
FEATURE_VERSION = 4  # bump when featurize() changes; older saved models are ignored and must be retrained


def featurize(embeddings: np.ndarray, text_embedding: np.ndarray | None = None) -> np.ndarray:
    """Collapse a profile into one feature vector:
    [photo mean, photo max, lead photo, text mean, text max, has_text].

    Mean captures the overall look; max keeps standout photos from being averaged away; the lead
    (first) photo gets its own block so the model can learn how much the photo you see first matters
    versus the rest. `text_embedding` is the stored [mean, max] of the profile's text lines. The text
    blocks are zeros (with has_text = 0) when there is no text, so photo-only profiles still work.
    """
    if embeddings.size == 0:
        raise ValueError("profile has no embeddings")
    dim = embeddings.shape[1]
    if text_embedding is None:
        text, has_text = np.zeros(2 * dim, dtype=np.float32), 0.0
    else:
        text, has_text = np.asarray(text_embedding, dtype=np.float32), 1.0
        if text.shape != (2 * dim,):
            raise ValueError(f"text embedding has {text.shape} values, expected {2 * dim}")
    return np.concatenate([embeddings.mean(axis=0), embeddings.max(axis=0), embeddings[0], text, [has_text]]).astype(np.float32)


class PhotoAverageLR(ClassifierMixin, BaseEstimator):
    """Logistic regression on the L2-normalised mean photo embedding only.

    Measured with cross-validation: the extra blocks of `featurize` (best photo, lead photo, text) did not
    improve AUC (text alone is ~0.58), so the classifier ignores them. They stay in the feature vector so
    `datebot eval` can keep comparing photos with text, and so nothing stored is thrown away.
    """

    def __init__(self, C: float = 4.0):
        self.C = C

    def _photo_average(self, X: np.ndarray) -> np.ndarray:
        d = (X.shape[1] - 1) // 5  # featurize layout: 5 blocks of `d` plus the has_text flag
        return normalize(X[:, :d])

    def fit(self, X, y):
        self.lr_ = LogisticRegression(C=self.C, class_weight="balanced", max_iter=3000).fit(self._photo_average(X), y)
        self.classes_ = self.lr_.classes_
        return self

    def decision_function(self, X):
        return self.lr_.decision_function(self._photo_average(X))

    def predict_proba(self, X):
        return self.lr_.predict_proba(self._photo_average(X))

    def predict(self, X):
        return self.lr_.predict(self._photo_average(X))


@dataclass
class Trained:
    clf: PhotoAverageLR
    embed_model: str
    n_samples: int
    n_likes: int
    cv_accuracy: float | None
    trained_at: str
    feature_version: int = FEATURE_VERSION


def _path(name: str) -> Path:
    return config.models_dir() / f"{name}.joblib"


def training_data(conn: sqlite3.Connection, embed_model: str) -> tuple[list[int], np.ndarray, np.ndarray]:
    """(profile ids, features, labels) for every rated profile that has embeddings, oldest rating first.

    Profiles that trip a hard deal-breaker are left out (not deleted): a pass caused by a rule says
    nothing about how attractive they are, so it must not teach the model your type."""
    ids, X, y = [], [], []
    f = hard_filters.load()
    for pid, liked in db.latest_ratings(conn).items():
        if not f.is_empty() and hard_filters.check(db.get_profile(conn, pid), f) is not None:
            continue
        emb = db.profile_embeddings(conn, pid, embed_model)
        if emb.size == 0:
            continue
        ids.append(pid)
        X.append(featurize(emb, db.text_embedding(conn, pid, embed_model)))
        y.append(int(liked))
    if not X:
        return [], np.empty((0, 0)), np.empty((0,), dtype=int)
    return ids, np.stack(X), np.array(y)


def training_set(conn: sqlite3.Connection, embed_model: str) -> tuple[np.ndarray, np.ndarray]:
    _, X, y = training_data(conn, embed_model)
    return X, y


def make_classifier() -> PhotoAverageLR:
    """The classifier used for training and for evaluation, so both measure the same thing."""
    return PhotoAverageLR()


def train(conn: sqlite3.Connection, embed_model: str, name: str = "default") -> Trained:
    X, y = training_set(conn, embed_model)
    likes = int(y.sum())
    passes = len(y) - likes
    if min(likes, passes) < MIN_PER_CLASS:
        raise ValueError(
            f"Need at least {MIN_PER_CLASS} likes and {MIN_PER_CLASS} passes with embeddings "
            f"(have {likes} likes, {passes} passes)."
        )
    clf = make_classifier()
    cv = None
    folds = min(5, likes, passes)
    if folds >= 2:
        cv = float(cross_val_score(clf, X, y, cv=StratifiedKFold(folds, shuffle=True, random_state=0)).mean())
    clf.fit(X, y)
    trained = Trained(clf, embed_model, len(y), likes, cv, datetime.now(timezone.utc).isoformat())
    joblib.dump(trained, _path(name))
    return trained


def load(name: str = "default") -> Trained | None:
    path = _path(name)
    if not path.exists():
        return None
    trained = joblib.load(path)
    return trained if getattr(trained, "feature_version", 1) == FEATURE_VERSION else None  # stale: retrain


def reset(name: str = "default") -> bool:
    """Forget a trained model. Ratings are kept; retrain any time."""
    path = _path(name)
    if path.exists():
        path.unlink()
        return True
    return False


def list_models() -> list[str]:
    return sorted(p.stem for p in config.models_dir().glob("*.joblib"))


def score(trained: Trained, embeddings: np.ndarray, text_embedding: np.ndarray | None = None) -> float:
    """Score a profile 0-100 (probability you'd like them)."""
    p = trained.clf.predict_proba(featurize(embeddings, text_embedding).reshape(1, -1))[0, 1]
    return float(p * 100)


def score_profile(conn: sqlite3.Connection, trained: Trained, profile_id: int) -> float | None:
    emb = db.profile_embeddings(conn, profile_id, trained.embed_model)
    if emb.size == 0:
        return None
    return score(trained, emb, db.text_embedding(conn, profile_id, trained.embed_model))


BLOCKS = ["photo average"]


def explain_blocks(trained: Trained, embeddings: np.ndarray, text_embedding: np.ndarray | None = None) -> dict:
    """Why a profile got its score. The classifier is linear in the normalised photo average, so the log-odds
    are exactly the intercept plus one photo-average contribution."""
    x = trained.clf._photo_average(featurize(embeddings, text_embedding).reshape(1, -1))[0]
    w = trained.clf.lr_.coef_[0]
    return {"intercept": float(trained.clf.lr_.intercept_[0]),
            "factors": [{"name": BLOCKS[0], "logit": float(w @ x)}]}


def most_uncertain(
    conn: sqlite3.Connection, trained: Trained, limit: int = 10, candidates: list[int] | None = None
) -> list[tuple[int, float]]:
    """Profiles whose score is closest to 50, i.e. the most informative to rate next.

    `candidates` restricts the pool (e.g. to profiles passing the hard filters); default is all unrated.
    """
    scored = []
    for pid in db.unrated_profile_ids(conn) if candidates is None else candidates:
        s = score_profile(conn, trained, pid)
        if s is not None:
            scored.append((pid, s))
    scored.sort(key=lambda t: abs(t[1] - 50))
    return scored[:limit]
