from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

from datebot import config, db

MIN_PER_CLASS = 3  # need at least this many likes and passes to train


def featurize(embeddings: np.ndarray) -> np.ndarray:
    """Collapse a profile's (n_photos, dim) embeddings into one feature vector.

    Mean captures the overall look; max keeps standout photos from being averaged away.
    """
    if embeddings.size == 0:
        raise ValueError("profile has no embeddings")
    return np.concatenate([embeddings.mean(axis=0), embeddings.max(axis=0)])


@dataclass
class Trained:
    clf: LogisticRegression
    embed_model: str
    n_samples: int
    n_likes: int
    cv_accuracy: float | None
    trained_at: str


def _path(name: str) -> Path:
    return config.models_dir() / f"{name}.joblib"


def training_set(conn: sqlite3.Connection, embed_model: str) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for pid, liked in db.latest_ratings(conn).items():
        emb = db.profile_embeddings(conn, pid, embed_model)
        if emb.size == 0:
            continue
        X.append(featurize(emb))
        y.append(int(liked))
    if not X:
        return np.empty((0, 0)), np.empty((0,), dtype=int)
    return np.stack(X), np.array(y)


def train(conn: sqlite3.Connection, embed_model: str, name: str = "default") -> Trained:
    X, y = training_set(conn, embed_model)
    likes = int(y.sum())
    passes = len(y) - likes
    if min(likes, passes) < MIN_PER_CLASS:
        raise ValueError(
            f"Need at least {MIN_PER_CLASS} likes and {MIN_PER_CLASS} passes with embeddings "
            f"(have {likes} likes, {passes} passes)."
        )
    clf = LogisticRegression(C=0.5, class_weight="balanced", max_iter=2000)
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
    return joblib.load(path) if path.exists() else None


def reset(name: str = "default") -> bool:
    """Forget a trained model. Ratings are kept; retrain any time."""
    path = _path(name)
    if path.exists():
        path.unlink()
        return True
    return False


def list_models() -> list[str]:
    return sorted(p.stem for p in config.models_dir().glob("*.joblib"))


def score(trained: Trained, embeddings: np.ndarray) -> float:
    """Score a profile 0-100 (probability you'd like them)."""
    p = trained.clf.predict_proba(featurize(embeddings).reshape(1, -1))[0, 1]
    return float(p * 100)


def score_profile(conn: sqlite3.Connection, trained: Trained, profile_id: int) -> float | None:
    emb = db.profile_embeddings(conn, profile_id, trained.embed_model)
    return None if emb.size == 0 else score(trained, emb)


def most_uncertain(conn: sqlite3.Connection, trained: Trained, limit: int = 10) -> list[tuple[int, float]]:
    """Unrated profiles whose score is closest to 50, i.e. the most informative to rate next."""
    scored = []
    for pid in db.unrated_profile_ids(conn):
        s = score_profile(conn, trained, pid)
        if s is not None:
            scored.append((pid, s))
    scored.sort(key=lambda t: abs(t[1] - 50))
    return scored[:limit]
