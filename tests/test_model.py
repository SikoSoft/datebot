import numpy as np
import pytest

from datebot import config, db, model

EMBED = "fake"


@pytest.fixture
def conn(tmp_path, monkeypatch):
    monkeypatch.setenv("DATEBOT_HOME", str(tmp_path))
    c = db.connect(str(tmp_path / "t.db"))
    yield c
    c.close()


def add(conn, center, rng, liked=None):
    pid = db.add_profile(conn, "manual", ["a.jpg", "b.jpg"])
    for photo_id, _ in db.photos_missing_embeddings(conn, EMBED):
        owner = conn.execute("SELECT profile_id FROM photos WHERE id=?", (photo_id,)).fetchone()[0]
        if owner == pid:
            db.set_embedding(conn, photo_id, center + rng.normal(0, 0.3, 8), EMBED)
    if liked is not None:
        db.add_rating(conn, pid, liked)
    return pid


def seed(conn, n=12):
    rng = np.random.default_rng(0)
    good, bad = np.ones(8), -np.ones(8)
    for _ in range(n):
        add(conn, good, rng, True)
        add(conn, bad, rng, False)
    return rng, good, bad


def test_train_refuses_without_enough_labels(conn):
    with pytest.raises(ValueError):
        model.train(conn, EMBED)


def test_learns_and_scores_in_range(conn):
    rng, good, bad = seed(conn)
    trained = model.train(conn, EMBED)
    assert trained.cv_accuracy > 0.9
    hi = model.score(trained, np.stack([good + rng.normal(0, 0.3, 8)] * 2))
    lo = model.score(trained, np.stack([bad + rng.normal(0, 0.3, 8)] * 2))
    assert 0 <= lo < 20 and 80 < hi <= 100


def test_save_load_and_reset(conn):
    seed(conn)
    model.train(conn, EMBED, name="v1")
    assert model.list_models() == ["v1"]
    assert model.load("v1").n_samples == 24
    assert model.reset("v1") is True
    assert model.load("v1") is None
    assert model.reset("v1") is False


def test_most_uncertain_prefers_midpoint(conn):
    rng, good, bad = seed(conn)
    clear = add(conn, good, rng)
    ambiguous = add(conn, np.zeros(8), rng)
    trained = model.train(conn, EMBED)
    ranked = [pid for pid, _ in model.most_uncertain(conn, trained)]
    assert ranked[0] == ambiguous and ranked.index(clear) > 0


def add_with_text(conn, text_vec, rng, liked):
    """Profile whose photos are pure noise, so only the bio carries signal."""
    pid = add(conn, np.zeros(8), rng, liked)
    conn.execute("UPDATE photos SET embedding = ? WHERE profile_id = ?",
                 (rng.normal(0, 1, 8).astype(np.float32).tobytes(), pid))
    db.set_text_embedding(conn, pid, text_vec + rng.normal(0, 0.2, 8), EMBED)
    return pid


def test_learns_from_bio_text_when_photos_are_noise(conn):
    rng = np.random.default_rng(1)
    liked_bio, passed_bio = np.ones(8), -np.ones(8)
    for _ in range(15):
        add_with_text(conn, liked_bio, rng, True)
        add_with_text(conn, passed_bio, rng, False)
    trained = model.train(conn, EMBED)
    assert trained.cv_accuracy > 0.9
    hi = add_with_text(conn, liked_bio, rng, None)
    lo = add_with_text(conn, passed_bio, rng, None)
    assert model.score_profile(conn, trained, hi) > 80 > 20 > model.score_profile(conn, trained, lo)


def test_profiles_without_bio_still_score(conn):
    rng, good, _ = seed(conn)
    trained = model.train(conn, EMBED)
    assert 0 <= model.score(trained, np.stack([good, good])) <= 100


def test_stale_model_is_ignored(conn):
    seed(conn)
    trained = model.train(conn, EMBED, name="old")
    trained.feature_version = 1
    import joblib
    joblib.dump(trained, model._path("old"))
    assert model.load("old") is None
