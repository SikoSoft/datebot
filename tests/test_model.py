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
