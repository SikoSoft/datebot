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
    db.set_text_embedding(conn, pid, np.tile(text_vec, 2) + rng.normal(0, 0.2, 16), EMBED)
    return pid


def test_classifier_reads_only_the_photo_average_not_text(conn):
    """Cross-validation showed text/best/lead blocks do not help (text alone AUC ~0.58), so the model
    ignores them: a bio that perfectly predicts the label must not move the score."""
    rng = np.random.default_rng(1)
    liked_bio, passed_bio = np.ones(8), -np.ones(8)
    for _ in range(15):
        add_with_text(conn, liked_bio, rng, True)
        add_with_text(conn, passed_bio, rng, False)
    trained = model.train(conn, EMBED)
    assert trained.cv_accuracy < 0.8  # photos are noise, so nothing to learn without the text
    hi = add_with_text(conn, liked_bio, rng, None)
    emb = db.profile_embeddings(conn, hi, EMBED)
    assert model.score(trained, emb, db.text_embedding(conn, hi, EMBED)) == model.score(trained, emb, None)


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


def test_featurize_layout_has_a_lead_photo_block():
    photos = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 3.0]], dtype=np.float32)
    f = model.featurize(photos)
    assert f.shape == (5 * 2 + 1,)
    assert list(f[0:2]) == pytest.approx([1 / 3, 4 / 3])  # mean
    assert list(f[2:4]) == [1.0, 3.0]                    # max
    assert list(f[4:6]) == [1.0, 0.0]                    # lead photo = first by position
    assert not f[6:].any()                               # no text: zeros and has_text = 0


def test_featurize_rejects_text_of_the_wrong_size():
    with pytest.raises(ValueError):
        model.featurize(np.ones((2, 4), dtype=np.float32), np.ones(4, dtype=np.float32))


def test_lead_photo_is_what_the_model_learns_when_other_photos_are_noise(conn):
    rng = np.random.default_rng(3)
    for liked in (True, False) * 15:
        pid = db.add_profile(conn, "manual", ["a.jpg", "b.jpg", "c.jpg"])
        lead = (np.ones(8) if liked else -np.ones(8)) + rng.normal(0, 0.2, 8)
        for photo_id, vec in zip([r["id"] for r in conn.execute("SELECT id FROM photos WHERE profile_id = ? ORDER BY position", (pid,))],
                                 [lead, rng.normal(0, 1, 8), rng.normal(0, 1, 8)]):
            db.set_embedding(conn, photo_id, vec, EMBED)
        db.add_rating(conn, pid, liked)
    assert model.train(conn, EMBED).cv_accuracy > 0.9


def test_explain_blocks_sum_to_the_score(conn):
    rng, good, _ = seed(conn)
    trained = model.train(conn, EMBED)
    pid = add(conn, good, rng)
    emb = db.profile_embeddings(conn, pid, EMBED)
    out = model.explain_blocks(trained, emb, None)
    assert [f["name"] for f in out["factors"]] == model.BLOCKS
    logit = out["intercept"] + sum(f["logit"] for f in out["factors"])
    assert 1 / (1 + np.exp(-logit)) * 100 == pytest.approx(model.score(trained, emb), abs=1e-3)


def test_ratings_behind_counts_human_ratings_since_training(tmp_path):
    from datebot import db, model

    conn = db.connect(str(tmp_path / "b.db"))
    pids = [db.add_profile(conn, "manual", f"p{i}") for i in range(4)]
    assert model.ratings_behind(conn, None) is None
    trained = model.Trained(None, "m", 0, 0, None, "2026-01-01T12:00:00+00:00")
    conn.execute("INSERT INTO ratings (profile_id, liked, origin, created_at) VALUES (?,1,'manual','2026-01-01 11:59:59')", (pids[0],))
    assert model.ratings_behind(conn, trained) == 0
    conn.execute("INSERT INTO ratings (profile_id, liked, origin, created_at) VALUES (?,1,'manual','2026-01-01 12:05:00')", (pids[0],))  # re-rated
    conn.execute("INSERT INTO ratings (profile_id, liked, origin, created_at) VALUES (?,0,'app','2026-01-01 12:06:00')", (pids[1],))
    conn.execute("INSERT INTO ratings (profile_id, liked, origin, created_at) VALUES (?,0,'auto','2026-01-01 12:07:00')", (pids[2],))  # bot: ignored
    assert model.ratings_behind(conn, trained) == 2
