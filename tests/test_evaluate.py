import numpy as np
import pytest

from datebot import evaluate, model


def make_data(n=60, signal=1.0, d=8, seed=0, text_signal=0.0, text_frac=1.0):
    """Features in the real layout [mean(d), max(d), text(d), has_text(1)] with a controllable signal."""
    rng = np.random.default_rng(seed)
    y = np.array([0, 1] * (n // 2))
    rng.shuffle(y)
    sign = (2 * y - 1)[:, None]
    photo = rng.normal(0, 1, (n, d)) + signal * sign
    X = []
    for i in range(n):
        has = rng.random() < text_frac
        text = rng.normal(0, 1, d) + text_signal * sign[i] if has else None
        X.append(model.featurize(np.stack([photo[i], photo[i]]), text))
    return np.stack(X), y


def test_oof_scores_cover_every_profile_and_beat_chance_with_signal():
    X, y = make_data(signal=1.5)
    oof = evaluate.oof_scores(X, y)
    assert oof.shape == y.shape and ((0 <= oof) & (oof <= 1)).all()
    h = evaluate.headline(y, oof)
    assert h.auc > 0.9 and h.accuracy > h.majority_baseline


def test_pure_noise_is_near_chance():
    X, y = make_data(signal=0.0, n=80)
    assert 0.3 < evaluate.headline(y, evaluate.oof_scores(X, y)).auc < 0.7


def test_refuses_too_few_labels():
    X, y = make_data(n=10)
    y[:] = 0
    y[0] = 1
    with pytest.raises(ValueError, match="at least"):
        evaluate.oof_scores(X, y)


def test_safe_auc_single_class_is_none():
    assert evaluate.safe_auc(np.zeros(5, dtype=int), np.random.rand(5)) is None


def test_ablation_attributes_signal_to_the_right_block():
    X, y = make_data(signal=0.0, text_signal=1.5, n=80)  # only the bio is informative
    a = evaluate.ablation(X, y)
    assert a["text only"] > 0.85 and a["photos + text"] > 0.85 and a["photos only"] < 0.7

    X, y = make_data(signal=1.5, text_signal=0.0, n=80)  # only photos are informative
    a = evaluate.ablation(X, y)
    assert a["photos only"] > 0.85 and a["text only"] < 0.7


def test_ablation_skips_text_when_nobody_has_a_bio():
    X, y = make_data(text_frac=0.0)
    assert evaluate.ablation(X, y)["text only"] is None


def test_learning_curve_rises_with_more_data():
    X, y = make_data(n=240, signal=0.12, d=32, seed=3)  # weak signal: needs data to find
    curve = evaluate.learning_curve(X, y)
    assert len(curve) >= 4
    assert [p.n_train for p in curve] == sorted(p.n_train for p in curve)
    assert curve[0].n_train == evaluate.MIN_CURVE_TRAIN
    assert curve[-1].auc_mean > curve[0].auc_mean + 0.1
    assert curve[-1].n_train <= int(240 * 0.8)


def test_learning_curve_empty_when_too_little_data():
    X, y = make_data(n=12)
    assert evaluate.learning_curve(X, y) == []
    X, y = make_data(n=18)  # 80% = 14 < 10 + 5
    assert evaluate.learning_curve(X, y) == []


def test_drift_check_flags_a_taste_reversal():
    X, y = make_data(n=200, signal=1.5, seed=5)
    stable = evaluate.drift_check(X, y)
    assert not stable.drifting and stable.future_auc > 0.85

    y_flipped = y.copy()
    y_flipped[140:] = 1 - y_flipped[140:]  # newest 30% of ratings follow the opposite taste
    drift = evaluate.drift_check(X, y_flipped)
    assert drift.drifting and drift.future_auc < 0.3
    assert (drift.n_train, drift.n_test) == (140, 60)


def test_drift_check_needs_enough_recent_data():
    X, y = make_data(n=12)
    assert evaluate.drift_check(X, y) is None


def test_thresholds_hit_targets_on_separable_data():
    X, y = make_data(n=200, signal=1.5, seed=7)
    oof = evaluate.oof_scores(X, y)
    t = evaluate.suggest_thresholds(y, oof)
    assert t.like_at is not None and t.pass_at is not None and t.pass_at < t.like_at
    assert t.like_precision >= 0.9 and t.pass_precision >= 0.95
    assert t.like_count >= 10 and t.pass_count >= 10
    assert 0.5 < t.coverage <= 1.0
    assert t.likes_lost == int(y[oof * 100 <= t.pass_at].sum()) and t.total_likes == int(y.sum())


def test_thresholds_unreachable_on_noise():
    X, y = make_data(n=80, signal=0.0, seed=0)
    t = evaluate.suggest_thresholds(y, evaluate.oof_scores(X, y))
    assert t.like_at is None and t.pass_at is None and t.coverage == 0.0


def test_stricter_targets_lower_coverage():
    X, y = make_data(n=200, signal=0.6, seed=2, d=16)
    oof = evaluate.oof_scores(X, y)
    loose = evaluate.suggest_thresholds(y, oof, like_precision=0.8, pass_precision=0.8)
    strict = evaluate.suggest_thresholds(y, oof, like_precision=0.95, pass_precision=0.95)
    assert strict.coverage <= loose.coverage


def test_surprises_are_the_most_wrong_predictions():
    y = np.array([1, 1, 1, 0, 0, 0])
    oof = np.array([0.9, 0.1, 0.5, 0.8, 0.2, 0.05])
    ids = [10, 11, 12, 13, 14, 15]
    got = evaluate.surprises(ids, y, oof, k=1)
    assert [(s.profile_id, s.liked) for s in got] == [(11, True), (13, False)]
    assert got[0].score == pytest.approx(10) and got[1].score == pytest.approx(80)
    assert len(evaluate.surprises(ids, y, oof, k=5)) == 6


def seed_db(path, n):
    """DB with fake embeddings (liked profiles cluster apart from passed ones) and some bios."""
    from datebot import db

    conn = db.connect(str(path))
    rng = np.random.default_rng(0)
    for i in range(n):
        liked = i % 2 == 0
        center = (1 if liked else -1) * np.ones(8)
        pid = db.add_profile(conn, "manual", [f"{i}a.jpg", f"{i}b.jpg"], name=f"P{i}", bio="hello" if i % 3 == 0 else None)
        for photo_id, _ in db.photos_missing_embeddings(conn, "fake"):
            if conn.execute("SELECT profile_id FROM photos WHERE id=?", (photo_id,)).fetchone()[0] == pid:
                db.set_embedding(conn, photo_id, center + rng.normal(0, 1.5, 8), "fake")
        if pid % 3 == 1:
            db.set_text_embedding(conn, pid, center + rng.normal(0, 1.5, 8), "fake")
        db.add_rating(conn, pid, liked)
    return conn


@pytest.fixture
def seeded(tmp_path):
    conn = seed_db(tmp_path / "t.db", 60)
    yield conn
    conn.close()


def test_run_builds_a_complete_report(seeded):
    r = evaluate.run(seeded, "fake")
    assert (r.n, r.likes, r.passes) == (60, 30, 30)
    assert r.headline.auc > 0.8
    assert set(r.ablation) == {"photos only", "text only", "photos + text"}
    assert r.curve and r.drift and not r.drift.drifting
    assert len(r.surprises) == 10
    assert r.warnings == []  # 60 clean ratings with a clear signal: nothing to warn about


def test_small_rating_sets_are_flagged_as_noisy(tmp_path):
    r = evaluate.run(seed_db(tmp_path / "s.db", 16), "fake")
    assert any("Only 16 ratings" in w for w in r.warnings)
    assert r.curve == []  # too few for a learning curve
    assert "needs about 20+ ratings" in evaluate.format_report(r)


def test_format_report_is_readable_and_uses_describe(seeded):
    r = evaluate.run(seeded, "fake")
    text = evaluate.format_report(r, describe=lambda pid: f"profile-{pid}")
    for needle in ("Ratings: 60 (30 likes, 30 passes", "ROC AUC", "What is it learning from", "Does more rating help",
                   "Auto-swipe bands", "Biggest surprises", "profile-"):
        assert needle in text


def test_run_errors_are_actionable(tmp_path):
    from datebot import db

    conn = db.connect(str(tmp_path / "e.db"))
    with pytest.raises(ValueError, match="No rated profiles"):
        evaluate.run(conn, "fake")
    pid = db.add_profile(conn, "manual", ["a.jpg"])
    db.add_rating(conn, pid, True)
    db.set_embedding(conn, conn.execute("SELECT id FROM photos").fetchone()[0], np.ones(8), "fake")
    with pytest.raises(ValueError, match="at least"):
        evaluate.run(conn, "fake")


def test_verdict_bands():
    assert [evaluate.verdict(a) for a in (0.5, 0.65, 0.75, 0.85, 0.95, None)] == [
        "barely better than guessing", "weak", "decent", "good", "strong", "undefined"]


def test_by_source_splits_the_pooled_report(seeded):
    seeded.execute("UPDATE profiles SET source = 'badoo' WHERE id % 3 = 0")
    seeded.commit()
    r = evaluate.run(seeded, "fake")
    assert {s.source for s in r.by_source} == {"manual", "badoo"}
    assert sum(s.n for s in r.by_source) == r.n and sum(s.likes for s in r.by_source) == r.likes
    assert r.by_source[0].n >= r.by_source[1].n
    text = evaluate.format_report(r)
    assert "By source" in text and "badoo" in text and "manual" in text
