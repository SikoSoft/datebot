import json

import numpy as np
import pytest

from datebot import behavior, behavior_model as bm, db


@pytest.fixture
def conn(tmp_path, monkeypatch):
    monkeypatch.setenv("DATEBOT_HOME", str(tmp_path))
    return db.connect(str(tmp_path / "t.db"))


def add(conn, sid, pid, dwell_s, liked=True, origin="app", at=None, events=(), source="hinge"):
    raw = {"start": 0.0, "decided": dwell_s * 1000.0, "basis": "prev_swipe", "events": list(events)}
    db.add_interaction(conn, sid, source, origin, at if at is not None else 1000.0 + pid, "timing", raw,
                       behavior.derive(raw), behavior.FEATURE_VERSION, profile_id=pid, liked=liked)


def seeded(conn, scores_to_dwell):
    """One profile + one interaction each; returns the fake score map."""
    sid = db.start_session(conn, "hinge", "iphone-proxy", 0.0)
    scores = {}
    for i, (score, dwell) in enumerate(scores_to_dwell, start=1):
        pid = db.add_profile(conn, "hinge", [], external_id=str(i))
        scores[pid] = score
        add(conn, sid, pid, dwell, liked=score >= 50)
    return scores


def train(conn, monkeypatch, scores, **kw):
    monkeypatch.setattr(bm, "preference_scores", lambda *a, **k: scores)
    return bm.train(conn, "embed", **kw)


def test_bands_hold_the_dwell_of_profiles_in_that_score_range(conn, monkeypatch):
    data = [(80 + i % 20, 10 + i % 3) for i in range(12)] + [(10 + i, 3 + i % 2) for i in range(12)]
    m = train(conn, monkeypatch, seeded(conn, data))
    top, low = m.bands[-1], m.bands[0]
    assert (top.n, low.n) == (12, 12)
    assert 10 <= top.stats["dwell_ms"].p50 / 1000 <= 12 and low.stats["dwell_ms"].p50 / 1000 <= 4
    assert top.like_rate == 1.0 and low.like_rate == 0.0 and m.bands[1].n == 0


def test_a_score_of_exactly_100_belongs_to_the_top_band_and_the_edges_are_lower_inclusive(conn, monkeypatch):
    m = train(conn, monkeypatch, seeded(conn, [(100, 5)] * 5 + [(70, 6)] * 5))
    assert [b.n for b in m.bands] == [0, 0, 0, 10]
    assert m.bands[-1].lo == 70


def test_breaks_are_counted_but_not_treated_as_looking(conn, monkeypatch):
    m = train(conn, monkeypatch, seeded(conn, [(90, 8)] * 9 + [(90, 3600)]))
    top = m.bands[-1]
    assert top.n == 10 and top.breaks == 1 and top.stats["dwell_ms"].n == 9 and top.stats["dwell_ms"].p90 == 8000


def test_bot_swipes_are_never_learned_from(conn, monkeypatch):
    scores = seeded(conn, [(90, 8)] * 10)
    sid = db.start_session(conn, "tinder", "desktop-chrome", 0.0)
    pid = next(iter(scores))
    for _ in range(5):
        add(conn, sid, pid, 0.4, origin="auto")  # fast, machine-like swipes
    m = train(conn, monkeypatch, scores)
    assert m.n_interactions == 10 and m.overall.stats["dwell_ms"].p10 >= 8000


def test_source_filter(conn, monkeypatch):
    scores = seeded(conn, [(90, 8)] * 10)
    sid = db.start_session(conn, "tinder", "desktop-chrome", 0.0)
    for i in range(10):
        pid = db.add_profile(conn, "tinder", [], external_id=f"t{i}")
        scores[pid] = 90
        add(conn, sid, pid, 2, source="tinder")
    assert train(conn, monkeypatch, scores, source="tinder").overall.stats["dwell_ms"].p50 == 2000
    assert train(conn, monkeypatch, scores).n_interactions == 20


def test_interactions_without_a_score_are_skipped_and_too_few_raises(conn, monkeypatch):
    scores = seeded(conn, [(90, 8)] * 6)
    pid = db.add_profile(conn, "hinge", [], external_id="unscored")
    add(conn, db.start_session(conn, "hinge", "", 0.0), pid, 5)
    with pytest.raises(ValueError, match="at least"):
        train(conn, monkeypatch, scores)


def test_sampler_draws_around_the_bands_own_dwell(conn, monkeypatch):
    data = [(90, 10 + (i % 5) - 2) for i in range(40)] + [(10, 2 + (i % 3) * 0.5) for i in range(40)]
    m = train(conn, monkeypatch, seeded(conn, data))
    rng = np.random.default_rng(0)
    high = [bm.sample_dwell_ms(m, 85, rng) for _ in range(300)]
    low = [bm.sample_dwell_ms(m, 10, rng) for _ in range(300)]
    assert 8000 < np.median(high) < 12500 and 1500 < np.median(low) < 3500


def test_a_thin_band_falls_back_to_all_scores(conn, monkeypatch):
    m = train(conn, monkeypatch, seeded(conn, [(90, 10)] * 3 + [(10, 4)] * 20))
    assert bm.band_for(m, 90) is m.overall and bm.band_for(m, 10) is m.bands[0]


def test_stale_features_are_rederived_from_raw_before_training(conn, monkeypatch):
    scores = seeded(conn, [(90, 8)] * 10)
    conn.execute("UPDATE interactions SET features = '{\"dwell_ms\": 1}', feature_version = 0")
    assert bm.refresh_features(conn) == 10
    assert json.loads(db.interactions(conn)[0]["features"])["dwell_ms"] == 8000
    assert train(conn, monkeypatch, scores).overall.stats["dwell_ms"].p50 == 8000


def test_rhythm_splits_sessions_by_breaks(conn, monkeypatch):
    sid = db.start_session(conn, "hinge", "", 0.0)
    scores = {}
    for i, t in enumerate([0, 10, 20, 30, 40, 50, 60, 3000, 3010, 3020, 3030, 3040]):  # one long break
        pid = db.add_profile(conn, "hinge", [], external_id=str(i))
        scores[pid] = 90
        add(conn, sid, pid, 10, at=1_000_000 + t)
    r = train(conn, monkeypatch, scores).rhythm
    assert r["breaks"] == 1 and r["gap_ms"].p50 == 10_000 and r["swipes_per_active_min"] == 6.0


def test_save_load_reset_and_report(conn, monkeypatch):
    train(conn, monkeypatch, seeded(conn, [(90, 8)] * 10), name="mine")
    assert bm.list_models() == ["mine"] and bm.load("mine").n_interactions == 10
    text = bm.format_report(bm.load("mine"))
    assert "70-100" in text and "8.0s" in text and "only timing was captured" in text
    assert bm.reset("mine") and bm.load("mine") is None and not bm.reset("mine")


def test_pointer_features_show_up_in_bands(conn, monkeypatch):
    sid = db.start_session(conn, "tinder", "desktop-chrome", 0.0)
    scores = {}
    for i in range(10):
        pid = db.add_profile(conn, "tinder", [], external_id=str(i))
        scores[pid] = 90
        ev = [{"t": 100.0 * k, "k": "m", "x": 10.0 * k, "y": 0} for k in range(5)] + [{"t": 7000.0, "k": "w", "dy": 100}]
        add(conn, sid, pid, 8, events=ev, source="tinder")
    m = train(conn, monkeypatch, scores)
    assert m.overall.stats["n_moves"].p50 == 5 and m.overall.stats["wheel_down"].p50 == 1
    assert m.overall.hesitation_rate == 1.0  # 100ms moves then a 6.5s silence before the wheel, then the swipe
