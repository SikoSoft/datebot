import pytest

from datebot import db
from datebot.capture.recorder import Recorder
from datebot.capture.tinder import CapturedProfile, Swipe
from datebot.interactions import InteractionTracker
from test_recorder import png


class Clock:
    t = 1000.0

    def __call__(self):
        return self.t


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("DATEBOT_HOME", str(tmp_path))
    clock = Clock()
    tracker = InteractionTracker("hinge", "iphone-proxy", now=clock)
    path = str(tmp_path / "t.db")
    rec = Recorder(fetch=lambda url: png() + url.encode(), source="hinge", db_path=path, tracker=tracker)
    return rec, clock, path


def profile(i):
    return CapturedProfile(external_id=i, name=i, photo_urls=[f"https://x/{i}.jpg"])


def test_first_swipe_dwell_runs_from_profile_arrival_then_from_previous_swipe(env):
    rec, clock, path = env
    rec.record_profiles([profile("a"), profile("b")])  # both arrive at t=1000
    clock.t = 1004
    rec.record_swipe(Swipe("a", True))
    clock.t = 1019
    rec.record_swipe(Swipe("b", False))
    conn = db.connect(path)
    rows = db.interactions(conn)
    import json
    f = [json.loads(r["features"]) for r in rows]
    assert [x["dwell_ms"] for x in f] == [4000, 15000]
    assert [db.raw_of(r)["basis"] for r in rows] == ["ready", "prev_swipe"]
    assert all(r["rating_id"] and r["profile_id"] and r["signal"] == "timing" for r in rows)
    assert len(conn.execute("SELECT id FROM sessions").fetchall()) == 1


def test_swipe_on_uncaptured_profile_is_still_a_timing_sample_without_profile(env):
    rec, clock, path = env
    rec.record_profiles([profile("a")])
    clock.t = 1005
    rec.record_swipe(Swipe("a", True))
    clock.t = 1012
    assert rec.record_swipe(Swipe("ghost", True)) is False
    rows = db.interactions(db.connect(path))
    assert [r["profile_id"] for r in rows] == [rows[0]["profile_id"], None] and rows[1]["external_id"] == "ghost"
    assert rows[1]["rating_id"] is None and rec.counts.unknown_swipes == 1


def test_input_event_sets_decision_time_and_signal(env):
    rec, clock, path = env
    rec.record_profiles([profile("a")])
    clock.t = 1010.3  # the response arrives 300ms after the key press
    events = [{"t": 1_005_000, "k": "m", "x": 0, "y": 0}, {"t": 1_010_000, "k": "kd", "key": "ArrowRight"}]
    rec.record_swipe(Swipe("a", True), events=events)
    row = db.interactions(db.connect(path))[0]
    assert row["signal"] == "pointer" and row["decided_at"] == 1010.0
    import json
    assert json.loads(row["features"])["dwell_ms"] == 10000


def test_unknown_start_is_not_guessed(env):
    rec, clock, path = env
    rec.record_swipe(Swipe("a", True))  # never saw the profile arrive, no previous swipe, no events
    assert db.interactions(db.connect(path)) == []


def test_telemetry_failure_never_loses_the_rating(env, monkeypatch):
    rec, clock, path = env
    rec.record_profiles([profile("a")])
    monkeypatch.setattr(rec.tracker, "record", lambda *a, **k: 1 / 0)
    assert rec.record_swipe(Swipe("a", True)) is True
    conn = db.connect(path)
    assert list(db.latest_ratings(conn).values()) == [True] and rec.counts.telemetry_failed == 1


def test_auto_swipes_are_tagged_so_they_can_be_excluded(env):
    rec, clock, path = env
    rec.record_profiles([profile("a")])
    clock.t = 1002
    rec.record_swipe(Swipe("a", True), origin="auto")
    conn = db.connect(path)
    assert db.interactions(conn) == [] and len(db.interactions(conn, origin=None)) == 1
