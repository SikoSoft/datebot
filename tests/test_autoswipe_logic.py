import pytest

from datebot.autoswipe.logic import Bands, CardQueue, Governor, Handshake, decide


def test_bands_must_not_overlap():
    with pytest.raises(ValueError):
        Bands(like_at=40, pass_at=60)
    with pytest.raises(ValueError):
        Bands(like_at=50, pass_at=50)
    assert Bands().empty and not Bands(like_at=80).empty


def test_decide_by_band():
    b = Bands(like_at=80, pass_at=20)
    assert decide(95, b).action == "like" and decide(80, b).action == "like"
    assert decide(5, b).action == "pass" and decide(20, b).action == "pass"
    d = decide(50, b)
    assert d.action == "human" and "uncertain" in d.reason


def test_decide_with_only_one_band_leaves_the_rest_to_you():
    assert decide(5, Bands(like_at=80)).action == "human"
    assert decide(95, Bands(pass_at=20)).action == "human"
    assert decide(95, Bands()).action == "human"


def test_missing_score_is_never_acted_on():
    assert decide(None, Bands(like_at=80, pass_at=20)).action == "human"


def test_deal_breakers_pass_even_without_a_score_unless_disabled():
    b = Bands(like_at=80, pass_at=20)
    d = decide(95, b, filter_reason="age 19 < 21")
    assert d.action == "pass" and "age 19" in d.reason          # a deal-breaker beats a high score
    assert decide(None, b, filter_reason="x").action == "pass"
    assert decide(95, b, filter_reason="x", filters_pass=False).action == "like"


def test_queue_tracks_top_card_and_dedupes():
    q = CardQueue()
    assert q.current() is None
    q.add(["a", "b", "c"])
    q.add(["c", "d"])                      # overlapping batch
    assert q.current() == "a" and len(q) == 4
    assert q.swiped("a") is True and q.current() == "b"
    assert q.swiped("c") is False          # swiped out of order: prediction was wrong
    assert q.current() == "b" and len(q) == 2
    q.add(["a", "c"])                      # already-swiped cards never come back
    assert len(q) == 2


def test_queue_unknown_swipe_is_a_mismatch():
    q = CardQueue()
    q.add(["a"])
    assert q.swiped("zzz") is False and q.current() == "a"


def test_handshake_needs_a_streak_and_a_mismatch_resets_it():
    h = Handshake(required=3)
    assert not h.trusted
    for _ in range(3):
        h.observe(True)
    assert h.trusted
    h.observe(False)
    assert not h.trusted and h.streak == 0
    h.observe(True); h.observe(True)
    assert not h.trusted


class Clock:
    t = 100.0

    def __call__(self):
        return self.t


def test_governor_enforces_delay_run_cap_and_daily_cap():
    c = Clock()
    g = Governor(run_cap=2, daily_cap=10, min_delay=5, clock=c)
    assert g.ready()
    g.record()
    assert not g.ready()                   # too soon
    c.t += 5
    assert g.ready()
    g.record()
    assert g.exhausted and "run limit" in g.exhausted and not g.ready()

    d = Governor(run_cap=50, daily_cap=3, already_today=3, min_delay=0, clock=c)
    assert d.exhausted and "daily limit" in d.exhausted and not d.ready()
    e = Governor(run_cap=50, daily_cap=3, already_today=2, min_delay=0, clock=c)
    assert e.ready()
    e.record()
    assert e.exhausted and "daily" in e.exhausted
