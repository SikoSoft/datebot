from datebot.autoswipe.logic import Bands, Governor, Handshake
from datebot.autoswipe.runner import LIKE_KEY, PASS_KEY, AutoSwiper
from datebot.capture.tinder import CapturedProfile, Swipe

SCORES = {"hi": 95.0, "lo": 5.0, "mid": 50.0, "hi2": 90.0}


class FakeDriver:
    """Plays Tinder: a key press swipes the top card, optionally misbehaving."""

    def __init__(self, mode="ok"):
        self.swiper = None
        self.mode = mode
        self.keys = []

    def press(self, key):
        self.keys.append(key)
        if self.mode == "silent":
            return
        sw = self.swiper
        top = sw.queue.current()
        liked = key == LIKE_KEY
        if self.mode == "flip":
            liked = not liked
        if self.mode == "other":
            top = "mid"
        s = Swipe(top, liked)
        sw.on_swipe(s, True, sw.swipe_origin(s))

    def wait(self, ms):
        pass


def make(mode="ok", live=True, trusted=True, bands=Bands(80, 20), cap=10):
    d = FakeDriver(mode)
    h = Handshake(2)
    if trusted:
        h.observe(True); h.observe(True)
    t = [0.0]

    def clock():
        t[0] += 1
        return t[0]

    s = AutoSwiper(None, lambda i: (SCORES[i], None), bands, Governor(cap, 100, min_delay=0, clock=clock), h, d,
                   live=live, key_timeout=3, log=lambda m: None, clock=clock)
    d.swiper = s
    return s, d


def feed(s, *ids):
    s.on_profiles([CapturedProfile(i, name=i) for i in ids])


def test_observe_only_never_presses():
    s, d = make(live=False)
    feed(s, "hi")
    assert s.step() and d.keys == []


def test_untrusted_bot_does_not_press():
    s, d = make(trusted=False)
    feed(s, "lo")
    s.step()
    assert d.keys == []


def test_first_action_must_be_a_pass():
    s, d = make()
    feed(s, "hi", "lo")
    s.step()
    assert d.keys == []                      # top card is a like; mapping unverified -> left to you
    s.on_swipe(Swipe("hi", True), True, s.swipe_origin(Swipe("hi", True)))  # you swipe it yourself
    s.step()
    assert d.keys == [PASS_KEY] and s.passed == 1 and s.mapping_verified
    feed(s, "hi2")
    s.step()
    assert d.keys == [PASS_KEY, LIKE_KEY] and s.liked == 1


def test_uncertain_card_waits_for_you():
    s, d = make()
    feed(s, "mid", "lo")
    s.step()
    assert d.keys == [] and s.left_to_you == 1


def test_wrong_direction_halts():
    s, d = make(mode="flip")
    feed(s, "lo")
    assert s.step() is False
    assert "WRONG DIRECTION" in s.halted


def test_no_swipe_halts():
    s, d = make(mode="silent")
    feed(s, "lo")
    s.step()
    assert "no swipe happened" in s.halted


def test_different_card_halts():
    s, d = make(mode="other")
    feed(s, "lo", "mid")
    s.step()
    assert "different profile" in s.halted


def test_run_cap_stops():
    s, d = make(cap=1)
    feed(s, "lo", "lo")
    s.step()
    assert s.step() is False and "run limit" in s.finished


def test_scoring_failure_is_left_to_you():
    s, d = make()
    s.assess = lambda i: 1 / 0
    feed(s, "lo")
    s.step()
    assert d.keys == [] and "scoring failed" in s.decisions["lo"].reason


def test_bot_swipes_tagged_auto_and_yours_app():
    s, d = make()
    feed(s, "lo", "x")
    s.step()
    assert s.swipe_origin(Swipe("x", True)) == "app"
