import json

from datebot.capture.telemetry import MAX_EVENTS, EventBuffer


def batch(*events):
    return json.dumps(list(events))


def test_drain_returns_events_up_to_the_swipe_in_time_order_and_keeps_later_ones():
    b = EventBuffer()
    b.add(batch({"t": 30, "k": "m", "x": 1, "y": 1}, {"t": 10, "k": "m", "x": 0, "y": 0}))
    b.add(batch({"t": 90, "k": "m", "x": 2, "y": 2}))
    assert [e["t"] for e in b.drain(50)] == [10, 30]
    assert [e["t"] for e in b.drain(100)] == [90]
    assert b.drain(1000) == []


def test_garbage_from_the_page_is_ignored():
    b = EventBuffer()
    for payload in ("not json", "{}", batch("x", {"k": "m"}, {"t": "late", "k": "m"}, {"t": 5, "k": "m"})):
        b.add(payload)
    assert [e["t"] for e in b.drain(10)] == [5]


def test_buffer_is_bounded():
    b = EventBuffer()
    b.add(batch(*({"t": i, "k": "m", "x": 0, "y": 0} for i in range(MAX_EVENTS + 50))))
    got = b.drain(10**9)
    assert len(got) == MAX_EVENTS and got[0]["t"] == 50
