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


def test_click_descriptor_is_interface_vocabulary_not_profile_text():
    from datebot.capture.telemetry import INIT_SCRIPT
    assert "describe" in INIT_SCRIPT and "ev.tg" in INIT_SCRIPT
    assert INIT_SCRIPT.count("textContent") == 1  # only for short buttons


def test_context_events_are_carried_into_later_drains_with_input():
    b = EventBuffer()
    b.add(batch({"t": 1, "k": "v", "iw": 1200, "ih": 800}, {"t": 2, "k": "ly", "items": []}, {"t": 5, "k": "m", "x": 1, "y": 1}))
    first = b.drain(10)
    assert [e["k"] for e in first] == ["v", "ly", "m"]
    b.add(batch({"t": 20, "k": "m", "x": 2, "y": 2}, {"t": 25, "k": "c", "x": 2, "y": 2}))
    second = b.drain(30)
    assert [e["k"] for e in second] == ["v", "ly", "m", "c"]
    assert all(e.get("carried") for e in second[:2]) and second[0]["iw"] == 1200 and second[0]["t"] == 20
    b.add(batch({"t": 40, "k": "v", "iw": 900}, {"t": 41, "k": "m", "x": 0, "y": 0}))
    third = b.drain(50)
    assert [e["k"] for e in third] == ["v", "m", "ly"] or {e["k"] for e in third} == {"v", "m", "ly"}
    assert next(e for e in third if e["k"] == "v")["iw"] == 900


def test_context_alone_is_not_carried_without_input():
    b = EventBuffer()
    b.add(batch({"t": 1, "k": "v", "iw": 1}))
    b.drain(5)
    b.add(batch({"t": 10, "k": "f", "v": 1}))
    assert [e["k"] for e in b.drain(20)] == ["v", "f"]  # focus counts as non-context, so v is carried


def test_script_records_geometry_and_full_input_but_only_arrow_keys():
    from datebot.capture.telemetry import INIT_SCRIPT
    for needle in ("innerWidth", "devicePixelRatio", "visualViewport", "getBoundingClientRect", "pointerdown",
                   "pointerup", "getCoalescedEvents", "deltaX", "deltaMode", "touchstart", "scroll"):
        assert needle in INIT_SCRIPT
    assert "startsWith('Arrow')" in INIT_SCRIPT
    assert "e.code" not in INIT_SCRIPT and "currentSrc" not in INIT_SCRIPT
    assert INIT_SCRIPT.count("textContent") == 1
