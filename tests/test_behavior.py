import pytest

from datebot import behavior


def raw(events=(), start=0, decided=10_000):
    return {"start": start, "decided": decided, "basis": "prev_swipe", "events": list(events)}


def test_timing_only_has_just_dwell():
    assert behavior.derive(raw()) == {"dwell_ms": 10_000}


def test_dwell_never_negative():
    assert behavior.derive(raw(start=5000, decided=4000))["dwell_ms"] == 0


def test_pointer_straight_line_slowing_to_stop():
    moves = [{"t": t, "k": "m", "x": x, "y": 0} for t, x in
             [(0, 0), (100, 100), (200, 200), (300, 300), (9800, 360), (9900, 370), (10000, 372)]]
    f = behavior.derive(raw(moves))
    assert f["n_moves"] == 7 and f["straightness"] == 1.0 and f["path_px"] == 372
    assert f["max_speed"] == 1.0 and f["end_speed_ratio"] < 0.1  # decelerated into the target
    assert f["peak_decel"] > 0 and f.get("jitter", 0) == 0


def test_jitter_and_reversals_on_a_zigzag():
    pts = [(0, 0), (50, 0), (0, 0), (50, 0), (0, 0)]
    f = behavior.derive(raw([{"t": i * 100, "k": "m", "x": x, "y": y} for i, (x, y) in enumerate(pts)]))
    assert f["reversals"] == 3 and f["straightness"] == 0.0


def test_pauses_and_hesitation():
    f = behavior.derive(raw([{"t": 0, "k": "m", "x": 0, "y": 0}, {"t": 1500, "k": "m", "x": 5, "y": 0},
                             {"t": 1600, "k": "m", "x": 9, "y": 0}]))
    assert f["pauses"] == 1 and f["longest_pause_ms"] == 1500 and f["hesitated"] == 1 and f["first_input_ms"] == 0
    quick = behavior.derive(raw([{"t": 100, "k": "m", "x": 0, "y": 0}, {"t": 200, "k": "m", "x": 5, "y": 0}]))
    assert quick["hesitated"] == 0 and quick["pauses"] == 0 and quick["first_input_ms"] == 100


def test_scrolling_counts_direction_and_reversals():
    f = behavior.derive(raw([{"t": 10, "k": "w", "dy": 100}, {"t": 20, "k": "w", "dy": 100},
                             {"t": 30, "k": "w", "dy": -50}, {"t": 40, "k": "w", "dy": 80}]))
    assert (f["wheel_down"], f["wheel_up"], f["wheel_px"], f["scroll_reversals"]) == (3, 1, 330, 2)


def test_key_swipe_and_hold_time():
    f = behavior.derive(raw([{"t": 9000, "k": "kd", "key": "ArrowRight"}, {"t": 9090, "k": "ku", "key": "ArrowRight"},
                             {"t": 8000, "k": "m", "x": 0, "y": 0}]))
    assert f["swipe_by_key"] == 1 and f["key_hold_ms"] == 90 and f["n_keys"] == 1 and f["pre_action_pause_ms"] == 1000


def test_click_position_inside_target():
    f = behavior.derive(raw([{"t": 9000, "k": "c", "x": 5, "y": 5, "rx": 0.4, "ry": 0.6}]))
    assert f["swipe_by_key"] == 0 and (f["click_rx"], f["click_ry"]) == (0.4, 0.6)


def test_blur_time_is_excluded_from_attention():
    f = behavior.derive(raw([{"t": 1000, "k": "f", "v": 0}, {"t": 4000, "k": "f", "v": 1}]))
    assert f["blur_ms"] == 3000 and f["attention_ms"] == 7000
    still_blurred = behavior.derive(raw([{"t": 8000, "k": "f", "v": 0}]))
    assert still_blurred["blur_ms"] == 2000


def test_events_outside_the_window_are_ignored():
    f = behavior.derive(raw([{"t": -5000, "k": "m", "x": 0, "y": 0}], start=0))
    assert "n_events" not in f


def click(t, **tg):
    return {"t": t, "k": "c", "x": 1, "y": 1, "tg": tg}


def test_target_labels():
    assert behavior.target_label(None) == "unknown"
    assert behavior.target_label({"tag": "div"}) == "other"
    assert behavior.target_label({"al": "Next Photo"}) == "gallery_next"
    assert behavior.target_label({"al": "previous photo"}) == "gallery_prev"
    assert behavior.target_label({"tx": "open profile"}) == "open_profile"
    assert behavior.target_label({"tx": "like"}) == "like"
    assert behavior.target_label({"tx": "nope"}) == "pass"
    assert behavior.target_label({"al": "dislike"}) == "pass"


def test_click_counts_and_profile_open_time():
    f = behavior.derive(raw([click(1000, al="open profile"), click(2000, al="next photo"), click(3000, al="next photo"),
                             click(4000, al="previous photo"), click(9000, tx="like")]))
    assert (f["n_clicks"], f["n_gallery_next"], f["n_gallery_prev"], f["n_open_profile"]) == (5, 2, 1, 1)
    assert f["open_profile_ms"] == 1000 and f["n_other_clicks"] == 0


def test_navigation_clicks_are_not_the_swipe_action():
    f = behavior.derive(raw([{"t": 8000, "k": "kd", "key": "ArrowRight"}, click(9000, al="next photo")]))
    assert f["swipe_by_key"] == 1
    assert "click_rx" not in f


def test_clicks_without_a_descriptor_still_count_and_can_be_the_action():
    f = behavior.derive(raw([{"t": 9000, "k": "c", "x": 5, "y": 5, "rx": 0.4, "ry": 0.6}]))
    assert f["n_clicks"] == 1 and f["n_other_clicks"] == 1 and f["swipe_by_key"] == 0 and f["click_rx"] == 0.4
