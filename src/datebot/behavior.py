"""Behaviour telemetry: how *you* look at a profile before you swipe.

An interaction is the stretch between a profile becoming the current card and your swipe on it. What is stored is
its raw inputs (`raw`, see below); `derive` turns them into features and is the only place features are defined, so
a change here bumps `FEATURE_VERSION` and old interactions are re-derived from their stored raw inputs.

raw = {
  "start": epoch ms the card became current (previous swipe, else when the profile arrived),
  "decided": epoch ms of the swipe,
  "basis": "prev_swipe" | "ready"          how `start` was chosen (an inferred dwell, not a measured one),
  "events": [ {"t": epoch ms, "k": kind, ...} ]   browser only; empty for network-timing-only sources (Hinge)
}
event kinds: m move {x,y} | w wheel {dy} | kd/ku key down/up {key} | c click {x,y,rx,ry,tg} (rx,ry: 0..1 inside the
target element; tg: descriptor of it, classified by `target_label`) | f focus {v: 1 focused, 0 blurred}.

Hinge runs on a phone behind a proxy, so it only has timing; touch, scroll and pointer kinematics exist only for
browser sources. Features that need events are simply absent otherwise (never zero-filled: absent means unmeasured).
"""
from __future__ import annotations

import math

FEATURE_VERSION = 2
PAUSE_MS = 400        # an input gap at least this long counts as a pause
HESITATE_MS = 1000    # a pause this long followed by more input before the swipe counts as hesitation
APPROACH_MS = 150     # window before the swipe used for the approach (deceleration) features
# Click targets are classified from the stored descriptor (event["tg"]), so the rules can change and old
# interactions be re-derived. These keyword rules are GUESSES at Tinder/Badoo's labels, not verified against the
# live sites: a click that matches nothing is "other" (descriptor kept), and one without a descriptor is "unknown".
TARGET_RULES = (
    ("pass", ("dislike", "nope", "pass", "reject")),
    ("like", ("like", "vote-yes", "heart")),
    ("gallery_prev", ("previous", "prev", "back photo", "left photo")),
    ("gallery_next", ("next", "forward photo", "right photo")),
    ("open_profile", ("open profile", "profile", "info", "details", "expand", "see more", "more about")),
)
NAV_LABELS = ("gallery_prev", "gallery_next", "open_profile")
ARROWS = ("ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown")


def _mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def target_label(tg: dict | None) -> str:
    """Classify a click's stored descriptor: pass | like | gallery_prev | gallery_next | open_profile | other | unknown."""
    if not isinstance(tg, dict):
        return "unknown"
    hay = " ".join(str(tg.get(k, "")) for k in ("al", "ti", "id", "qa", "dt", "cl", "tx")).lower()
    for label, words in TARGET_RULES:
        if any(w in hay for w in words):
            return label
    return "other"


def _pointer(moves: list[dict], decided: float) -> dict:
    f: dict = {"n_moves": len(moves)}
    segs = []  # (t_end, length_px, speed_px_per_ms, heading)
    for a, b in zip(moves, moves[1:]):
        dt = b["t"] - a["t"]
        dx, dy = b["x"] - a["x"], b["y"] - a["y"]
        d = math.hypot(dx, dy)
        if dt > 0:
            segs.append((b["t"], d, d / dt, math.atan2(dy, dx) if d >= 2 else None))
    if not segs:
        return f
    path = sum(s[1] for s in segs)
    speeds = [s[2] for s in segs]
    f["path_px"] = round(path, 1)
    f["mean_speed"] = round(_mean(speeds), 4)
    f["max_speed"] = round(max(speeds), 4)
    chord = math.hypot(moves[-1]["x"] - moves[0]["x"], moves[-1]["y"] - moves[0]["y"])
    if path > 0:
        f["straightness"] = round(chord / path, 4)
    accel = [(speeds[i + 1] - speeds[i]) / max(1.0, segs[i + 1][0] - segs[i][0]) for i in range(len(segs) - 1)]
    if accel:
        f["peak_decel"] = round(max(0.0, -min(accel)), 6)
    turns = []
    headings = [s[3] for s in segs if s[3] is not None]
    for a, b in zip(headings, headings[1:]):
        turns.append(abs((b - a + math.pi) % (2 * math.pi) - math.pi))
    if turns:
        f["jitter"] = round(_mean(turns), 4)
        f["reversals"] = sum(t > math.pi / 2 for t in turns)
    last = [s[2] for s in segs if decided - s[0] <= APPROACH_MS]
    if last and f["max_speed"] > 0:
        f["end_speed_ratio"] = round(_mean(last) / f["max_speed"], 4)  # ~0: slowed to a stop at the target
    return f


def derive(raw: dict) -> dict:
    """Features from one interaction's raw inputs. Numeric values only; a missing key means 'not measured'."""
    start, decided = float(raw["start"]), float(raw["decided"])
    f: dict = {"dwell_ms": max(0, round(decided - start))}
    events = sorted((e for e in raw.get("events") or [] if start <= e["t"] <= decided + 50), key=lambda e: e["t"])
    if not events:
        return f
    f["n_events"] = len(events)
    f["first_input_ms"] = max(0, round(events[0]["t"] - start))
    f.update(_pointer([e for e in events if e["k"] == "m"], decided))

    ts = [e["t"] for e in events]
    gaps = [b - a for a, b in zip(ts, ts[1:])]
    pauses = [g for g in gaps if g >= PAUSE_MS]
    f["pauses"] = len(pauses)
    if pauses:
        f["longest_pause_ms"] = round(max(pauses))
    f["hesitated"] = int(any(g >= HESITATE_MS for g in gaps))

    wheel = [e["dy"] for e in events if e["k"] == "w" and e.get("dy")]
    if wheel:
        f["wheel_up"] = sum(d < 0 for d in wheel)
        f["wheel_down"] = sum(d > 0 for d in wheel)
        f["wheel_px"] = round(sum(abs(d) for d in wheel))
        f["scroll_reversals"] = sum((a < 0) != (b < 0) for a, b in zip(wheel, wheel[1:]))

    downs: dict[str, float] = {}
    holds = []
    for e in events:
        if e["k"] == "kd":
            downs.setdefault(e.get("key", ""), e["t"])
        elif e["k"] == "ku" and e.get("key", "") in downs:
            holds.append(e["t"] - downs.pop(e.get("key", "")))
    n_keys = sum(e["k"] == "kd" for e in events)
    if n_keys:
        f["n_keys"] = n_keys
    if holds:
        f["key_hold_ms"] = round(_mean(holds), 1)

    clicks = [e for e in events if e["k"] == "c"]
    if clicks:
        labels = [target_label(e.get("tg")) for e in clicks]
        f["n_clicks"] = len(clicks)
        f["n_gallery_next"] = labels.count("gallery_next")
        f["n_gallery_prev"] = labels.count("gallery_prev")
        f["n_open_profile"] = labels.count("open_profile")
        f["n_other_clicks"] = sum(l in ("other", "unknown") for l in labels)
        opened = [e for e, l in zip(clicks, labels) if l == "open_profile"]
        if opened:
            f["open_profile_ms"] = max(0, round(opened[0]["t"] - start))
    nav = {id(e) for e in clicks if target_label(e.get("tg")) in NAV_LABELS}  # browsing the profile, not deciding
    actions = [e for e in events if id(e) not in nav and (e["k"] == "c" or (e["k"] == "kd" and e.get("key") in ARROWS))]
    if actions:
        act = actions[-1]
        f["swipe_by_key"] = int(act["k"] == "kd")
        moves = [e for e in events if e["k"] == "m" and e["t"] <= act["t"]]
        if moves:
            f["pre_action_pause_ms"] = round(act["t"] - moves[-1]["t"])
        if act["k"] == "c" and "rx" in act and "ry" in act:
            f["click_rx"], f["click_ry"] = round(act["rx"], 3), round(act["ry"], 3)

    blurred, away = None, 0.0
    for e in events:
        if e["k"] == "f":
            if not e.get("v") and blurred is None:
                blurred = e["t"]
            elif e.get("v") and blurred is not None:
                away, blurred = away + e["t"] - blurred, None
    if blurred is not None:
        away += decided - blurred
    f["blur_ms"] = round(away)
    f["attention_ms"] = max(0, f["dwell_ms"] - f["blur_ms"])
    return f
