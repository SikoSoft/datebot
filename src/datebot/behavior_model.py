"""The behaviour model: how you behave on a profile, as a function of how much the preference model likes it.

Trained alongside the preference model (model.py) and independent of it: a profile never needs an interaction, and
interactions only ever come from your own swipes (origin 'app'; the bot's are never learned from).

Version 1 is a set of distributions, not a classifier: interactions are grouped into bands of preference score
(default 0-30, 30-50, 50-70, 70-100) and each band keeps the quantiles of every behaviour feature, plus a log-normal
fit of dwell time that `sample_dwell_ms` can draw from. That answers "on a profile scoring 70+, I look for about
10 seconds" directly and works with little data. Session rhythm (gaps between swipes, breaks, time of day) is kept too.

The score of a profile you rated is its *out-of-fold* preference score (from a model that never saw that rating),
so a profile isn't scored by a model that memorised your verdict on it. Dwell above BREAK_MS is a break (you put the
phone down), not a look, and is counted separately.
"""
from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np

from datebot import behavior, config, db, evaluate, model

MODEL_VERSION = 2  # bump when the stored structure changes; older saved behaviour models are ignored and must be retrained
BAND_EDGES = (0.0, 30.0, 50.0, 70.0, 100.0)
MIN_BAND = 8                 # a band with fewer interactions is shown but not used for sampling
MIN_TOTAL = 10               # interactions with a score needed to train at all
BREAK_MS = 5 * 60_000
STAT_FEATURES = (
    "dwell_ms", "attention_ms", "first_input_ms", "n_moves", "path_px", "mean_speed", "max_speed", "end_speed_ratio",
    "peak_decel", "straightness", "jitter", "reversals", "pauses", "longest_pause_ms", "pre_action_pause_ms",
    "wheel_up", "wheel_down", "scroll_reversals", "key_hold_ms", "n_clicks", "n_gallery_next", "n_gallery_prev",
    "n_open_profile", "open_profile_ms", "swipe_by_key", "n_keys", "click_rx", "click_ry",
)
# key -> (label, unit, "browser" if only browser sessions measure it). Used by the UI page and `heuristics`.
FEATURE_INFO: dict[str, tuple[str, str, bool]] = {
    "dwell_ms": ("Time on the card", "ms", False),
    "attention_ms": ("Time with the window focused", "ms", True),
    "first_input_ms": ("Time before first input", "ms", True),
    "n_moves": ("Pointer movements", "", True),
    "path_px": ("Pointer path length", "px", True),
    "mean_speed": ("Mean pointer speed", "px/ms", True),
    "max_speed": ("Peak pointer speed", "px/ms", True),
    "end_speed_ratio": ("Speed at the swipe vs peak (0 = stopped)", "", True),
    "peak_decel": ("Peak deceleration", "", True),
    "straightness": ("Path straightness (1 = straight line)", "", True),
    "jitter": ("Direction jitter (mean turn, rad)", "", True),
    "reversals": ("Sharp direction reversals", "", True),
    "pauses": ("Pauses in input", "", True),
    "longest_pause_ms": ("Longest pause", "ms", True),
    "pre_action_pause_ms": ("Pointer still before the swipe", "ms", True),
    "wheel_up": ("Scroll ticks up", "", True),
    "wheel_down": ("Scroll ticks down", "", True),
    "scroll_reversals": ("Scroll direction reversals", "", True),
    "key_hold_ms": ("Arrow-key hold", "ms", True),
    "n_keys": ("Arrow-key presses", "", True),
    "swipe_by_key": ("Swipes with an arrow key (rest: click)", "", True),
    "n_clicks": ("Clicks", "", True),
    "n_gallery_next": ("Next-photo clicks", "", True),
    "n_gallery_prev": ("Previous-photo clicks", "", True),
    "n_open_profile": ("Open-profile clicks", "", True),
    "open_profile_ms": ("Time until profile opened", "ms", True),
    "click_rx": ("Click position in button, left to right (0-1)", "", True),
    "click_ry": ("Click position in button, top to bottom (0-1)", "", True),
}
HEURISTIC_CHANGE = 0.25      # a feature must differ by this much (relative) between the lowest and highest band to be called out


@dataclass
class Dist:
    n: int
    mean: float
    p10: float
    p50: float
    p90: float


def dist(values: list[float]) -> Dist | None:
    if not values:
        return None
    a = np.asarray(values, dtype=float)
    p10, p50, p90 = np.percentile(a, [10, 50, 90])
    return Dist(len(a), float(a.mean()), float(p10), float(p50), float(p90))


@dataclass
class Band:
    lo: float
    hi: float
    n: int
    like_rate: float | None
    hesitation_rate: float | None
    stats: dict[str, Dist]
    dwell_by_decision: dict[str, Dist] = field(default_factory=dict)
    breaks: int = 0                                   # interactions with dwell above BREAK_MS, left out of the stats
    log_dwell: tuple[float, float] | None = None      # (mu, sigma) of ln(dwell ms)


@dataclass
class BehaviorModel:
    edges: tuple[float, ...]
    bands: list[Band]
    overall: Band
    rhythm: dict
    n_interactions: int
    source: str | None
    trained_at: str
    preference_model: str
    version: int = MODEL_VERSION
    feature_version: int = behavior.FEATURE_VERSION


def _dir() -> Path:
    path = config.models_dir() / "behavior"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _path(name: str) -> Path:
    return _dir() / f"{name}.joblib"


def load(name: str = "default") -> BehaviorModel | None:
    path = _path(name)
    if not path.exists():
        return None
    m = joblib.load(path)
    ok = getattr(m, "version", 0) == MODEL_VERSION and getattr(m, "feature_version", 0) == behavior.FEATURE_VERSION
    return m if ok else None  # stale: retrain


def reset(name: str = "default") -> bool:
    """Forget a trained behaviour model. Collected interactions are kept; retrain any time."""
    path = _path(name)
    if path.exists():
        path.unlink()
        return True
    return False


def list_models() -> list[str]:
    return sorted(p.stem for p in _dir().glob("*.joblib"))


def refresh_features(conn: sqlite3.Connection) -> int:
    """Re-derive features for interactions stored under an older FEATURE_VERSION, from their stored raw inputs."""
    n = 0
    for r in conn.execute("SELECT * FROM interactions WHERE feature_version != ?", (behavior.FEATURE_VERSION,)).fetchall():
        db.set_interaction_features(conn, r["id"], behavior.derive(db.raw_of(r)), behavior.FEATURE_VERSION)
        n += 1
    return n


def preference_scores(conn: sqlite3.Connection, embed_model: str, pref_name: str = "default") -> dict[int, float]:
    """Preference score 0-100 per profile: out-of-fold for rated profiles, else the trained model's (if any)."""
    scores: dict[int, float] = {}
    ids, X, y = model.training_data(conn, embed_model)
    if len(y):
        try:
            scores = {pid: float(s) * 100 for pid, s in zip(ids, evaluate.oof_scores(X, y))}
        except ValueError:
            pass  # too few ratings to cross-validate
    trained = model.load(pref_name)
    if trained:
        for (pid,) in conn.execute("SELECT DISTINCT profile_id FROM interactions WHERE profile_id IS NOT NULL").fetchall():
            if pid not in scores and (s := model.score_profile(conn, trained, pid)) is not None:
                scores[pid] = s
    return scores


def _band(lo: float, hi: float, rows: list[dict]) -> Band:
    fs = [r["features"] for r in rows]
    breaks = sum(f["dwell_ms"] > BREAK_MS for f in fs)
    looks = [r for r in rows if r["features"]["dwell_ms"] <= BREAK_MS]
    stats = {}
    for key in STAT_FEATURES:
        d = dist([r["features"][key] for r in looks if key in r["features"]])
        if d:
            stats[key] = d
    by_dec = {}
    for label, want in (("like", 1), ("pass", 0)):
        d = dist([r["features"]["dwell_ms"] for r in looks if r["liked"] == want])
        if d:
            by_dec[label] = d
    logs = [math.log(max(1.0, r["features"]["dwell_ms"])) for r in looks]
    liked = [r["liked"] for r in rows if r["liked"] is not None]
    return Band(
        lo, hi, len(rows),
        like_rate=sum(liked) / len(liked) if liked else None,
        hesitation_rate=(sum(f.get("hesitated", 0) for f in fs if "n_events" in f)
                         / max(1, sum("n_events" in f for f in fs))) if any("n_events" in f for f in fs) else None,
        stats=stats, dwell_by_decision=by_dec, breaks=breaks,
        log_dwell=(float(np.mean(logs)), float(np.std(logs))) if len(logs) >= 2 else None,
    )


def rhythm(rows: list[sqlite3.Row]) -> dict:
    """Session-level pacing from consecutive swipes: gap quantiles, breaks, swipes per active minute, time of day."""
    by_session: dict[int, list[float]] = {}
    for r in rows:
        by_session.setdefault(r["session_id"], []).append(r["decided_at"])
    gaps, breaks, lengths = [], 0, []
    for times in by_session.values():
        times.sort()
        lengths.append((times[-1] - times[0]) / 60)
        for a, b in zip(times, times[1:]):
            if (b - a) * 1000 > BREAK_MS:
                breaks += 1
            else:
                gaps.append((b - a) * 1000)
    hours = [0] * 24
    for r in rows:
        hours[datetime.fromtimestamp(r["decided_at"]).hour] += 1
    active_min = sum(gaps) / 60_000
    return {
        "sessions": len(by_session), "swipes": len(rows), "breaks": breaks,
        "gap_ms": dist(gaps), "session_minutes": dist(lengths),
        "swipes_per_active_min": round(len(gaps) / active_min, 2) if active_min > 0 else None,
        "hour_hist": hours,
    }


def train(conn: sqlite3.Connection, embed_model: str, name: str = "default", source: str | None = None,
          edges: tuple[float, ...] = BAND_EDGES, pref_name: str = "default") -> BehaviorModel:
    refresh_features(conn)
    rows = [r for r in db.interactions(conn, origin="app") if source in (None, r["source"])]
    scores = preference_scores(conn, embed_model, pref_name)
    scored = [{"features": json.loads(r["features"]), "liked": r["liked"], "score": scores[r["profile_id"]]}
              for r in rows if r["profile_id"] in scores]
    if len(scored) < MIN_TOTAL:
        raise ValueError(
            f"Need at least {MIN_TOTAL} of your swipes with a captured profile and a preference score "
            f"(have {len(scored)} of {len(rows)} interactions"
            + ("" if len(scores) else "; the preference model has too few ratings to score anything") + ").")
    bands = []
    for i, (lo, hi) in enumerate(zip(edges, edges[1:])):
        last = i == len(edges) - 2
        bands.append(_band(lo, hi, [r for r in scored if lo <= r["score"] < hi or (last and r["score"] == hi)]))
    m = BehaviorModel(
        edges=tuple(edges), bands=bands, overall=_band(edges[0], edges[-1], scored), rhythm=rhythm(rows),
        n_interactions=len(scored), source=source, trained_at=datetime.now(timezone.utc).isoformat(),
        preference_model=pref_name)
    joblib.dump(m, _path(name))
    return m


def band_for(m: BehaviorModel, score: float) -> Band:
    """The band `score` falls in if it has enough interactions to trust, else the all-scores fallback."""
    for b in m.bands:
        if b.lo <= score < b.hi or (b is m.bands[-1] and score == b.hi):
            if b.n >= MIN_BAND and b.log_dwell:
                return b
    return m.overall


def sample_dwell_ms(m: BehaviorModel, score: float, rng: np.random.Generator | None = None) -> int:
    """Draw a dwell time (ms) from the log-normal fitted to your own looks at profiles scoring like `score`.

    Nothing calls this yet: it exists so the model's output can be inspected and tested. Wiring it into any action on
    your account is a separate decision."""
    rng = rng or np.random.default_rng()
    b = band_for(m, score)
    if not b.log_dwell:
        raise ValueError("model has no dwell data")
    mu, sigma = b.log_dwell
    return int(min(BREAK_MS, max(1.0, rng.lognormal(mu, sigma))))


def _s(ms: float | None) -> str:
    return "-" if ms is None else f"{ms / 1000:.1f}s"


def format_report(m: BehaviorModel) -> str:
    out = [f"Behaviour model: {m.n_interactions} of your swipes"
           + (f" ({m.source})" if m.source else " (all sources)")
           + f", trained {m.trained_at[:16].replace('T', ' ')}, scored by preference model '{m.preference_model}'.",
           "",
           "Dwell = time from the card appearing to your swipe (inferred from timestamps). Median (10th-90th percentile):",
           f"{'score':>9} {'n':>5} {'like%':>6} {'dwell':>22} {'like dwell':>11} {'pass dwell':>11} {'hesitate':>9}"]
    for b in m.bands + [m.overall]:
        label = "all" if b is m.overall else f"{b.lo:.0f}-{b.hi:.0f}"
        d = b.stats.get("dwell_ms")
        span = f"{_s(d.p50)} ({_s(d.p10)}-{_s(d.p90)})" if d else "-"
        like, pas = b.dwell_by_decision.get("like"), b.dwell_by_decision.get("pass")
        hes = "-" if b.hesitation_rate is None else f"{b.hesitation_rate:.0%}"
        lr = "-" if b.like_rate is None else f"{b.like_rate:.0%}"
        thin = "  (too few to sample)" if b is not m.overall and b.n < MIN_BAND else ""
        out.append(f"{label:>9} {b.n:>5} {lr:>6} {span:>22} {_s(like and like.p50):>11} {_s(pas and pas.p50):>11} {hes:>9}{thin}")
    pointer = [(k, m.overall.stats[k]) for k in STAT_FEATURES[1:] if k in m.overall.stats]
    if pointer:
        out += ["", "Input behaviour (browser sessions only), all bands, median (10th-90th):"]
        out += [f"  {k:<20} {d.p50:10.3f}  ({d.p10:.3f}-{d.p90:.3f})  n={d.n}" for k, d in pointer]
    else:
        out += ["", "No input behaviour yet: only timing was captured (Hinge via proxy gives dwell, not touch or scroll)."]
    r = m.rhythm
    out += ["", f"Rhythm: {r['sessions']} session(s), {r['swipes']} swipes, {r['breaks']} break(s) over {BREAK_MS // 60000} min."]
    if r["gap_ms"]:
        g = r["gap_ms"]
        out.append(f"  gap between swipes: median {_s(g.p50)} ({_s(g.p10)}-{_s(g.p90)})"
                   + (f", {r['swipes_per_active_min']} swipes per active minute" if r["swipes_per_active_min"] else ""))
    busy = sorted(range(24), key=lambda h: -r["hour_hist"][h])[:3]
    if r["swipes"]:
        out.append("  busiest hours: " + ", ".join(f"{h:02d}:00 ({r['hour_hist'][h]})" for h in busy if r["hour_hist"][h]))
    return "\n".join(out)


def _spoken(key: str, v: float) -> str:
    unit = FEATURE_INFO[key][1]
    if unit == "ms":
        return f"{v / 1000:.2f}s" if abs(v) < 1000 else f"{v / 1000:.1f}s"
    if unit == "px":
        return f"{v:.0f}px"
    return f"{v:.2f}" if abs(v) < 10 else f"{v:.1f}"


def heuristics(m: BehaviorModel) -> list[dict]:
    """Plain-language rules read off the model: how each behaviour changes between your lowest- and highest-scored
    profiles. Only bands with MIN_BAND interactions are compared; a rule needs a HEURISTIC_CHANGE relative difference.
    Each is {"text", "feature", "direction": up|down|flat, "strength"} (for ordering: like rate and hesitation first, then the biggest relative changes)."""
    usable = [b for b in m.bands if b.n >= MIN_BAND]
    out: list[dict] = []
    if len(usable) < 2:
        return out
    lo, hi = usable[0], usable[-1]
    names = (f"{lo.lo:.0f}-{lo.hi:.0f}", f"{hi.lo:.0f}-{hi.hi:.0f}")
    if lo.like_rate is not None and hi.like_rate is not None and abs(hi.like_rate - lo.like_rate) >= 0.1:
        out.append({"text": f"You like {hi.like_rate:.0%} of profiles scored {names[1]} but {lo.like_rate:.0%} of those scored {names[0]}.",
                    "feature": "like_rate", "direction": "up" if hi.like_rate > lo.like_rate else "down",
                    "strength": abs(hi.like_rate - lo.like_rate)})
    if lo.hesitation_rate is not None and hi.hesitation_rate is not None and abs(hi.hesitation_rate - lo.hesitation_rate) >= 0.1:
        up = hi.hesitation_rate > lo.hesitation_rate
        out.append({"text": f"You hesitate (a 1s+ stop, then more input) on {hi.hesitation_rate:.0%} of {names[1]} profiles versus "
                            f"{lo.hesitation_rate:.0%} of {names[0]} ones.",
                    "feature": "hesitation", "direction": "up" if up else "down",
                    "strength": abs(hi.hesitation_rate - lo.hesitation_rate)})
    for key, (label, _unit, _browser) in FEATURE_INFO.items():
        a, b = lo.stats.get(key), hi.stats.get(key)
        if not a or not b or min(a.n, b.n) < MIN_BAND // 2:
            continue
        # counts are mostly 0 or 1, where the median hides the effect: compare means for those
        count = key.startswith(("n_", "wheel_", "swipe_by")) or key in ("pauses", "reversals", "scroll_reversals")
        x, y = (a.mean, b.mean) if count else (a.p50, b.p50)
        base = max(abs(x), abs(y))
        if base < 1e-9 or (count and base < 0.1):
            continue
        change = (y - x) / max(abs(x), 1e-9) if abs(x) > 1e-9 else float("inf")
        if abs(y - x) / base < HEURISTIC_CHANGE:
            continue
        up = y > x
        word = "more" if up else "less"
        out.append({"text": f"{label}: {_spoken(key, y)} on {names[1]} profiles vs {_spoken(key, x)} on {names[0]} ({word} for profiles you rate higher).",
                    "feature": key, "direction": "up" if up else "down",
                    "strength": min(abs(change), 10.0)})
    for label, hi_b in (("lowest", lo), ("highest", hi)):
        dec = hi_b.dwell_by_decision
        if "like" in dec and "pass" in dec and dec["like"].n >= 3 and dec["pass"].n >= 3:
            a, b = dec["like"].p50, dec["pass"].p50
            if abs(a - b) / max(a, b, 1) >= HEURISTIC_CHANGE:
                out.append({"text": f"Among your {label}-scored profiles you take {_spoken('dwell_ms', a)} to like and {_spoken('dwell_ms', b)} to pass (median).",
                            "feature": "dwell_by_decision", "direction": "up" if a > b else "down",
                            "strength": abs(a - b) / max(a, b, 1)})
    return sorted(out, key=lambda r: (r["feature"] not in ("like_rate", "hesitation"), -r["strength"]))


def describe(m: BehaviorModel) -> dict:
    """The model as plain JSON for the UI: bands, overall, rhythm, rules, which bands can be sampled, and feature labels."""
    from dataclasses import asdict

    d = asdict(m)
    for b, raw in zip(d["bands"], m.bands):
        b["usable"] = raw.n >= MIN_BAND and raw.log_dwell is not None
    d["heuristics"] = heuristics(m)
    d["features"] = {k: {"label": v[0], "unit": v[1], "browser_only": v[2]} for k, v in FEATURE_INFO.items()}
    d["min_band"], d["break_ms"] = MIN_BAND, BREAK_MS
    return d


# ---- replaying one of your real swipes (display only) -------------------------------------------------------------

CAL_LABELS = ("pass", "like", "gallery_next", "gallery_prev")


def _median_xy(points: list[tuple[float, float]]) -> list[float] | None:
    return [float(np.median([p[0] for p in points])), float(np.median([p[1] for p in points]))] if points else None


def calibration(rows: list[sqlite3.Row], session_id: int | None) -> dict[str, list[float]]:
    """Where the pass/like buttons and the photo's next/back arrows were on your screen, from where you clicked them.

    Medians of the labelled clicks in the same session (window size can change between sessions), falling back to all
    of `rows` per label. Lets the page map real pointer coordinates onto its phone window."""
    pts: dict[str, dict[bool, list[tuple[float, float]]]] = {k: {True: [], False: []} for k in CAL_LABELS}
    for r in rows:
        same = r["session_id"] == session_id
        for e in db.raw_of(r).get("events") or []:
            if e["k"] == "c" and "x" in e:
                label = behavior.target_label(e.get("tg"))
                if label in pts:
                    pts[label][same].append((e["x"], e["y"]))
    out = {}
    for k, by in pts.items():
        m = _median_xy(by[True]) or _median_xy(by[False])
        if m:
            out[k] = m
    return out


def replay_events(raw: dict) -> list[dict]:
    """The stored events of one interaction, relative to the card appearing and stripped to what a replay needs
    (positions, wheel direction, arrow-key names, and each click's label; no element descriptors)."""
    start, decided = float(raw["start"]), float(raw["decided"])
    out = []
    for e in sorted((e for e in raw.get("events") or [] if start <= e["t"] <= decided + 50), key=lambda e: e["t"]):
        ev: dict = {"t": round(e["t"] - start), "k": e["k"]}
        if e["k"] == "m":
            ev.update(x=e["x"], y=e["y"])
        elif e["k"] == "w":
            ev["dy"] = 1 if e.get("dy", 0) > 0 else -1 if e.get("dy", 0) < 0 else 0
        elif e["k"] in ("kd", "ku"):
            ev["key"] = e.get("key", "")
        elif e["k"] == "c":
            ev.update(x=e.get("x"), y=e.get("y"), label=behavior.target_label(e.get("tg")))
        elif e["k"] == "f":
            ev["v"] = e.get("v", 1)
        out.append(ev)
    return out


def replay(conn: sqlite3.Connection, embed_model: str, source: str | None, lo: float, hi: float,
           liked: bool | None, index: int, pref_name: str = "default") -> dict:
    """One of your own recorded swipes whose preference score is in [lo, hi] (hi inclusive at 100), for the Behaviour page.

    `index` steps through the matches (modulo their count) in a stable order. Only origin 'app' interactions that have
    input events and were a look (not a break) are candidates."""
    scores = preference_scores(conn, embed_model, pref_name)
    rows = [r for r in db.interactions(conn, origin="app") if source in (None, "", r["source"])]
    cands = []
    for r in rows:
        s = scores.get(r["profile_id"])
        if s is None or not (lo <= s < hi or (hi >= 100 and s == hi)):
            continue
        if liked is not None and r["liked"] != int(liked):
            continue
        f = json.loads(r["features"])
        if "n_events" in f and f["dwell_ms"] <= BREAK_MS:
            cands.append((r, s))
    if not cands:
        return {"count": 0}
    r, s = cands[index % len(cands)]
    raw = db.raw_of(r)
    return {"count": len(cands), "index": index % len(cands), "id": r["id"], "source": r["source"], "score": round(s, 1),
            "liked": None if r["liked"] is None else bool(r["liked"]), "dwell_ms": round(raw["decided"] - raw["start"]),
            "events": replay_events(raw), "calibration": calibration(rows, r["session_id"])}
