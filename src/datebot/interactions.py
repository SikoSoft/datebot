"""Turns swipes into behaviour interactions (see behavior.py), one session per capture run."""
from __future__ import annotations

import time
from typing import Callable

from datebot import behavior, db

ACTION_WINDOW_MS = 2000  # the input that caused a swipe precedes the network event by well under this


class InteractionTracker:
    """Tracks when each card became current so a swipe can be stored with its dwell and any input events.

    `device` says what kind of signal to expect (desktop-chrome: input events; iphone-proxy: timing only)."""

    def __init__(self, source: str, device: str, now: Callable[[], float] = time.time):
        self.source, self.device, self.now = source, device, now
        self._session: int | None = None
        self._last_swipe_ms: float | None = None
        self._ready: dict[str, float] = {}

    def profile_ready(self, external_id: str, t_ms: float | None = None) -> None:
        """A profile is now available to swipe on (its data arrived)."""
        self._ready.setdefault(external_id, self.now() * 1000 if t_ms is None else t_ms)
        while len(self._ready) > 1000:
            self._ready.pop(next(iter(self._ready)))

    def advance(self, t_ms: float | None = None) -> None:
        """A swipe happened that is not recorded; the next card still starts now."""
        self._last_swipe_ms = self.now() * 1000 if t_ms is None else t_ms

    def record(self, conn, *, external_id: str, profile_id: int | None, liked: bool, origin: str,
               rating_id: int | None, decided_ms: float | None = None, events: list[dict] | None = None) -> int | None:
        decided = self.now() * 1000 if decided_ms is None else decided_ms
        events = events or []
        acts = [e for e in events if decided - ACTION_WINDOW_MS <= e["t"] <= decided
                and (e["k"] == "c" or (e["k"] == "kd" and e.get("key") in behavior.ARROWS))]
        if acts:
            decided = acts[-1]["t"]  # the input itself, not when the app's response got back to us
        ready = self._ready.pop(external_id, None)
        prev = self._last_swipe_ms if self._last_swipe_ms is not None and self._last_swipe_ms <= decided else None
        if ready is not None and ready > decided:
            ready = None
        if prev is None and ready is None:
            if not events:
                self._last_swipe_ms = decided
                return None  # nothing says when this card appeared
            start, basis = min(e["t"] for e in events), "first_event"
        elif ready is not None and (prev is None or ready > prev):
            start, basis = ready, "ready"  # the card only appeared once its data had arrived
        else:
            start, basis = prev, "prev_swipe"
        self._last_swipe_ms = decided
        raw = {"start": start, "decided": decided, "basis": basis, "events": events}
        if self._session is None:
            self._session = db.start_session(conn, self.source, self.device, start / 1000)
        db.end_session(conn, self._session, decided / 1000)
        return db.add_interaction(
            conn, self._session, self.source, origin, decided / 1000, "pointer" if events else "timing", raw,
            behavior.derive(raw), behavior.FEATURE_VERSION, profile_id=profile_id, external_id=external_id,
            rating_id=rating_id, liked=liked)
