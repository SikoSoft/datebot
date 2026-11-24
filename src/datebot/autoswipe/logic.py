"""Pure decision logic for auto-swiping (no browser, no database), so it is easy to test and reason about."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Literal

Action = Literal["like", "pass", "human"]


@dataclass(frozen=True)
class Bands:
    """Score cut-offs (0-100). At or above like_at -> like; at or below pass_at -> pass; between -> you decide."""
    like_at: float | None = None
    pass_at: float | None = None

    def __post_init__(self):
        if self.like_at is not None and self.pass_at is not None and self.pass_at >= self.like_at:
            raise ValueError(f"pass_at ({self.pass_at}) must be below like_at ({self.like_at})")

    @property
    def empty(self) -> bool:
        return self.like_at is None and self.pass_at is None


@dataclass(frozen=True)
class Decision:
    action: Action
    reason: str
    score: float | None = None


def decide(score: float | None, bands: Bands, filter_reason: str | None = None, filters_pass: bool = True) -> Decision:
    """Turn a model score (and any hard-filter rejection) into an action. Anything unclear is left to a human."""
    if filter_reason and filters_pass:
        return Decision("pass", f"deal-breaker: {filter_reason}", score)
    if score is None:
        return Decision("human", "no score yet (photos not embedded)", None)
    if bands.like_at is not None and score >= bands.like_at:
        return Decision("like", f"score {score:.0f} >= {bands.like_at:.0f}", score)
    if bands.pass_at is not None and score <= bands.pass_at:
        return Decision("pass", f"score {score:.0f} <= {bands.pass_at:.0f}", score)
    return Decision("human", f"uncertain (score {score:.0f})", score)


class CardQueue:
    """Predicts which profile is on top of Tinder's card stack, from the order recommendations arrived in.

    The order is an assumption about Tinder, so it is verified against every real swipe (see Handshake).
    """

    def __init__(self):
        self._order: list[str] = []
        self._done: set[str] = set()

    def add(self, ids: list[str]) -> None:
        known = set(self._order)
        for i in ids:
            if i not in known and i not in self._done:
                self._order.append(i)
                known.add(i)

    def current(self) -> str | None:
        return self._order[0] if self._order else None

    def swiped(self, user_id: str) -> bool:
        """Note a swipe. True if it was on the card we predicted was on top."""
        was_top = bool(self._order) and self._order[0] == user_id
        self._done.add(user_id)
        if user_id in self._order:
            self._order.remove(user_id)
        return was_top

    def __len__(self) -> int:
        return len(self._order)


class Handshake:
    """The bot may only act after watching you swipe `required` cards in a row exactly as it predicted.

    That proves its idea of "which card is on top" matches reality before it presses anything. One surprise
    (you swiped a different card than predicted) revokes trust until you re-prove it.
    """

    def __init__(self, required: int = 5):
        self.required = required
        self.streak = 0

    def observe(self, matched: bool) -> None:
        self.streak = self.streak + 1 if matched else 0

    @property
    def trusted(self) -> bool:
        return self.streak >= self.required


class Governor:
    """Hard limits on bot activity: swipes per run, swipes per day, and minimum seconds between swipes."""

    def __init__(self, run_cap: int, daily_cap: int, already_today: int = 0, min_delay: float = 3.0,
                 clock: Callable[[], float] = time.monotonic):
        self.run_cap, self.daily_cap, self.min_delay, self.clock = run_cap, daily_cap, min_delay, clock
        self.done = 0
        self.today = already_today
        self._last: float | None = None

    @property
    def exhausted(self) -> str | None:
        """Why the bot must stop for good, or None."""
        if self.done >= self.run_cap:
            return f"run limit reached ({self.run_cap} swipes)"
        if self.today >= self.daily_cap:
            return f"daily limit reached ({self.daily_cap} swipes)"
        return None

    def ready(self) -> bool:
        return self.exhausted is None and (self._last is None or self.clock() - self._last >= self.min_delay)

    def record(self) -> None:
        self.done += 1
        self.today += 1
        self._last = self.clock()
