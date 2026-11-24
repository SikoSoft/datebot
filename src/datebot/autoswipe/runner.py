"""The auto-swipe controller.

It watches the same Tinder traffic as capture, works out which card is on top, and presses the like/pass
key only when the model is confident. Everything else is left for you, and your own swipes keep
teaching the model. Safety rules, in order of importance:

* Nothing is pressed until the bot has watched you swipe several cards exactly as it predicted
  (Handshake), so its idea of "which card is on top" is proven first.
* After every press it checks the swipe Tinder actually received (right profile, right direction).
  Any surprise halts the bot immediately, so a wrong guess costs at most one swipe.
* The first thing it ever does in a run is pass someone (cheaper to get wrong than a like) to confirm
  the arrow keys mean what it thinks.
* Hard limits per run and per day, and a minimum delay between swipes.
* Its swipes are stored with origin 'auto' and are never used as training labels.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, Protocol

from datebot.autoswipe.logic import Bands, CardQueue, Decision, Governor, Handshake, decide
from datebot.capture.recorder import Recorder
from datebot.capture.tinder import CapturedProfile, Swipe

LIKE_KEY, PASS_KEY = "ArrowRight", "ArrowLeft"

Assess = Callable[[str], "tuple[float | None, str | None]"]  # external id -> (model score, hard-filter reason)


class Driver(Protocol):
    def press(self, key: str) -> None: ...

    def wait(self, ms: int) -> None: ...


class AutoSwiper:
    """Implements the capture Observer protocol, plus `step()` to be called repeatedly from the event loop."""

    def __init__(
        self,
        recorder: Recorder,
        assess: Assess,
        bands: Bands,
        governor: Governor,
        handshake: Handshake,
        driver: Driver,
        *,
        live: bool = False,
        filters_pass: bool = True,
        key_timeout: float = 6.0,
        log: Callable[[str], None] = print,
        journal: Callable[[dict], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.recorder, self.assess, self.bands = recorder, assess, bands
        self.governor, self.handshake, self.driver = governor, handshake, driver
        self.live, self.filters_pass, self.key_timeout = live, filters_pass, key_timeout
        self.log, self.journal, self.clock = log, journal, clock

        self.queue = CardQueue()
        self.decisions: dict[str, Decision] = {}
        self.names: dict[str, str] = {}
        self.halted: str | None = None
        self.finished: str | None = None
        self.mapping_verified = False  # becomes True once one bot swipe went the intended way
        self.liked = self.passed = self.left_to_you = 0
        self._announced: str | None = None
        self._pending: tuple[str, str] | None = None  # (user id, "like" | "pass") while a key press is in flight
        self._result: tuple[Swipe, bool] | None = None
        self._unexpected: Swipe | None = None
        self._notes_shown: set[str] = set()

    # ---- Observer protocol (called by the capture listener) ----

    def on_profiles(self, profiles: list[CapturedProfile]) -> None:
        self.queue.add([p.external_id for p in profiles])
        for p in profiles:
            self.names[p.external_id] = p.name or p.external_id
            if p.external_id not in self.decisions:
                try:
                    score, reason = self.assess(p.external_id)
                    self.decisions[p.external_id] = decide(score, self.bands, reason, self.filters_pass)
                except Exception as e:  # a scoring problem must never turn into a swipe
                    self.decisions[p.external_id] = Decision("human", f"scoring failed: {type(e).__name__}: {e}")

    def swipe_origin(self, swipe: Swipe) -> str:
        return "auto" if self._pending and swipe.user_id == self._pending[0] else "app"

    def on_swipe(self, swipe: Swipe, recorded: bool, origin: str) -> None:
        if self._pending and swipe.user_id != self._pending[0]:
            self._unexpected = swipe
        was_top = self.queue.swiped(swipe.user_id)
        if origin == "auto":
            self._result = (swipe, was_top)
            return
        self.handshake.observe(was_top)  # your own swipe: does it match where the bot thought you were?
        if not self.handshake.trusted or self.handshake.streak == self.handshake.required:
            self._say(f"order check: {self.handshake.streak}/{self.handshake.required} swipes matched the prediction"
                      + ("" if was_top else " (this one did not; starting over)"))

    # ---- control loop ----

    @property
    def stopped(self) -> bool:
        return self.halted is not None or self.finished is not None

    def step(self) -> bool:
        """Do whatever is due. Returns False when the bot should stop."""
        if self.stopped:
            return False
        if why := self.governor.exhausted:
            self.finished = why
            self._say(f"Stopping: {why}.")
            return False
        card = self.queue.current()
        if card is None:
            return True
        d = self.decisions.get(card) or Decision("human", "unknown card")
        if card != self._announced:
            self._announced = card
            self._announce(card, d)
        if d.action == "human" or not self.live:
            return True
        if not self.handshake.trusted:
            self._note(f"wait:{self.handshake.streak}", f"Not acting yet: swipe {self.handshake.required} cards "
                       f"yourself first so the bot can verify it follows the card order "
                       f"({self.handshake.streak}/{self.handshake.required}).")
            return True
        if not self.mapping_verified and d.action != "pass":
            self._note(f"verify:{card}", f"Leaving {self.names.get(card, card)} to you: the bot confirms the arrow "
                       "keys on a pass before it will like anyone.")
            return True
        if not self.governor.ready():
            return True
        self._act(card, d)
        return not self.stopped

    def _announce(self, card: str, d: Decision) -> None:
        name = self.names.get(card, card)
        if d.action == "human":
            self.left_to_you += 1
            self._say(f"\a{name}: YOUR TURN ({d.reason})")
        elif self.live and self.handshake.trusted:
            self._say(f"{name}: {d.action} ({d.reason})")
        else:
            self._say(f"{name}: would {d.action} ({d.reason})" + ("" if self.live else "  [observe only]"))
        self._record({"event": "card", "id": card, "name": name, "action": d.action, "reason": d.reason,
                      "score": d.score})

    def _act(self, card: str, d: Decision) -> None:
        name = self.names.get(card, card)
        self._pending, self._result, self._unexpected = (card, d.action), None, None
        try:
            self.driver.press(LIKE_KEY if d.action == "like" else PASS_KEY)
            deadline = self.clock() + self.key_timeout
            while self._result is None and self._unexpected is None and self.clock() < deadline:
                self.driver.wait(100)
        except Exception as e:
            self._pending = None
            return self._halt(f"could not press a key: {type(e).__name__}: {e}")
        result, unexpected, self._pending = self._result, self._unexpected, None

        if unexpected is not None:
            return self._halt(f"a different profile ({unexpected.user_id}) was swiped while acting on {name}")
        if result is None:
            return self._halt(f"no swipe happened within {self.key_timeout:.0f}s of pressing the key for {name}. "
                              "Is a pop-up open (match, out of likes), or is the Tinder cards screen not showing?")
        swipe, was_top = result
        if swipe.liked != (d.action == "like"):
            return self._halt(f"WRONG DIRECTION: wanted to {d.action} {name} but Tinder recorded a "
                              f"{'like' if swipe.liked else 'pass'}. The arrow keys do not mean what the bot assumed.")
        if not was_top:
            return self._halt(f"{name} was not the card the bot thought was on top; card order is not as assumed")
        self.mapping_verified = True
        self.governor.record()
        self.liked += d.action == "like"
        self.passed += d.action == "pass"
        self._say(f"  -> {d.action}ed {name} ({self.governor.done}/{self.governor.run_cap} this run)")
        self._record({"event": "swipe", "id": card, "name": name, "action": d.action, "reason": d.reason,
                      "score": d.score})

    def _halt(self, reason: str) -> None:
        self.halted = reason
        self._say(f"HALTED: {reason}")
        self._record({"event": "halt", "reason": reason})

    # ---- helpers ----

    def _say(self, msg: str) -> None:
        self.log(msg)

    def _note(self, key: str, msg: str) -> None:
        if key not in self._notes_shown:  # say each explanation once, not every tick
            self._notes_shown.add(key)
            self._say(msg)

    def _record(self, entry: dict) -> None:
        if self.journal:
            self.journal({"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "live": self.live, **entry})

    def summary(self) -> str:
        if self.halted:
            head = f"Halted: {self.halted}"
        else:
            head = self.finished or "Stopped"
        acted = f"{self.liked} liked and {self.passed} passed automatically; " if self.live else ""
        return f"{head}\n{acted}{self.left_to_you} card(s) left to you. Order check streak: {self.handshake.streak}."


def file_journal(path: Path) -> Callable[[dict], None]:
    """Append decisions and swipes to a JSON-lines file so every action can be audited later."""
    path.parent.mkdir(parents=True, exist_ok=True)

    def write(entry: dict) -> None:
        with path.open("a") as f:
            f.write(json.dumps(entry) + "\n")

    return write
