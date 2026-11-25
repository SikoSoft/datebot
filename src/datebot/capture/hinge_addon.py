"""mitmproxy addon: record Hinge profiles and your own swipes while you use the app on your iPhone.

Run it with `datebot capture hinge` (needs the `phone` extra and the phone set up as in the readme). Read-only: it
never modifies, replays or sends anything; you swipe in the app yourself. Profiles are stored locally with their
photos; your swipes become `app` ratings (a like = "note", a pass = "skip", see capture/hinge.py).
Photos are downloaded in a background thread so the phone's traffic is never held up.
"""
from __future__ import annotations

import json
import queue
import threading

from datebot import config
from datebot.capture import hinge
from datebot.capture.recorder import Recorder

SOURCE = "hinge"


def prompts_cache_path():
    return config.data_dir() / "hinge-prompts.json"


def load_prompts() -> dict[str, str]:
    try:
        return json.loads(prompts_cache_path().read_text())
    except (OSError, ValueError):
        return {}


class HingeCapture:
    def __init__(self, recorder: Recorder | None = None) -> None:
        self.recorder = recorder or Recorder(source=SOURCE)
        self.session = hinge.HingeSession(load_prompts())
        self._events: queue.Queue = queue.Queue()
        self._worker = threading.Thread(target=self._work, daemon=True)
        self._worker.start()

    def _work(self) -> None:
        while True:
            event = self._events.get()
            try:
                if event.kind == "profile":
                    self.recorder.record_profiles([event.profile])
                else:
                    self.recorder.record_swipe(event.swipe)
            except Exception as e:  # one bad profile must not stop the capture
                print(f"[hinge] skipped one event: {type(e).__name__}: {e}")
            finally:
                self._events.task_done()

    def response(self, flow) -> None:
        if flow.request.pretty_host != hinge.HOST:
            return
        try:
            events = self.session.handle(
                flow.request.path.split("?")[0], flow.response.status_code,
                flow.request.get_text(strict=False), flow.response.get_text(strict=False))
        except Exception as e:
            print(f"[hinge] could not parse {flow.request.path[:40]}: {type(e).__name__}")
            return
        if self.session.prompts_changed:
            self.session.prompts_changed = False
            prompts_cache_path().parent.mkdir(parents=True, exist_ok=True)
            prompts_cache_path().write_text(json.dumps(self.session.prompts))
        for e in events:
            self._events.put(e)

    def done(self) -> None:
        self._events.join()
        c = self.recorder.counts
        print(f"[hinge] {c.new} new profile(s), {c.swipes} swipe(s) recorded"
              + (f", {c.unknown_swipes} on uncaptured profiles" if c.unknown_swipes else "")
              + (f", {c.failed} skipped (photos failed)" if c.failed else "")
              + (f"; unrecognised ratings {dict(self.session.unknown_ratings)}" if self.session.unknown_ratings else "")
              + ".")


addons = [HingeCapture()]
