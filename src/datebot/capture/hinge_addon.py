"""mitmproxy addon: record Hinge profiles and your own swipes while you use the app on your iPhone.

Run it with `datebot capture hinge` (needs the `phone` extra and the phone set up as in the readme). Read-only: it
never modifies, replays or sends anything; you swipe in the app yourself. Profiles are stored locally with their
photos; your swipes become `app` ratings (a like = "note", a pass = "skip", see capture/hinge.py).
Photos are downloaded in a background thread so the phone's traffic is never held up.
Each swipe also stores a behaviour interaction. Through a proxy only network timing is visible, so that is just the
dwell: time from the previous swipe (or the profile's arrival) to the swipe request. No touch or scroll data.
"""
from __future__ import annotations

import json
import queue
import threading

from datebot import config
from datebot.interactions import InteractionTracker
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
        self.recorder = recorder or Recorder(source=SOURCE, tracker=InteractionTracker(SOURCE, "iphone-proxy"))
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
                    self.recorder.record_swipe(
                        event.swipe, at_ms=None if event.at is None else event.at * 1000)
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
        tracker = self.recorder.tracker
        for e in events:
            if e.kind == "profile":
                e.at = flow.response.timestamp_end or flow.request.timestamp_start
                if tracker:  # stamped here, not when the worker gets to it, so photo downloads don't delay it
                    tracker.profile_ready(e.profile.external_id, e.at * 1000)
            else:
                e.at = flow.request.timestamp_start
            self._events.put(e)

    def tls_failed_server(self, data) -> None:
        """Name the host whose certificate the proxy could not verify (mitmproxy only logs the phone's address)."""
        conn = data.conn
        host = conn.sni or (conn.address[0] if conn.address else "?")
        print(f"[hinge] upstream TLS failed for {host}{' (Hinge API)' if host == hinge.HOST else ''}: {conn.error}")

    def tls_failed_client(self, data) -> None:
        """The phone rejected the proxy's certificate: not trusted, or the app pins its certificate."""
        conn = data.conn
        print(f"[hinge] phone TLS failed for {conn.sni or '?'}: {conn.error}")

    def done(self) -> None:
        self._events.join()
        c = self.recorder.counts
        print(f"[hinge] {c.new} new profile(s), {c.swipes} swipe(s) recorded"
              + (f", {c.unknown_swipes} on uncaptured profiles" if c.unknown_swipes else "")
              + (f", {c.failed} skipped (photos failed)" if c.failed else "")
              + (f"; unrecognised ratings {dict(self.session.unknown_ratings)}" if self.session.unknown_ratings else "")
              + ".")


addons = [HingeCapture()]
