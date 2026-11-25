"""Passively watches a Chrome window you drive yourself and records what it sees.

Nothing is clicked or sent: this only listens to the responses Tinder's web app already receives,
so your own browsing (and your own swipes) become training data.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, Protocol

from datebot import config
from datebot.capture import telemetry as input_telemetry, tinder
from datebot.capture.recorder import Recorder

START_URL = "https://tinder.com/app/recs"


def browser_profile_dir() -> Path:
    path = config.data_dir() / "browser" / "tinder"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _dump(dump_dir: Path | None, label: str, url: str, status: int, body: str) -> None:
    if dump_dir is None:
        return
    dump_dir.mkdir(parents=True, exist_ok=True)
    (dump_dir / f"{time.strftime('%H%M%S')}-{time.time_ns() % 10**6}-{label}.json").write_text(
        json.dumps({"url": url, "status": status, "body": body}, indent=2)
    )


class Observer(Protocol):
    def on_profiles(self, profiles: list[tinder.CapturedProfile]) -> None: ...

    def swipe_origin(self, swipe: tinder.Swipe) -> str: ...

    def on_swipe(self, swipe: tinder.Swipe, recorded: bool, origin: str) -> None: ...


def attach(
    context,
    recorder: Recorder,
    hosts: tuple[str, ...] = tinder.HOSTS,
    dump_dir: Path | None = None,
    log: Callable[[str], None] = print,
    observer: "Observer | None" = None,
    events: "input_telemetry.EventBuffer | None" = None,
) -> None:
    """Listen to every response in `context` (all current and future pages).

    An optional `observer` is told about new recommendations and swipes, and chooses the origin a swipe is
    recorded under (the auto-swipe bot uses this to tag its own swipes so they never become training labels).
    `events` (see telemetry.py) supplies the input events behind each swipe for behaviour interactions.
    """
    def handle(resp) -> None:
        url = resp.url
        if not tinder.is_tinder_url(url, hosts) or not resp.ok:
            return
        if resp.request.method not in ("GET", "POST"):
            return  # skips CORS preflights and other noise
        if tinder.is_recs_url(url):
            body = resp.text()
            _dump(dump_dir, "recs", url, resp.status, body)
            profiles = tinder.parse_recs(json.loads(body))
            new = recorder.record_profiles(profiles)
            log(f"recs: {len(profiles)} profiles ({new} new)")
            if observer:
                observer.on_profiles(profiles)
        elif (swipe := tinder.parse_swipe(url)) is not None:
            _dump(dump_dir, "like" if swipe.liked else "pass", url, resp.status, resp.text())
            origin = observer.swipe_origin(swipe) if observer else "app"
            swipe_events = events.drain(time.time() * 1000) if events else None
            if recorder.take_skip(origin):
                log(f"swipe ({'liked' if swipe.liked else 'passed'} {swipe.user_id}): first of the session, "
                    "not recorded (use --keep-first to record it)")
                if observer:
                    observer.on_swipe(swipe, False, origin)
                return
            recorded = recorder.record_swipe(swipe, origin=origin, events=swipe_events)
            who = "bot" if origin == "auto" else "you"
            if recorded:
                log(f"swipe ({who}): {'liked' if swipe.liked else 'passed'} {swipe.user_id}")
            else:
                log(f"swipe on a profile we never captured ({swipe.user_id}); skipped")
            if observer:
                observer.on_swipe(swipe, recorded, origin)

    def on_response(resp) -> None:
        try:
            handle(resp)
        except Exception as e:  # never let a parsing problem break the browser session
            log(f"capture error on {resp.url}: {type(e).__name__}: {e}")

    context.on("response", on_response)


class PageDriver:
    """Presses keys on the Tinder tab, like a person at the keyboard."""

    def __init__(self, page):
        self.page = page

    def press(self, key: str) -> None:
        self.page.keyboard.press(key)

    def wait(self, ms: int) -> None:
        self.page.wait_for_timeout(ms)


def pump(context, tick_ms: int = 500, tick: Callable[[], bool] | None = None) -> None:
    """Keep Playwright's event loop running until every page is closed (or `tick()` returns False)."""
    from playwright.sync_api import Error

    try:
        while context.pages:
            context.pages[0].wait_for_timeout(tick_ms)
            if tick is not None and not tick():
                break
    except Error:
        pass  # browser was closed


def run(
    recorder: Recorder,
    cdp: str | None = None,
    dump_dir: Path | None = None,
    log: Callable[[str], None] = print,
    observer: Observer | None = None,
    tick: Callable[[object], bool] | None = None,
    tick_ms: int = 500,
) -> None:
    """Open (or attach to) Chrome and record until it is closed.

    cdp=None launches Chrome with a dedicated profile under data/browser/ (log in once, it is remembered).
    cdp="http://localhost:9222" attaches to a Chrome you started yourself with --remote-debugging-port.
    """
    from playwright.sync_api import sync_playwright

    with sync_playwright() as pw:
        if cdp:
            context = pw.chromium.connect_over_cdp(cdp).contexts[0]
        else:
            context = pw.chromium.launch_persistent_context(
                str(browser_profile_dir()), channel="chrome", headless=False, no_viewport=True
            )
        events = input_telemetry.EventBuffer() if recorder.tracker else None
        if events:
            input_telemetry.install(context, events)  # before any page loads, so the first page has it too
        attach(context, recorder, dump_dir=dump_dir, log=log, observer=observer, events=events)
        if cdp:
            log("Attached. Reload the Tinder tab so its profiles" + (" and your input timing" if events else "")
                + " are captured.")
        else:
            page = context.pages[0] if context.pages else context.new_page()
            page.goto(START_URL)
            log("Chrome opened. Log in to Tinder yourself, then swipe as usual. Close the window to stop.")
        pump(context, tick_ms, (lambda: tick(context)) if tick else None)
