"""Passively records Badoo profiles and your own swipes from a Chrome window you drive yourself."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from datebot import config
from datebot.capture import badoo, discover
from datebot.capture.browser import Observer, pump
from datebot.capture.recorder import Recorder

SOURCE = "badoo"
START_URL = "https://badoo.com/encounters"
GET_ENCOUNTERS, VOTE = "SERVER_GET_ENCOUNTERS", "SERVER_ENCOUNTERS_VOTE"


def attach(context, recorder: Recorder, log: Callable[[str], None] = print, observer: Observer | None = None) -> None:
    def handle(resp) -> None:
        if not resp.ok or resp.request.method != "POST":
            return
        name = badoo.api_name(resp.url)
        if name == GET_ENCOUNTERS:
            profiles = badoo.parse_encounters(resp.text())
            new = recorder.record_profiles(profiles)
            log(f"encounters: {len(profiles)} profiles ({new} new)")
            if observer:
                observer.on_profiles(profiles)
        elif name == VOTE:
            swipe = badoo.parse_vote(resp.request.post_data)
            if swipe is None:
                log("vote with an unrecognized code; ignored")
                return
            origin = observer.swipe_origin(swipe) if observer else "app"
            recorded = recorder.record_swipe(swipe, origin=origin)
            who = "bot" if origin == "auto" else "you"
            log(f"swipe ({who}): {'liked' if swipe.liked else 'passed'}" if recorded
                else "swipe on a profile we never captured; skipped")
            if observer:
                observer.on_swipe(swipe, recorded, origin)

    def on_response(resp) -> None:
        try:
            handle(resp)
        except Exception as e:  # a parsing problem must never break the browser session
            log(f"capture error on {resp.url[:80]}: {type(e).__name__}: {e}")

    context.on("response", on_response)


def run(recorder: Recorder, cdp: str | None = None, dump_dir: Path | None = None,
        log: Callable[[str], None] = print, observer: Observer | None = None,
        tick: Callable[[object], bool] | None = None, tick_ms: int = 500) -> None:
    """Open a normal Chrome (Google/phone sign-in works) or attach to yours, and record until it is closed."""
    from playwright.sync_api import sync_playwright

    proc = None
    with sync_playwright() as pw:
        if cdp:
            endpoint = cdp
            log("Attached. Reload the Badoo tab so its profiles are captured.")
        else:
            profile = config.data_dir() / "browser" / "badoo"
            profile.mkdir(parents=True, exist_ok=True)
            proc = discover.launch_chrome(profile, START_URL)
            endpoint = "http://127.0.0.1:9222"
            log("Chrome opened (logins are remembered). Swipe as usual on Badoo; close the window to stop.")
        try:
            context = pw.chromium.connect_over_cdp(endpoint).contexts[0]
            attach(context, recorder, log, observer)
            if dump_dir:
                discover.attach(context, dump_dir, badoo.HOSTS, log)
            pump(context, tick_ms, (lambda: tick(context)) if tick else None)
        finally:
            if proc and proc.poll() is None:
                proc.terminate()
