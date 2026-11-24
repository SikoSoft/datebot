"""Traffic discovery for sites whose API we have not mapped yet (Badoo web).

Listens to Chrome, like the Tinder capture, and saves every JSON-ish API exchange to a folder so the real
wire format can be read and a proper parser written from facts instead of guesses. Nothing is clicked or sent.

The dump holds other people's profile data and may contain session identifiers in URLs or bodies: keep it
local and do not share it as-is. Request/response headers and cookies are never saved.
"""
from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse

from datebot import config

BADOO_HOSTS = ("badoo.com",)
BADOO_START_URL = "https://badoo.com/encounters"
MAX_BODY = 2_000_000
TEXTUAL = ("json", "text/plain", "x-www-form-urlencoded", "protobuf", "octet-stream")


def host_matches(url: str, hosts: tuple[str, ...]) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return any(host == h or host.endswith("." + h) for h in hosts)


def worth_saving(url: str, content_type: str, hosts: tuple[str, ...]) -> bool:
    """API-looking exchanges only: skip pages, scripts, styles, images, fonts and other hosts."""
    return host_matches(url, hosts) and any(t in (content_type or "").lower() for t in TEXTUAL)


def save_exchange(out_dir: Path, seq: int, method: str, url: str, status: int, request_body: str | None,
                  response_body: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{seq:05d}-{method}.json"
    path.write_text(json.dumps({
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"), "method": method, "url": url, "status": status,
        "request_body": request_body, "response_body": response_body[:MAX_BODY],
    }, indent=2))
    return path


def attach(context, out_dir: Path, hosts: tuple[str, ...] = BADOO_HOSTS, log: Callable[[str], None] = print) -> None:
    counter = {"n": 0}

    def on_response(resp) -> None:
        try:
            ctype = resp.headers.get("content-type", "")
            if resp.request.method not in ("GET", "POST") or not worth_saving(resp.url, ctype, hosts):
                return
            body = resp.text()
            counter["n"] += 1
            save_exchange(out_dir, counter["n"], resp.request.method, resp.url, resp.status,
                          resp.request.post_data, body)
            if counter["n"] % 10 == 1:
                log(f"saved {counter['n']} API exchange(s) so far")
        except Exception as e:  # binary bodies etc. must never break the session
            log(f"skipped {resp.url[:80]}: {type(e).__name__}")

    context.on("response", on_response)


def summarize(out_dir: Path, top: int = 15) -> str:
    """What was captured, grouped by endpoint, to see at a glance which calls carry profiles and votes."""
    files = sorted(out_dir.glob("*.json")) if out_dir.exists() else []
    if not files:
        return "Nothing saved."
    by_endpoint: Counter = Counter()
    sizes: dict[str, int] = {}
    for f in files:
        d = json.loads(f.read_text())
        key = f"{d['method']} {urlparse(d['url']).path}"
        if (q := urlparse(d["url"]).query):
            key += "?" + q.split("&")[0].split("=")[0]
        by_endpoint[key] += 1
        sizes[key] = max(sizes.get(key, 0), len(d["response_body"] or ""))
    lines = [f"{len(files)} exchange(s) saved to {out_dir}", "Most frequent endpoints (count, largest response):"]
    lines += [f"  {n:4d}  {sizes[k]:8d}B  {k}" for k, n in by_endpoint.most_common(top)]
    return "\n".join(lines)


def run(out_dir: Path, hosts: tuple[str, ...] = BADOO_HOSTS, start_url: str = BADOO_START_URL,
        cdp: str | None = None, log: Callable[[str], None] = print, profile: str = "badoo") -> None:
    from playwright.sync_api import sync_playwright

    from datebot.capture import browser

    with sync_playwright() as pw:
        if cdp:
            context = pw.chromium.connect_over_cdp(cdp).contexts[0]
            log("Attached. Reload the tab so its requests are captured.")
        else:
            d = config.data_dir() / "browser" / profile
            d.mkdir(parents=True, exist_ok=True)
            context = pw.chromium.launch_persistent_context(str(d), channel="chrome", headless=False, no_viewport=True)
            (context.pages[0] if context.pages else context.new_page()).goto(start_url)
            log("Chrome opened. Log in yourself, then swipe ~15 cards (a few likes, a few passes). Close the window to stop.")
        attach(context, out_dir, hosts, log)
        browser.pump(context)
