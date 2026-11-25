import io
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from PIL import Image

pytest.importorskip("playwright")
if not os.path.exists("/Applications/Google Chrome.app"):
    pytest.skip("needs Google Chrome", allow_module_level=True)

from playwright.sync_api import sync_playwright  # noqa: E402

from datebot import db  # noqa: E402
from datebot.capture import browser  # noqa: E402
from datebot.capture.recorder import Recorder  # noqa: E402

PAGE = b"""<script>
(async () => {
  await fetch('/v2/recs/core?locale=en');
  await fetch('/like/u1?locale=en');
  await fetch('/pass/u2');
  await fetch('/like/stranger');
})();
</script>"""


def _png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (4, 4), "red").save(buf, "PNG")
    return buf.getvalue()


@pytest.fixture
def fake_tinder():
    port_holder = {}

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            path = self.path.split("?")[0]
            if path == "/":
                body, ctype = PAGE, "text/html"
            elif path == "/v2/recs/core":
                port = port_holder["port"]
                users = [{"type": "user", "user": {"_id": i, "name": i.upper(), "birth_date": "1995-01-01",
                          "photos": [{"url": f"http://127.0.0.1:{port}/photo/{i}.png"}]}} for i in ("u1", "u2")]
                body, ctype = json.dumps({"data": {"results": users}}).encode(), "application/json"
            elif path.startswith("/photo/"):
                body, ctype = _png() + path.encode(), "image/png"
            else:  # /like/*, /pass/*
                body, ctype = b"{}", "application/json"
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = HTTPServer(("127.0.0.1", 0), H)
    port_holder["port"] = server.server_port
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}/"
    server.shutdown()


@pytest.mark.slow
def test_attach_records_profiles_and_swipes_from_live_browser_traffic(fake_tinder, tmp_path, monkeypatch):
    monkeypatch.setenv("DATEBOT_HOME", str(tmp_path))
    db_path = str(tmp_path / "t.db")
    logs: list[str] = []
    rec = Recorder(db_path=db_path)
    dump = tmp_path / "dump"
    with sync_playwright() as pw:
        chrome = pw.chromium.launch(channel="chrome", headless=True)
        ctx = chrome.new_context()
        browser.attach(ctx, rec, hosts=("127.0.0.1",), dump_dir=dump, log=logs.append)
        page = ctx.new_page()
        page.goto(fake_tinder)
        page.wait_for_timeout(1500)
        chrome.close()

    conn = db.connect(db_path)
    u1, u2 = (db.find_profile_id(conn, "tinder", i) for i in ("u1", "u2"))
    assert u1 and u2
    assert db.latest_ratings(conn) == {u1: True, u2: False}
    assert db.get_profile(conn, u1).age is not None
    assert (rec.counts.new, rec.counts.swipes, rec.counts.unknown_swipes) == (2, 2, 1)
    assert any("recs: 2 profiles" in m for m in logs)
    assert len(list(dump.glob("*-recs.json"))) == 1 and len(list(dump.glob("*-like.json"))) == 2


CLICK_PAGE = b"""<button id=like style="position:fixed;left:100px;top:100px;width:80px;height:40px">like</button>
<script>
fetch('/v2/recs/core?locale=en');
document.getElementById('like').addEventListener('click', () => fetch('/like/u1?locale=en'));
</script>"""


@pytest.mark.slow
def test_pointer_telemetry_reaches_the_interaction_for_a_real_click(fake_tinder, tmp_path, monkeypatch):
    from datebot.capture import telemetry
    from datebot.interactions import InteractionTracker

    monkeypatch.setenv("DATEBOT_HOME", str(tmp_path))
    db_path = str(tmp_path / "t.db")
    rec = Recorder(db_path=db_path, tracker=InteractionTracker("tinder", "desktop-chrome"))
    events = telemetry.EventBuffer()
    with sync_playwright() as pw:
        chrome = pw.chromium.launch(channel="chrome", headless=True)
        ctx = chrome.new_context()
        telemetry.install(ctx, events)
        browser.attach(ctx, rec, hosts=("127.0.0.1",), log=lambda m: None, events=events)
        page = ctx.new_page()
        page.route("**/", lambda route: route.fulfill(body=CLICK_PAGE, content_type="text/html"))
        page.goto(fake_tinder)
        page.wait_for_timeout(600)
        page.mouse.move(10, 10)
        page.mouse.move(80, 90, steps=8)
        page.keyboard.type("secret")  # must never be recorded
        page.mouse.move(140, 120, steps=8)
        page.mouse.click(140, 120)
        page.wait_for_timeout(800)
        chrome.close()

    import json
    row = db.interactions(db.connect(db_path))[0]
    f = json.loads(row["features"])
    assert row["signal"] == "pointer" and row["profile_id"] and row["rating_id"]
    assert f["n_moves"] >= 17 and f["swipe_by_key"] == 0 and abs(f["click_rx"] - 0.5) < 0.1
    assert "n_keys" not in f
    assert "secret" not in json.dumps(db.raw_of(row))


@pytest.mark.slow
def test_click_descriptor_reaches_the_events_without_reading_profile_text():
    from datebot.capture import telemetry

    page_html = """<div id=bio style="position:fixed;left:0;top:0;width:100px;height:40px">Anna loves hiking</div>
<button aria-label="Next Photo" style="position:fixed;left:200px;top:0;width:60px;height:40px"></button>
<button style="position:fixed;left:300px;top:0;width:60px;height:40px">Like</button>"""
    events = telemetry.EventBuffer()
    with sync_playwright() as pw:
        chrome = pw.chromium.launch(channel="chrome", headless=True)
        ctx = chrome.new_context()
        telemetry.install(ctx, events)
        page = ctx.new_page()
        page.route("**/", lambda route: route.fulfill(body=page_html, content_type="text/html"))
        page.goto("http://127.0.0.1:1/")
        page.wait_for_timeout(300)
        for x in (50, 230, 330):
            page.mouse.click(x, 20)
        page.wait_for_timeout(600)
        chrome.close()
    clicks = [e for e in events.drain(1e18) if e["k"] == "c"]
    from datebot import behavior
    assert [behavior.target_label(c["tg"]) for c in clicks] == ["other", "gallery_next", "like"]
    assert "anna" not in json.dumps(clicks).lower()
