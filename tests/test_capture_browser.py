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
