"""mitmproxy addon that probes whether Hinge's phone traffic is readable (Hinge is mobile-only).

Run it through mitmdump with the phone's Wi-Fi proxy pointed at this Mac (see readme, "Hinge probe"):

    mitmdump -s src/datebot/capture/hinge_probe.py

Read-only: it records exchanges and never modifies, replays or sends anything. Hinge's hostnames and wire format are
NOT known yet, so every host is recorded (images/fonts/media skipped) and the summary shows which hosts carry
profile-like JSON. TLS handshakes the phone refuses are logged to `tls-failures.txt`; hosts that appear only there
are certificate-pinned, which means this route cannot read them (fall back to iPhone Mirroring + screen parsing).

Dumps hold other people's profile data and are kept under data/ (gitignored). Headers and cookies are never saved.
Stdlib only apart from mitmproxy itself, so it runs inside mitmdump's own environment.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

SKIP_TYPES = ("image/", "video/", "audio/", "font/", "text/css", "javascript", "text/html")
MAX_BODY = 2_000_000


def out_dir() -> Path:
    home = Path(os.environ.get("DATEBOT_HOME") or Path(__file__).resolve().parents[3] / "data")
    d = home / "hinge-probe"
    d.mkdir(parents=True, exist_ok=True)
    return d


def worth_saving(content_type: str) -> bool:
    return not any(t in (content_type or "").lower() for t in SKIP_TYPES)


class HingeProbe:
    def __init__(self) -> None:
        self.dir = out_dir()
        self.n = len(list(self.dir.glob("[0-9]*.json")))

    def response(self, flow) -> None:
        ctype = flow.response.headers.get("content-type", "")
        if not worth_saving(ctype):
            return
        try:
            body = flow.response.get_text(strict=False) or ""
            req_body = flow.request.get_text(strict=False) if flow.request.content else None
        except Exception:
            body, req_body = f"<undecodable {len(flow.response.content or b'')}B {ctype}>", None
        self.n += 1
        (self.dir / f"{self.n:05d}-{flow.request.method}.json").write_text(json.dumps({
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"), "method": flow.request.method, "url": flow.request.pretty_url,
            "status": flow.response.status_code, "content_type": ctype, "request_body": req_body,
            "response_body": body[:MAX_BODY],
        }, indent=2))

    def tls_failed_client(self, data) -> None:
        host = data.context.server.address[0] if data.context.server.address else data.conn.sni or "?"
        with (self.dir / "tls-failures.txt").open("a") as f:
            f.write(f"{time.strftime('%H:%M:%S')} {data.conn.sni or host}\n")


addons = [HingeProbe()]


def summarize(d: Path | None = None, top: int = 25) -> str:
    """Per host: exchanges, largest response, and TLS refusals, so profile-bearing hosts and pinned hosts stand out."""
    from collections import Counter
    from urllib.parse import urlparse

    d = d or out_dir()
    hosts: Counter = Counter()
    biggest: dict[str, tuple[int, str]] = {}
    for f in sorted(d.glob("[0-9]*.json")):
        x = json.loads(f.read_text())
        u = urlparse(x["url"])
        hosts[u.hostname] += 1
        size = len(x["response_body"] or "")
        if size > biggest.get(u.hostname, (-1, ""))[0]:
            biggest[u.hostname] = (size, f"{x['method']} {u.path}")
    lines = [f"{sum(hosts.values())} exchange(s) in {d}", "Readable hosts (count, largest response):"]
    lines += [f"  {n:4d}  {biggest[h][0]:8d}B  {h}  {biggest[h][1]}" for h, n in hosts.most_common(top)]
    fails = d / "tls-failures.txt"
    if fails.exists():
        pinned = Counter(l.split(" ", 1)[1].strip() for l in fails.read_text().splitlines() if " " in l)
        lines.append("Phone refused the proxy certificate (likely pinned, unreadable):")
        lines += [f"  {n:4d}  {h}" for h, n in pinned.most_common(top)]
    return "\n".join(lines)


if __name__ == "__main__":
    print(summarize())
