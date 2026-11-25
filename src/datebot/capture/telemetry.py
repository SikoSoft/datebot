"""Browser input telemetry for behaviour interactions (see behavior.py).

A small passive script is injected into the page you drive: it only listens (mouse position, wheel, arrow keys,
clicks with their position inside the clicked element and a short descriptor of it, window focus) and hands batches to this process. It
sends nothing to the site. Two limits worth knowing:
  * Only the *names* of arrow keys are recorded; every other keystroke is ignored, so typed messages never leave the page.
  * A click's descriptor (`tg`) holds interface vocabulary only (tag, role, aria-label, title, test ids, class tokens,
    and the text of buttons shorter than 25 characters); behavior.target_label turns it into gallery/open-profile/like/pass.
  * The script is ordinary page JavaScript and exposes a function named __datebot_events, so the page could see it.
Events carry epoch-millisecond timestamps (performance.timeOrigin + performance.now()), the same clock as time.time().
"""
from __future__ import annotations

import json
import threading

BINDING = "__datebot_events"
MAX_EVENTS = 20_000  # a runaway page can't grow this without bound; the oldest events go first

INIT_SCRIPT = """
(() => {
  if (window.__datebotTelemetry) return;
  window.__datebotTelemetry = true;
  const q = [];
  const now = () => Math.round((performance.timeOrigin + performance.now()) * 10) / 10;
  const send = () => {
    if (!q.length || typeof window.__datebot_events !== 'function') return;
    try { window.__datebot_events(JSON.stringify(q.splice(0))); } catch (_) {}
  };
  const clip = (v, n) => (typeof v === 'string' ? v.trim().toLowerCase().replace(/\\s+/g, ' ').slice(0, n) : '');
  // What was clicked, as UI vocabulary only: tag, role, aria-label/title, test ids, a few class tokens and the
  // text of short buttons ("Like"). Never the text of anything else, so bios and names are not read.
  const describe = t => {
    const c = t.closest('button, a, [role=button], [role=link], [role=tab]') || t;
    const d = { tag: c.tagName.toLowerCase() };
    if (t !== c) d.in = t.tagName.toLowerCase();
    for (const [k, a] of [['role', 'role'], ['al', 'aria-label'], ['ti', 'title'], ['id', 'data-testid'], ['qa', 'data-qa'], ['dt', 'data-test']]) {
      const v = clip(c.getAttribute(a), 40);
      if (v) d[k] = v;
    }
    const cls = (typeof c.className === 'string' ? c.className : '').split(/\\s+/).filter(x => /^[A-Za-z0-9_-]{2,32}$/.test(x)).slice(0, 6);
    if (cls.length) d.cl = cls.join(' ').toLowerCase();
    if (/^(button|a)$/.test(d.tag) || d.role === 'button') {
      const tx = clip(c.textContent, 25);
      if (tx && tx.length < 25) d.tx = tx;
    }
    return d;
  };
  const push = (e, flush) => { q.push(e); if (flush) send(); };
  const opt = { capture: true, passive: true };
  setInterval(send, 250);
  addEventListener('mousemove', e => push({ t: now(), k: 'm', x: e.clientX, y: e.clientY }), opt);
  addEventListener('wheel', e => push({ t: now(), k: 'w', dy: e.deltaY }), opt);
  const arrow = e => typeof e.key === 'string' && e.key.startsWith('Arrow');
  addEventListener('keydown', e => { if (arrow(e) && !e.repeat) push({ t: now(), k: 'kd', key: e.key }, true); }, opt);
  addEventListener('keyup', e => { if (arrow(e)) push({ t: now(), k: 'ku', key: e.key }); }, opt);
  addEventListener('click', e => {
    const el = e.target && e.target.closest ? (e.target.closest('button, [role=button]') || e.target) : null;
    const r = el && el.getBoundingClientRect ? el.getBoundingClientRect() : null;
    const ev = { t: now(), k: 'c', x: e.clientX, y: e.clientY };
    if (r && r.width > 0 && r.height > 0) { ev.rx = (e.clientX - r.left) / r.width; ev.ry = (e.clientY - r.top) / r.height; }
    if (e.target && e.target.closest) { try { ev.tg = describe(e.target); } catch (_) {} }
    push(ev, true);
  }, opt);
  addEventListener('blur', () => push({ t: now(), k: 'f', v: 0 }), opt);
  addEventListener('focus', () => push({ t: now(), k: 'f', v: 1 }), opt);
  document.addEventListener('visibilitychange', () => push({ t: now(), k: 'f', v: document.hidden ? 0 : 1 }), opt);
})();
"""


class EventBuffer:
    """Collects the page's input events and hands out the ones up to a swipe."""

    def __init__(self) -> None:
        self._events: list[dict] = []
        self._lock = threading.Lock()

    def add(self, payload: str) -> None:
        try:
            batch = json.loads(payload)
        except ValueError:
            return
        if not isinstance(batch, list):
            return
        good = [e for e in batch if isinstance(e, dict) and isinstance(e.get("t"), (int, float)) and "k" in e]
        with self._lock:
            self._events.extend(good)
            del self._events[:-MAX_EVENTS]

    def drain(self, until_ms: float) -> list[dict]:
        """Remove and return every event up to `until_ms`, in time order."""
        with self._lock:
            taken = [e for e in self._events if e["t"] <= until_ms]
            self._events = [e for e in self._events if e["t"] > until_ms]
        return sorted(taken, key=lambda e: e["t"])


def install(context, buffer: EventBuffer) -> None:
    """Inject the listener into every current and future page of `context` (existing pages need a reload)."""
    context.expose_function(BINDING, buffer.add)
    context.add_init_script(INIT_SCRIPT)
