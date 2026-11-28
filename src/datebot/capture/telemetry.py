"""Browser input telemetry for behaviour interactions (see behavior.py for every event kind).

POLICY: record generously. This data is the only source for how you really use the apps, and the behaviour model is
meant to be accurate, so anything that might be needed later (device, window, layout, raw input at full rate) is
captured now: it cannot be recovered afterwards. Stored raw events are cheap (zlib JSON); features are derived later.

A passive script is injected into the page you drive. It only listens and hands batches to this process; it sends
nothing to the site. Recorded: pointer position at full rate (including the browser's coalesced sub-frame samples),
buttons and pressure, mouse/pointer down and up, hover changes, wheel with all axes and delta mode, scroll positions,
touch, clicks with position inside the clicked element and a short descriptor, window focus/visibility, plus context
that is carried into every interaction: device and browser (`env`), window/screen/zoom geometry (`v`) and the
on-screen rectangles of buttons, images and card/gallery containers (`ly`).
Limits that stay:
  * Only the *names* of arrow keys are recorded (with modifier flags); every other keystroke is ignored, so typed
    messages never leave the page.
  * Descriptors (`tg`, and `ly` items) hold interface vocabulary only: tag, role, aria-label/title, test ids, class
    tokens, and the text of short buttons (click descriptors only). Layout items accept an aria-label only when it is
    short letters-and-spaces (so "next photo" is kept but a label with a name or number is not). Photos, bios and
    names are never read; image sources are never read.
  * The script is ordinary page JavaScript and exposes a function named __datebot_events, so the page could see it.
Events carry epoch-millisecond timestamps (performance.timeOrigin + the event's own timeStamp, else performance.now()),
the same clock as time.time().
"""
from __future__ import annotations

import json
import threading

BINDING = "__datebot_events"
MAX_EVENTS = 200_000  # a runaway page can't grow this without bound; the oldest events go first
CONTEXT_KINDS = ("env", "v", "ly")  # state, not input: see behavior.CONTEXT_KINDS

INIT_SCRIPT = """
(() => {
  if (window.__datebotTelemetry) return;
  window.__datebotTelemetry = true;
  const q = [];
  const origin = performance.timeOrigin;
  const now = () => Math.round((origin + performance.now()) * 10) / 10;
  const et = e => (e && e.timeStamp > 0 ? Math.round((origin + e.timeStamp) * 10) / 10 : now());  // when the input happened
  const send = () => {
    if (!q.length || typeof window.__datebot_events !== 'function') return;
    try { window.__datebot_events(JSON.stringify(q.splice(0))); } catch (_) {}
  };
  const clip = (v, n) => (typeof v === 'string' ? v.trim().toLowerCase().replace(/\\s+/g, ' ').slice(0, n) : '');
  const tokens = c => (typeof c.className === 'string' ? c.className : '').split(/\\s+/).filter(x => /^[A-Za-z0-9_-]{2,32}$/.test(x)).slice(0, 6).join(' ').toLowerCase();
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
    const cls = tokens(c);
    if (cls) d.cl = cls;
    if (/^(button|a)$/.test(d.tag) || d.role === 'button') {
      const tx = clip(c.textContent, 25);
      if (tx && tx.length < 25) d.tx = tx;
    }
    return d;
  };
  // Stricter descriptor for layout snapshots: no text at all, and an aria-label/title only if it is short plain words.
  const plain = v => { const x = clip(v, 30); return /^[a-z ]{2,30}$/.test(x) ? x : ''; };
  const describeLayout = c => {
    const d = { tag: c.tagName.toLowerCase() };
    const role = clip(c.getAttribute('role'), 20); if (role) d.role = role;
    const al = plain(c.getAttribute('aria-label')); if (al) d.al = al;
    const ti = plain(c.getAttribute('title')); if (ti) d.ti = ti;
    for (const [k, a] of [['id', 'data-testid'], ['qa', 'data-qa'], ['dt', 'data-test']]) { const v = clip(c.getAttribute(a), 40); if (v) d[k] = v; }
    const cls = tokens(c); if (cls) d.cl = cls;
    return d;
  };
  const rect = el => { const r = el.getBoundingClientRect(); return [Math.round(r.left * 10) / 10, Math.round(r.top * 10) / 10, Math.round(r.width * 10) / 10, Math.round(r.height * 10) / 10]; };
  const push = (e, flush) => { q.push(e); if (flush) send(); };
  const opt = { capture: true, passive: true };
  const on = (target, type, fn) => target.addEventListener(type, e => { try { fn(e); } catch (_) {} }, opt);
  setInterval(send, 250);

  // ---- context: device, window geometry, on-screen layout (carried into every interaction by the Python side) ----
  const env = () => {
    const mm = x => matchMedia(x).matches;
    let tz = ''; try { tz = Intl.DateTimeFormat().resolvedOptions().timeZone; } catch (_) {}
    return { t: now(), k: 'env', ua: navigator.userAgent, plat: navigator.platform, lang: navigator.language,
      langs: (navigator.languages || []).join(','), tz, hc: navigator.hardwareConcurrency, dm: navigator.deviceMemory,
      mtp: navigator.maxTouchPoints, cd: screen.colorDepth, pointer: mm('(pointer:fine)') ? 'fine' : mm('(pointer:coarse)') ? 'coarse' : 'none',
      hover: mm('(hover:hover)'), dark: mm('(prefers-color-scheme: dark)'), reduced: mm('(prefers-reduced-motion: reduce)') };
  };
  const view = () => {
    const vv = window.visualViewport;
    return { t: now(), k: 'v', iw: innerWidth, ih: innerHeight, ow: outerWidth, oh: outerHeight, sx: screenX, sy: screenY,
      sw: screen.width, sh: screen.height, aw: screen.availWidth, ah: screen.availHeight, dpr: devicePixelRatio,
      vz: vv ? vv.scale : 1, vw: vv ? vv.width : innerWidth, vh: vv ? vv.height : innerHeight, vox: vv ? vv.offsetLeft : 0, voy: vv ? vv.offsetTop : 0,
      scx: scrollX, scy: scrollY, fs: document.fullscreenElement ? 1 : 0, host: location.hostname, sec: location.pathname.split('/')[1] || '',
      vis: document.visibilityState };
  };
  let lastView = '';
  const pushView = force => {
    const v = view();
    const sig = JSON.stringify([v.iw, v.ih, v.ow, v.oh, v.sx, v.sy, v.sw, v.sh, v.dpr, v.vz, v.fs, v.sec]);
    if (force || sig !== lastView) { lastView = sig; push(v); }
  };
  const LAYOUT_SEL = 'button, [role=button], a[href], img, video, canvas, [class*=card i], [class*=slider i], [class*=photo i], [class*=gallery i], [class*=carousel i]';
  let lastLayout = '', layoutAt = 0, layoutTimer = 0;
  const pushLayout = () => {
    layoutTimer = 0; layoutAt = Date.now();
    const items = [];
    for (const el of document.querySelectorAll(LAYOUT_SEL)) {
      if (items.length >= 80) break;
      const r = el.getBoundingClientRect();
      if (r.width < 4 || r.height < 4 || r.bottom < 0 || r.right < 0 || r.top > innerHeight || r.left > innerWidth) continue;
      items.push({ d: describeLayout(el), r: rect(el) });
    }
    const sig = JSON.stringify(items);
    if (sig !== lastLayout) { lastLayout = sig; push({ t: now(), k: 'ly', items }); }
  };
  const scheduleLayout = () => {
    if (layoutTimer) return;
    layoutTimer = setTimeout(pushLayout, Math.max(150, 750 - (Date.now() - layoutAt)));  // at most about once per 750ms
  };
  const startContext = () => {
    push(env()); pushView(true); scheduleLayout();
    new MutationObserver(scheduleLayout).observe(document.documentElement, { childList: true, subtree: true, attributes: true, attributeFilter: ['class', 'style', 'aria-label'] });
    setInterval(() => pushView(false), 1000);  // window moved/resized/zoomed/fullscreen without a resize event
  };
  if (document.documentElement) startContext(); else document.addEventListener('DOMContentLoaded', startContext, { once: true });
  on(window, 'resize', () => { pushView(false); scheduleLayout(); });
  on(window, 'scroll', scheduleLayout);
  on(document, 'fullscreenchange', () => pushView(true));
  if (window.visualViewport) { on(visualViewport, 'resize', () => pushView(false)); on(visualViewport, 'scroll', () => pushView(true)); }

  // ---- pointer ----
  on(window, 'mousemove', e => { const ev = { t: et(e), k: 'm', x: e.clientX, y: e.clientY }; if (e.buttons) ev.b = e.buttons; push(ev); });
  on(window, 'pointermove', e => {  // the sub-frame samples the browser merged into one mousemove
    const cs = e.getCoalescedEvents ? e.getCoalescedEvents() : [];
    if (cs.length < 2) return;
    for (const c of cs) { const ev = { t: et(c), k: 'mc', x: c.clientX, y: c.clientY }; if (c.pressure) ev.pr = Math.round(c.pressure * 1000) / 1000; push(ev); }
  });
  const pointerEv = (k, e) => {
    const ev = { t: et(e), k, x: e.clientX, y: e.clientY, b: e.button, pt: e.pointerType, pr: Math.round((e.pressure || 0) * 1000) / 1000 };
    if (e.width > 1 || e.height > 1) { ev.cw = e.width; ev.ch = e.height; }
    if (e.target && e.target.closest) {
      try { ev.tg = describe(e.target); const el = e.target.closest('button, [role=button]') || e.target; const r = el.getBoundingClientRect(); ev.r = [Math.round(r.left), Math.round(r.top), Math.round(r.width), Math.round(r.height)]; } catch (_) {}
    }
    push(ev, k === 'md');
  };
  on(window, 'pointerdown', e => pointerEv('md', e));
  on(window, 'pointerup', e => pointerEv('mu', e));
  on(window, 'pointercancel', e => pointerEv('mx', e));
  on(window, 'dblclick', e => push({ t: et(e), k: 'dc', x: e.clientX, y: e.clientY }));
  on(window, 'contextmenu', e => push({ t: et(e), k: 'cm', x: e.clientX, y: e.clientY }));
  let hoverEl = null;
  on(window, 'mouseover', e => {  // hover changes between interactive things (buttons, links, images), with where they are
    const el = e.target && e.target.closest ? e.target.closest('button, [role=button], a[href], img, video') : null;
    if (el === hoverEl) return;
    hoverEl = el;
    const ev = { t: et(e), k: 'ov', x: e.clientX, y: e.clientY };
    if (el) { ev.tg = describeLayout(el); ev.r = rect(el); }
    push(ev);
  });
  on(document.documentElement || window, 'mouseleave', e => push({ t: et(e), k: 'ml', x: e.clientX, y: e.clientY }));
  on(document.documentElement || window, 'mouseenter', e => push({ t: et(e), k: 'me', x: e.clientX, y: e.clientY }));

  // ---- wheel, scroll, touch ----
  on(window, 'wheel', e => push({ t: et(e), k: 'w', dy: e.deltaY, dx: e.deltaX, dz: e.deltaZ, dm: e.deltaMode, x: e.clientX, y: e.clientY, ctrl: e.ctrlKey ? 1 : 0 }));
  const pending = new Map();
  on(window, 'scroll', e => {
    const t = e.target, key = t;
    if (pending.has(key)) return;
    pending.set(key, 1);
    requestAnimationFrame(() => {
      pending.delete(key);
      const doc = t === document || t === document.documentElement || !t.getBoundingClientRect;
      const ev = { t: now(), k: 'sc', sx: doc ? scrollX : t.scrollLeft, sy: doc ? scrollY : t.scrollTop };
      if (!doc) { ev.tg = describeLayout(t); }
      push(ev);
    });
  });
  const touchEv = (k, e) => push({ t: et(e), k, n: e.touches.length, p: Array.from(e.changedTouches).slice(0, 5).map(x =>
    ({ id: x.identifier, x: x.clientX, y: x.clientY, rx: x.radiusX, ry: x.radiusY, f: x.force })) });
  for (const [type, k] of [['touchstart', 'ts'], ['touchmove', 'tm'], ['touchend', 'te'], ['touchcancel', 'tc']]) on(window, type, e => touchEv(k, e));

  // ---- keyboard: arrow-key names only ----
  const arrow = e => typeof e.key === 'string' && e.key.startsWith('Arrow');
  const mods = e => (e.shiftKey ? 1 : 0) | (e.ctrlKey ? 2 : 0) | (e.altKey ? 4 : 0) | (e.metaKey ? 8 : 0);
  on(window, 'keydown', e => {
    if (!arrow(e)) return;
    if (e.repeat) push({ t: et(e), k: 'kr', key: e.key, mod: mods(e) });
    else push({ t: et(e), k: 'kd', key: e.key, mod: mods(e) }, true);
  });
  on(window, 'keyup', e => { if (arrow(e)) push({ t: et(e), k: 'ku', key: e.key, mod: mods(e) }); });

  // ---- clicks, focus ----
  on(window, 'click', e => {
    const el = e.target && e.target.closest ? (e.target.closest('button, [role=button]') || e.target) : null;
    const r = el && el.getBoundingClientRect ? el.getBoundingClientRect() : null;
    const ev = { t: et(e), k: 'c', x: e.clientX, y: e.clientY };
    if (r && r.width > 0 && r.height > 0) { ev.rx = (e.clientX - r.left) / r.width; ev.ry = (e.clientY - r.top) / r.height; ev.r = [Math.round(r.left), Math.round(r.top), Math.round(r.width), Math.round(r.height)]; }
    if (e.target && e.target.closest) { try { ev.tg = describe(e.target); } catch (_) {} }
    push(ev, true);
    scheduleLayout();
  });
  on(window, 'blur', () => push({ t: now(), k: 'f', v: 0 }));
  on(window, 'focus', () => push({ t: now(), k: 'f', v: 1 }));
  on(document, 'visibilitychange', () => push({ t: now(), k: 'f', v: document.hidden ? 0 : 1 }));
})();
"""


class EventBuffer:
    """Collects the page's input events and hands out the ones up to a swipe."""

    def __init__(self) -> None:
        self._events: list[dict] = []
        self._carry: dict[str, dict] = {}  # the latest context event of each kind drained so far
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
        """Remove and return every event up to `until_ms`, in time order.

        Context events (env/v/ly) are state that holds until it changes, so each drain that contains input also gets
        a copy (`carried: 1`, stamped at the first event) of the latest one of each kind from earlier drains; a swipe
        therefore always carries the device, window and layout it happened under."""
        with self._lock:
            taken = [e for e in self._events if e["t"] <= until_ms]
            self._events = [e for e in self._events if e["t"] > until_ms]
            taken.sort(key=lambda e: e["t"])
            have = {e["k"] for e in taken if e["k"] in CONTEXT_KINDS}
            carry = [{**self._carry[k], "carried": 1, "t": taken[0]["t"]} for k in CONTEXT_KINDS
                     if k not in have and k in self._carry and any(e["k"] not in CONTEXT_KINDS for e in taken)]
            for e in taken:
                if e["k"] in CONTEXT_KINDS:
                    self._carry[e["k"]] = e
        return carry + taken


def install(context, buffer: EventBuffer) -> None:
    """Inject the listener into every current and future page of `context` (existing pages need a reload)."""
    context.expose_function(BINDING, buffer.add)
    context.add_init_script(INIT_SCRIPT)
