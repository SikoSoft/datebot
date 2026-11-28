// Shared top navigation for every page of the UI. Add `<script src="/static/nav.js"></script>` as the first
// element of <body>; to add a page, add one line to PAGES (the server must serve it at that path).
// appIcon(source) -> inline element showing the app's logo (falls back to the name); tooltip is the name.
window.appIcon = source => {
  const ICONS = {
    tinder: ['#fe3c72', '<path fill="#fff" d="M12 21c-3.9 0-6.5-2.7-6.5-6.2 0-3.6 2.6-5.6 3.3-9.3 0-.3.4-.5.6-.2 1.2 1.3 1.6 2.8 1.6 3.8.9-.7 1.4-2 1.4-3.2 0-.3.3-.4.5-.3C16.6 7 18.5 10.600 18.500 14.100 18.500 18 16 21 12 21z"/>'],
    badoo: ['#783bf9', '<path fill="#fff" d="M7 5h4.500c2.600 0 4 1.300 4 3.200 0 1.200-.6 2.100-1.600 2.600 1.400.4 2.300 1.500 2.300 3 0 2.100-1.600 3.700-4.300 3.700H7zm2.600 2.200v2.500h1.700c.9 0 1.400-.5 1.400-1.200s-.5-1.300-1.400-1.300zm0 4.700v2.800h2c1 0 1.600-.5 1.600-1.400s-.6-1.400-1.600-1.400z"/>'],
    hinge: ['#1c1c1c', '<path fill="#fff" d="M7 5h2.400v5.300h5.200V5H17v14h-2.400v-6.400H9.400V19H7z"/>'],
  };
  const key = String(source || '').toLowerCase(), icon = ICONS[key];
  const span = document.createElement('span');
  span.title = source || '';
  if (!icon) { span.textContent = source || ''; return span; }
  span.style.cssText = 'display:inline-flex;vertical-align:-4px';
  span.innerHTML = `<svg viewBox="0 0 24 24" width="20" height="20" role="img" aria-label="${key}"><rect width="24" height="24" rx="6" fill="${icon[0]}"/>${icon[1]}</svg>`;
  return span;
};

(() => {
  const PAGES = [
    ['/', 'Rate'],
    ['/review', 'Review'],
    ['/browse', 'Browse'],
    ['/insights', 'Insights'],
    ['/wants', 'Wants'],
    ['/ideal', 'Ideal'],
    ['/behavior', 'Behaviour'],
  ];
  const css = `
  #appnav { position:sticky; top:0; z-index:30; display:flex; align-items:center; gap:4px; height:var(--nav-h); padding:0 12px;
    background:var(--card, #fff); border-bottom:1px solid var(--line, #e3e1dd); overflow-x:auto; white-space:nowrap; }
  #appnav .brand { font-weight:700; margin-right:10px; color:var(--fg, #1d1d1f); font-size:15px; }
  #appnav a.tab { padding:6px 14px; border-radius:8px; color:var(--fg, #1d1d1f); text-decoration:none; font:14px system-ui, sans-serif; }
  #appnav a.tab:hover { background:var(--bg, #f6f5f3); }
  #appnav a.tab[aria-current=page] { background:var(--accent, #3b6fd4); color:#fff; font-weight:600; }
  :root { --nav-h:46px; }`;
  const style = document.createElement('style');
  style.textContent = css;
  document.head.append(style);
  const nav = document.createElement('nav');
  nav.id = 'appnav';
  nav.setAttribute('aria-label', 'Main');
  const brand = document.createElement('span');
  brand.className = 'brand';
  brand.textContent = 'Datebot';
  nav.append(brand);
  const here = location.pathname.replace(/\/+$/, '') || '/';
  for (const [href, label] of PAGES) {
    const a = document.createElement('a');
    a.className = 'tab';
    a.href = href;
    a.textContent = label;
    if (href === here) a.setAttribute('aria-current', 'page');
    nav.append(a);
  }
  document.body.prepend(nav);
})();

// Shared by the rate, review and browse cards: the saved profile details (location, work, prompts, ...) as one row per
// fact with an icon and a divider. `compact` shows only the first rows (review lists), the rest stays on the browse card.
(() => {
  // First match wins; matched against "<section> <field name>". Emoji so no icon files are needed.
  const ICONS = [
    [/location|hometown|city/i, '📍'], [/height/i, '📏'], [/pet|dog|cat/i, '🐾'], [/workout|exercise|gym/i, '🏋️'],
    [/smok|tobacco/i, '🚬'], [/drink|alcohol/i, '🍷'], [/cannabis|weed|420/i, '🌿'], [/kids|children|family/i, '👶'],
    [/zodiac|star sign/i, '✨'], [/education|school|universit/i, '🎓'], [/job|work|company|career/i, '💼'],
    [/looking for|relationship|goal|intention/i, '🔍'], [/language/i, '💬'], [/communicat/i, '💬'],
    [/love|affection/i, '💞'], [/personality|mbti/i, '🧠'], [/religio/i, '🙏'], [/politic/i, '🗳️'], [/sleep/i, '😴'],
    [/diet|vegan|vegetarian|food/i, '🥗'], [/social/i, '📱'], [/ethnic|nationality/i, '🌍'],
  ];
  const icon = key => (ICONS.find(([re]) => re.test(key)) || [null, '•'])[1];
  const make = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; };
  const split = line => { const i = line.indexOf(': '); return i > 0 && i < 60 ? [line.slice(0, i), line.slice(i + 2)] : [null, line]; };

  window.renderDetails = (details, compact) => {
    const box = make('div', 'pdetails');
    let rows = 0;
    const limit = compact ? 6 : Infinity;
    for (const d of details || []) {
      if (d.label === 'Interests') {
        if (rows >= limit) continue;
        const chips = make('div', 'chips');
        d.lines.join(', ').split(', ').forEach(t => chips.append(make('span', 'chip', t)));
        box.append(chips); rows++;
        continue;
      }
      for (const line of d.lines) {
        if (rows >= limit) break;
        const [field, value] = d.label === 'Prompts' || d.label === 'About' ? split(line) : [d.label, line];
        const row = make('div', d.label === 'Prompts' ? 'prow prompt' : 'prow');
        if (d.label !== 'Prompts') row.append(make('span', 'ico', icon(`${d.label} ${field || ''}`)));
        const text = make('div', 'ptext');
        if (field) text.append(make('div', 'pfield', field));
        text.append(make('div', 'pvalue', value));
        row.append(text); box.append(row); rows++;
      }
    }
    return box;
  };
  const st = make('style', null, `
  .pdetails { margin:10px 0; font-size:14px; border-top:1px solid var(--line, #e3e1dd); }
  .prow { display:flex; gap:12px; align-items:flex-start; padding:9px 2px; border-bottom:1px solid var(--line, #e3e1dd); }
  .prow .ico { flex:none; width:24px; text-align:center; font-size:17px; line-height:1.3; }
  .prow .pfield { font-size:11px; text-transform:uppercase; letter-spacing:.04em; color:var(--muted, #777); }
  .prow .pvalue { overflow-wrap:anywhere; white-space:pre-wrap; color:var(--fg, #1d1d1f); }
  .prow.prompt { padding-left:10px; border-left:3px solid var(--line, #e3e1dd); }
  .chips { display:flex; flex-wrap:wrap; gap:6px; padding:10px 2px; border-bottom:1px solid var(--line, #e3e1dd); }
  .chip { padding:3px 10px; border:1px solid var(--line, #e3e1dd); border-radius:99px; font-size:12px; }`);
  document.head.append(st);
})();
