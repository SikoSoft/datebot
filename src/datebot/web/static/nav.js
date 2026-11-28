// Shared top navigation for every page of the UI. Add `<script src="/static/nav.js"></script>` as the first
// element of <body>; to add a page, add one line to PAGES (the server must serve it at that path).
(() => {
  const PAGES = [
    ['/', 'Rate'],
    ['/review', 'Review'],
    ['/browse', 'Browse'],
    ['/insights', 'Insights'],
    ['/wants', 'Wants'],
    ['/ideal', 'Ideal'],
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
