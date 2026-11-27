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
