// Collapse the control panel on a small screen.
//
// On a phone these panels sit on top of the thing they control. Measured at
// 390px wide: CortexCast 43 percent of the screen, Seismic Earth 36, Nazar 39.
// The panel is still wanted, just not all the time, so this adds one small
// button that folds it away and brings it back.
//
// Drop-in. One file, no dependencies, brings its own CSS. The page says which
// element to fold:
//
//   <script>window.PANEL_TOGGLE = { sel: '#panel', label: 'controls' };</script>
//   <script src="panel-toggle.js"></script>
//
// Only active below the breakpoint, so the desktop layout is untouched. No
// storage of any kind: it starts open on every load, which is the honest
// default when a visitor has never seen the page before.

(function () {
'use strict';

const CFG = window.PANEL_TOGGLE;
if (!CFG || !CFG.sel) return;

const MAXW = CFG.maxWidth || 720;   // phones and small tablets only
const LABEL = CFG.label || 'panel';

const CSS = `
.pt-btn{position:fixed;z-index:2147483630;right:14px;bottom:56px;
  display:none;align-items:center;gap:6px;padding:7px 11px;
  background:#12161a;border:1px solid #2f3941;color:#c3cbd2;cursor:pointer;
  border-radius:999px;font:11px/1.4 ui-monospace,Consolas,monospace;
  box-shadow:0 4px 14px rgba(0,0,0,.45)}
.pt-btn:hover{border-color:#7d8992;color:#eaf0f4}
.pt-btn i{font-style:normal;font-size:13px;line-height:1}
@media (max-width:${MAXW}px){
  .pt-btn{display:inline-flex}
  .pt-hidden{display:none !important}
}
`;

function boot() {
  const panels = Array.prototype.slice.call(document.querySelectorAll(CFG.sel));
  if (!panels.length) return;

  const st = document.createElement('style');
  st.textContent = CSS;
  document.head.appendChild(st);

  const btn = document.createElement('button');
  btn.type = 'button';
  btn.className = 'pt-btn';
  btn.setAttribute('aria-expanded', 'true');

  let open = true;
  const paint = () => {
    btn.innerHTML = '<i>' + (open ? '✕' : '≡') + '</i>' + (open ? 'hide ' : 'show ') + LABEL;
    btn.setAttribute('aria-label', (open ? 'Hide the ' : 'Show the ') + LABEL);
    btn.setAttribute('aria-expanded', open ? 'true' : 'false');
    panels.forEach((p) => p.classList.toggle('pt-hidden', !open));
  };

  btn.addEventListener('click', () => { open = !open; paint(); });
  document.body.appendChild(btn);
  paint();

  // If the page grows past the breakpoint the panel must come back, otherwise a
  // visitor who rotates to landscape is left with a panel that cannot return.
  window.addEventListener('resize', () => {
    if (window.innerWidth > MAXW && !open) { open = true; paint(); }
  });
}

if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot);
else boot();
})();
