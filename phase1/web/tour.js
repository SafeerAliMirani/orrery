// First run tour. Coach marks: one element lit at a time, everything else dimmed,
// a card next to it saying what the thing is.
//
// Drop-in. One file, no dependencies, no CDN, no build step. It brings its own
// CSS so it does not care what the host page's stylesheet looks like.
//
// The page declares its steps before loading this:
//
//   <script>window.TOUR_STEPS = [
//     { sel: '#hud', title: 'What you are looking at', body: '...' },
//   ];</script>
//   <script src="tour.js"></script>
//
// A step whose element is missing or hidden is dropped, so a tour never points at
// nothing and pages that build themselves late still work.
//
// No storage of any kind, so it opens once per page load and that is the whole
// memory it has. Esc closes it, the ? button bottom right opens it again.
//
// The spotlight is one div sized to the target with a very large box shadow. The
// shadow is the dim, the div is the hole. pointer-events stays off it so the page
// underneath keeps working while the tour is up.

(function () {
'use strict';

const STEPS = (window.TOUR_STEPS || []).slice();
if (!STEPS.length) return;

const OPTS = window.TOUR_OPTIONS || {};
const PAD = OPTS.pad != null ? OPTS.pad : 6;   // breathing room around the lit element
const CARD_GAP = 12;
const TOP_INSET = OPTS.topInset != null ? OPTS.topInset : 110;  // room for a sticky header

const CSS = `
.tour-spot{position:absolute;z-index:2147483640;border:1px solid #eaf0f4;border-radius:3px;
  box-shadow:0 0 0 9999px rgba(4,6,8,.8);pointer-events:none;
  transition:top .18s ease,left .18s ease,width .18s ease,height .18s ease}
.tour-card{position:absolute;z-index:2147483641;width:330px;max-width:calc(100vw - 20px);
  background:#12161a;border:1px solid #2f3941;padding:12px 14px;color:#c3cbd2;
  font:14px/1.5 system-ui,"Segoe UI",Roboto,sans-serif;box-shadow:0 8px 30px rgba(0,0,0,.5)}
.tour-head{display:flex;align-items:baseline;justify-content:space-between;gap:10px;margin-bottom:6px}
.tour-title{font:600 13px/1.4 system-ui,"Segoe UI",Roboto,sans-serif;color:#eaf0f4}
.tour-count{font:10px/1.4 ui-monospace,Consolas,monospace;color:#7d8992;white-space:nowrap}
.tour-body{margin:0 0 12px;font-size:13px;line-height:1.6;color:#c3cbd2}
.tour-btns{display:flex;justify-content:flex-end;gap:8px}
.tour-card button{font:11px/1.6 ui-monospace,Consolas,monospace;padding:4px 10px;cursor:pointer;
  background:#12161a;border:1px solid #2f3941;color:#c3cbd2}
.tour-card button:hover{border-color:#7d8992;color:#eaf0f4}
.tour-next{border-color:#7d8992 !important;color:#eaf0f4 !important}
.tour-open{position:fixed;right:14px;bottom:14px;z-index:2147483639;width:34px;height:34px;padding:0;
  border-radius:50%;background:#12161a;border:1px solid #2f3941;color:#c3cbd2;cursor:pointer;
  font:400 15px/1 ui-monospace,Consolas,monospace}
.tour-open:hover{border-color:#7d8992;color:#eaf0f4}
@media (prefers-reduced-motion: reduce){.tour-spot{transition:none}}
`;

let steps = [];
let at = 0;
let open = false;
let spot = null;
let card = null;
let opened = false;

const $ = (s) => { try { return document.querySelector(s); } catch (e) { return null; } };

function visible(el) {
  if (!el) return false;
  const r = el.getBoundingClientRect();
  if (r.width < 2 || r.height < 2) return false;
  return getComputedStyle(el).visibility !== 'hidden';
}

function build() {
  const st = document.createElement('style');
  st.textContent = CSS;
  document.head.appendChild(st);

  spot = document.createElement('div');
  spot.className = 'tour-spot';

  card = document.createElement('div');
  card.className = 'tour-card';
  card.setAttribute('role', 'dialog');
  card.innerHTML =
    '<div class="tour-head"><b class="tour-title"></b><span class="tour-count"></span></div>'
    + '<p class="tour-body"></p>'
    + '<div class="tour-btns">'
    + '<button type="button" class="tour-skip">Skip</button>'
    + '<button type="button" class="tour-next">Next</button>'
    + '</div>';
  card.querySelector('.tour-skip').addEventListener('click', close);
  card.querySelector('.tour-next').addEventListener('click', next);
}

// page coordinates, not viewport, so the hole stays put while the page scrolls.
// fast skips the tween: chasing a sticky header at 180ms looks broken.
function place(fast) {
  const step = steps[at];
  const el = $(step.sel);
  if (!el) return next();

  spot.style.transition = fast === true ? 'none' : '';
  const r = el.getBoundingClientRect();
  const top = r.top + window.scrollY - PAD;
  const left = r.left + window.scrollX - PAD;
  const w = r.width + PAD * 2;
  const h = r.height + PAD * 2;

  spot.style.top = top + 'px';
  spot.style.left = left + 'px';
  spot.style.width = w + 'px';
  spot.style.height = h + 'px';

  card.querySelector('.tour-title').textContent = step.title;
  card.querySelector('.tour-body').textContent = step.body;
  card.querySelector('.tour-count').textContent = (at + 1) + ' / ' + steps.length;
  card.querySelector('.tour-next').textContent = at === steps.length - 1 ? 'Done' : 'Next';

  card.style.visibility = 'hidden';
  card.style.top = '0px';
  card.style.left = '0px';
  const cw = card.offsetWidth;
  const ch = card.offsetHeight;

  let ct = top + h + CARD_GAP;
  if (ct + ch > window.scrollY + window.innerHeight - 8) {
    const above = top - ch - CARD_GAP;
    ct = above > window.scrollY + 8 ? above : Math.max(window.scrollY + 8,
      window.scrollY + window.innerHeight - ch - 8);
  }
  let cl = left;
  const maxL = window.scrollX + document.documentElement.clientWidth - cw - 8;
  if (cl > maxL) cl = maxL;
  if (cl < window.scrollX + 8) cl = window.scrollX + 8;

  card.style.top = ct + 'px';
  card.style.left = cl + 'px';
  card.style.visibility = '';
}

function show(i) {
  at = i;
  const el = $(steps[at].sel);
  if (el) {
    const r = el.getBoundingClientRect();
    if (r.top < TOP_INSET || r.bottom > window.innerHeight - 40) {
      window.scrollTo({ top: Math.max(0, r.top + window.scrollY - TOP_INSET), behavior: 'auto' });
    }
  }
  place();
}

function next() {
  if (at >= steps.length - 1) return close();
  show(at + 1);
}

function start() {
  steps = STEPS.filter((s) => visible($(s.sel)));
  if (!steps.length) return;
  open = true;
  opened = true;
  document.body.appendChild(spot);
  document.body.appendChild(card);
  document.addEventListener('keydown', onKey);
  window.addEventListener('resize', onMove);
  // a sticky header moves as you scroll and the hole would be left behind.
  // re-placing on scroll is a no-op for everything else.
  window.addEventListener('scroll', onMove, { passive: true });
  show(0);
}

function close() {
  if (!open) return;
  open = false;
  spot.remove();
  card.remove();
  document.removeEventListener('keydown', onKey);
  window.removeEventListener('resize', onMove);
  window.removeEventListener('scroll', onMove);
}

function onMove() { place(true); }

function onKey(ev) {
  if (ev.key === 'Escape') close();
  else if (ev.key === 'ArrowRight' || ev.key === 'Enter') next();
  else if (ev.key === 'ArrowLeft' && at > 0) show(at - 1);
}

function helpButton() {
  const b = document.createElement('button');
  b.type = 'button';
  b.className = 'tour-open';
  b.textContent = '?';
  b.title = 'What am I looking at';
  b.setAttribute('aria-label', 'Open the tour');
  b.addEventListener('click', () => (open ? close() : start()));
  document.body.appendChild(b);
}

// These pages fetch data and build themselves. Waiting on the first step only was
// not enough: a legend that fills in after the download, or a panel built from a
// manifest, was still empty when the tour opened, so its step was dropped and the
// counter read 1 / 2 instead of 1 / 6. Wait for every step to be real, and fall
// back to whatever exists once the budget is spent.
function whenReady(cb) {
  const t0 = Date.now();
  const budget = OPTS.readyTimeout || 12000;
  (function poll() {
    const all = STEPS.every((s) => visible($(s.sel)));
    if (all || Date.now() - t0 > budget) return cb();
    setTimeout(poll, 250);
  })();
}

function boot() {
  build();
  helpButton();
  whenReady(() => { if (!opened) start(); });
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', boot);
} else {
  boot();
}
})();
