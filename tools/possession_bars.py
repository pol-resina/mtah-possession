"""Possession bars: every submission as a coloured strip over the whole match, and where two differ.

    python tools/possession_bars.py                       # every CSV in evaluation/results/
    python tools/possession_bars.py a.csv b.csv ...       # just these
    python tools/possession_bars.py --out somewhere.html

Writes one self-contained HTML file (default evaluation/reports/possession_bars.html, git-ignored)
and prints a table of how much each submission disagrees with the reference (--ref, default
solution). Open the file in a browser:

    scroll       zoom around the cursor        drag      pan
    hover        every row's label at that frame, and the match clock
    two menus    choose the pair whose differences are drawn in the bottom rows

Colours: home (H) blue, away (A) red, dead ball (D) grey, not graded (X) blank. In the difference
rows: amber = both say the ball is alive but name different teams, purple = one says alive and the
other dead. The public ground truth is drawn as its own row; outside its 5 minutes it is blank.
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "starter"))
from load_smart_tagging import KICKOFF_VIDEO_FRAMES  # noqa: E402

RESULTS = os.path.join(ROOT, "evaluation", "results")
PUBLIC_GT = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
OUT = os.path.join(ROOT, "evaluation", "reports", "possession_bars.html")
FPS = 25


def read_labels(path):
    frame = pd.read_csv(path, dtype={"label": "object"})
    frame.columns = [str(c).strip().lower() for c in frame.columns]
    return frame["label"].fillna("X").astype(str).str.strip().str.upper().values


def run_lengths(labels):
    """[[first frame, label], ...]: one entry per run of equal labels."""
    change = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    starts = np.concatenate([[0], change])
    return [[int(s), str(labels[s])] for s in starts]


def possessions(labels):
    live = np.isin(labels, ("H", "A"))
    runs = run_lengths(np.where(live, labels, "-"))
    return sum(1 for _, label in runs if label in ("H", "A"))


def compare(ref, other):
    graded = (ref != "X") & (other != "X")
    both_live = graded & np.isin(ref, ("H", "A")) & np.isin(other, ("H", "A"))
    alive_dead = graded & ((ref == "D") != (other == "D"))
    return {
        "differ": float(np.mean(ref[graded] != other[graded])) if graded.any() else 0.0,
        "team": float(np.mean(ref[both_live] != other[both_live])) if both_live.any() else 0.0,
        "alive_dead": float(np.mean(alive_dead[graded])) if graded.any() else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("files", nargs="*", help="submission CSVs (default: evaluation/results/*.csv)")
    parser.add_argument("--ref", default="solution", help="reference for the printed table")
    parser.add_argument("--out", default=OUT)
    parser.add_argument("--baselines", action="store_true", help="include the organisers' baselines")
    args = parser.parse_args()

    files = args.files or sorted(glob.glob(os.path.join(RESULTS, "*.csv")))
    if not args.files and not args.baselines:
        files = [f for f in files if not os.path.basename(f).startswith("baseline_")]
    labels = {os.path.splitext(os.path.basename(f))[0]: read_labels(f) for f in files}
    n = len(next(iter(labels.values())))

    rows = []
    if os.path.isfile(PUBLIC_GT):
        gt = read_labels(PUBLIC_GT)
        if len(gt) == n:
            rows.append({"name": "public ground truth", "runs": run_lengths(gt), "gt": True})
    for name, lab in labels.items():
        rows.append({"name": name, "runs": run_lengths(lab), "poss": possessions(lab),
                     "dead": float(np.mean(lab[lab != "X"] == "D"))})

    ref = args.ref if args.ref in labels else next(iter(labels))
    print("{:<26} {:>6} {:>7} {:>10} {:>11}   vs {}".format(
        "", "poss", "D share", "any label", "team (live)", ref))
    for name, lab in labels.items():
        c = compare(labels[ref], lab)
        print("{:<26} {:>6} {:>7.3f} {:>9.1%} {:>11.1%}".format(
            name, possessions(lab), float(np.mean(lab[lab != "X"] == "D")), c["differ"], c["team"]))

    graded = np.flatnonzero(next(iter(labels.values())) != "X")
    k2 = KICKOFF_VIDEO_FRAMES["2nd Half"]
    gap = int(np.argmax(np.diff(graded))) if len(graded) > 1 else 0  # half-time: the longest X run
    meta = {"n": n, "fps": FPS, "first": int(graded[0]) if len(graded) else 0,
            "last": int(graded[-1]) if len(graded) else n - 1,
            "kickoffs": [KICKOFF_VIDEO_FRAMES["1st Half"], k2],
            "h1_end": int(graded[gap]) + 1 if len(graded) > 1 else k2,
            "h2_start": int(graded[gap + 1]) if len(graded) > 1 else k2,
            "ref": ref}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as handle:
        handle.write(PAGE.replace("/*DATA*/", json.dumps({"meta": meta, "rows": rows})))
    print("\nwrote", os.path.relpath(args.out, os.getcwd()))


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Possession bars</title>
<style>
:root { --bg:#fbfbfa; --fg:#1d1f21; --muted:#6b7075; --line:#e2e3e1; --panel:#ffffff;
        --H:#2f6fd6; --A:#d6453d; --D:#a3a8ad; --team:#e8a317; --alive:#8a4fd1; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#141517; --fg:#e7e8e9; --muted:#9a9fa5; --line:#2b2d30; --panel:#1b1c1f;
          --H:#5b8ff0; --A:#ef6a61; --D:#61666c; --team:#f0b43a; --alive:#a57be6; } }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--fg);
       font:14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width:1400px; margin:0 auto; padding:20px 16px 40px; }
h1 { font-size:18px; margin:0 0 4px; }
p.sub { margin:0 0 14px; color:var(--muted); }
.bar { display:flex; flex-wrap:wrap; gap:8px; align-items:center; margin-bottom:12px; }
button, select { font:inherit; color:var(--fg); background:var(--panel); border:1px solid var(--line);
                 border-radius:6px; padding:4px 10px; cursor:pointer; }
button:hover { border-color:var(--muted); }
.legend { display:flex; flex-wrap:wrap; gap:14px; color:var(--muted); font-size:13px; margin:0 0 10px; }
.legend i { display:inline-block; width:12px; height:12px; border-radius:2px; vertical-align:-1px; margin-right:5px; }
.wrap { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:10px; position:relative; }
canvas { display:block; width:100%; cursor:grab; touch-action:none; }
canvas.drag { cursor:grabbing; }
#tip { position:absolute; pointer-events:none; background:var(--panel); border:1px solid var(--line);
       border-radius:6px; padding:6px 8px; font-size:12px; white-space:nowrap; display:none;
       box-shadow:0 2px 8px rgba(0,0,0,.12); font-variant-numeric:tabular-nums; }
table { border-collapse:collapse; margin-top:16px; font-variant-numeric:tabular-nums; }
th, td { padding:4px 12px 4px 0; text-align:right; border-bottom:1px solid var(--line); }
th:first-child, td:first-child { text-align:left; }
th { color:var(--muted); font-weight:500; }
</style></head><body><main>
<h1>Possession bars</h1>
<p class="sub">Scroll to zoom, drag to pan, hover for labels. Bottom rows: where the chosen pair disagrees.</p>
<div class="bar">
  <button data-view="all">Whole match</button><button data-view="h1">1st half</button>
  <button data-view="h2">2nd half</button><button data-view="pub">Public 5 min</button>
  <span style="margin-left:12px;color:var(--muted)">compare</span>
  <select id="a"></select><span style="color:var(--muted)">with</span><select id="b"></select>
</div>
<div class="legend">
  <span><i style="background:var(--H)"></i>home (H)</span><span><i style="background:var(--A)"></i>away (A)</span>
  <span><i style="background:var(--D)"></i>dead (D)</span>
  <span><i style="background:var(--team)"></i>different team, both live</span>
  <span><i style="background:var(--alive)"></i>one live, one dead</span>
</div>
<div class="wrap"><canvas id="c"></canvas><div id="tip"></div></div>
<table id="t"></table>
</main>
<script>
const DATA = /*DATA*/;
const M = DATA.meta, ROWS = DATA.rows;
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const canvas = document.getElementById('c'), ctx = canvas.getContext('2d'), tip = document.getElementById('tip');
const LABEL_W = 190, ROW_H = 22, GAP = 6, AXIS_H = 22;
let view = [M.first, M.last + 1];

// label of a row at frame f (binary search over run starts)
function at(row, f) {
  const r = row.runs; let lo = 0, hi = r.length - 1;
  while (lo < hi) { const mid = (lo + hi + 1) >> 1; if (r[mid][0] <= f) lo = mid; else hi = mid - 1; }
  return r[lo][1];
}
// difference runs between two rows: label 'T' (team) or 'L' (alive/dead) or '' (same)
function diffRuns(a, b) {
  const cuts = new Set(); a.runs.forEach(r => cuts.add(r[0])); b.runs.forEach(r => cuts.add(r[0]));
  const starts = [...cuts].sort((x, y) => x - y), out = [];
  for (const s of starts) {
    const x = at(a, s), y = at(b, s); let d = '';
    if (x !== 'X' && y !== 'X' && x !== y) d = (x === 'D' || y === 'D') ? 'L' : 'T';
    if (!out.length || out[out.length - 1][1] !== d) out.push([s, d]);
  }
  return {runs: out};
}
function clock(f) {
  const [k1, k2] = M.kickoffs;
  if (f >= M.h1_end && f < M.h2_start) return 'HT';
  const s = f >= M.h2_start ? 2700 + (f - k2) / M.fps : (f - k1) / M.fps;
  const sign = s < 0 ? '-' : '', a = Math.abs(s);
  return sign + Math.floor(a / 60) + ':' + String(Math.floor(a % 60)).padStart(2, '0');
}
const sols = ROWS.filter(r => !r.gt);
const selA = document.getElementById('a'), selB = document.getElementById('b');
sols.forEach((r, i) => { selA.add(new Option(r.name, i)); selB.add(new Option(r.name, i)); });
const refIdx = Math.max(0, sols.findIndex(r => r.name === M.ref));
selA.value = refIdx; selB.value = sols.length > 1 ? (refIdx === 0 ? 1 : 0) : 0;

function rowsToDraw() {
  const a = sols[+selA.value], b = sols[+selB.value], d = diffRuns(a, b);
  const team = {runs: d.runs.map(([s, x]) => [s, x === 'T' ? 'T' : ''])};
  const live = {runs: d.runs.map(([s, x]) => [s, x === 'L' ? 'L' : ''])};
  return [...ROWS.map(r => ({...r, kind: 'labels'})),
          {name: 'different team', runs: team.runs, kind: 'diff'},
          {name: 'live vs dead', runs: live.runs, kind: 'diff'}];
}
let drawn = [];
function draw() {
  drawn = rowsToDraw();
  const dpr = window.devicePixelRatio || 1, w = canvas.clientWidth;
  const h = AXIS_H + drawn.length * (ROW_H + GAP) + 8;
  canvas.style.height = h + 'px'; canvas.width = w * dpr; canvas.height = h * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, w, h);
  const colors = {H: css('--H'), A: css('--A'), D: css('--D'), T: css('--team'), L: css('--alive')};
  const x0 = LABEL_W, pw = w - LABEL_W - 4, [v0, v1] = view, sx = pw / (v1 - v0);
  ctx.font = '12px system-ui, sans-serif'; ctx.textBaseline = 'middle';
  // axis
  ctx.fillStyle = css('--muted'); ctx.strokeStyle = css('--line');
  const span = (v1 - v0) / M.fps, steps = [5,10,30,60,120,300,600,900];
  const step = (steps.find(s => span / s <= 10) || 900) * M.fps;
  for (let f = Math.ceil(v0 / step) * step; f < v1; f += step) {
    const x = x0 + (f - v0) * sx; ctx.fillText(clock(f), x + 2, 10);
    ctx.beginPath(); ctx.moveTo(x, 16); ctx.lineTo(x, h); ctx.stroke();
  }
  drawn.forEach((row, i) => {
    const y = AXIS_H + i * (ROW_H + GAP);
    ctx.fillStyle = row.kind === 'diff' ? css('--muted') : css('--fg');
    ctx.fillText(row.name.length > 26 ? row.name.slice(0, 25) + '…' : row.name, 4, y + ROW_H / 2);
    const runs = row.runs;
    for (let j = 0; j < runs.length; j++) {
      const s = runs[j][0], e = j + 1 < runs.length ? runs[j + 1][0] : M.n, lab = runs[j][1];
      if (e <= v0 || s >= v1 || !colors[lab]) continue;
      const a = x0 + (Math.max(s, v0) - v0) * sx, b = x0 + (Math.min(e, v1) - v0) * sx;
      ctx.fillStyle = colors[lab]; ctx.fillRect(a, y, Math.max(b - a, 0.6), ROW_H);
    }
  });
  table();
}
function table() {
  const a = sols[+selA.value], b = sols[+selB.value];
  let diffTeam = 0, live = 0, diffLive = 0, graded = 0;
  const cuts = new Set(); a.runs.forEach(r => cuts.add(r[0])); b.runs.forEach(r => cuts.add(r[0]));
  const starts = [...cuts].sort((x, y) => x - y);
  starts.forEach((s, i) => {
    const e = i + 1 < starts.length ? starts[i + 1] : M.n, len = e - s, x = at(a, s), y = at(b, s);
    if (x === 'X' || y === 'X') return; graded += len;
    if ((x === 'D') !== (y === 'D')) diffLive += len;
    if (x !== 'D' && y !== 'D') { live += len; if (x !== y) diffTeam += len; }
  });
  const pct = v => (100 * v).toFixed(1) + '%';
  document.getElementById('t').innerHTML =
    '<tr><th>solution</th><th>possessions</th><th>dead share</th></tr>' +
    sols.map(r => `<tr><td>${r.name}</td><td>${r.poss}</td><td>${pct(r.dead)}</td></tr>`).join('') +
    `<tr><td colspan="3" style="padding-top:14px;color:var(--muted)">${a.name} vs ${b.name}: ` +
    `different team on ${pct(live ? diffTeam / live : 0)} of frames both call live, ` +
    `live/dead differs on ${pct(graded ? diffLive / graded : 0)} of graded frames</td></tr>`;
}
function frameAt(clientX) {
  const r = canvas.getBoundingClientRect(), pw = r.width - LABEL_W - 4;
  return view[0] + (clientX - r.left - LABEL_W) / pw * (view[1] - view[0]);
}
function setView(a, b) {
  const minSpan = 5 * M.fps, lo = 0, hi = M.n;
  if (b - a < minSpan) { const m = (a + b) / 2; a = m - minSpan / 2; b = m + minSpan / 2; }
  if (a < lo) { b += lo - a; a = lo; } if (b > hi) { a -= b - hi; b = hi; }
  view = [Math.max(lo, a), Math.min(hi, b)]; draw();
}
canvas.addEventListener('wheel', e => {
  e.preventDefault(); const f = frameAt(e.clientX), k = Math.exp(e.deltaY * 0.0015);
  setView(f - (f - view[0]) * k, f + (view[1] - f) * k);
}, {passive: false});
let drag = null;
canvas.addEventListener('pointerdown', e => { drag = {x: e.clientX, v: [...view]}; canvas.classList.add('drag'); canvas.setPointerCapture(e.pointerId); });
canvas.addEventListener('pointerup', () => { drag = null; canvas.classList.remove('drag'); });
canvas.addEventListener('pointermove', e => {
  const r = canvas.getBoundingClientRect();
  if (drag) {
    const df = (e.clientX - drag.x) / (r.width - LABEL_W - 4) * (drag.v[1] - drag.v[0]);
    setView(drag.v[0] - df, drag.v[1] - df); return;
  }
  const f = Math.round(frameAt(e.clientX));
  if (e.clientX - r.left < LABEL_W || f < 0 || f >= M.n) { tip.style.display = 'none'; return; }
  const names = {H: 'home', A: 'away', D: 'dead', X: '—'};
  tip.innerHTML = `<b>frame ${f}</b> · ${clock(f)}<br>` +
    ROWS.map(row => `${row.name}: ${names[at(row, f)] || at(row, f)}`).join('<br>');
  tip.style.display = 'block';
  const wrap = canvas.parentElement.getBoundingClientRect();
  let left = e.clientX - wrap.left + 14; if (left + tip.offsetWidth > wrap.width) left -= tip.offsetWidth + 28;
  tip.style.left = left + 'px'; tip.style.top = (e.clientY - wrap.top + 14) + 'px';
});
canvas.addEventListener('pointerleave', () => { tip.style.display = 'none'; });
document.querySelectorAll('[data-view]').forEach(btn => btn.addEventListener('click', () => {
  const [k1, k2] = M.kickoffs, gt = ROWS.find(r => r.gt);
  const v = btn.dataset.view;
  if (v === 'all') setView(M.first, M.last + 1);
  if (v === 'h1') setView(k1, M.h1_end);
  if (v === 'h2') setView(M.h2_start, M.last + 1);
  if (v === 'pub' && gt) {
    const r = gt.runs, s = (r.find(x => x[1] !== 'X') || [k1])[0];
    let e = s; for (let i = 0; i < r.length; i++) if (r[i][1] !== 'X' && i + 1 < r.length) e = r[i + 1][0];
    setView(s, e);
  }
}));
selA.addEventListener('change', draw); selB.addEventListener('change', draw);
window.addEventListener('resize', draw);
draw();
</script></body></html>
"""

if __name__ == '__main__':
    main()
