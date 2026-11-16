"""Aggregate EDA numbers for the slides: prints stats, writes small .dat/.tex files to slides/data.

    python slides/make_data.py
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
SOLUTION = "solution"  # the submitted solution's folder
sys.path.insert(0, os.path.join(ROOT, SOLUTION))
import tracking, deadball, possession  # noqa
from segments import runs, label_runs  # noqa

m = tracking.load()
gt = pd.read_csv(os.path.join(ROOT, "evaluation/gt/BAR-ATH_possession_gt_public.csv"))["label"].values
sol = pd.read_csv(os.path.join(ROOT, "evaluation/results", SOLUTION + ".csv"))["label"].values
b50 = pd.read_csv(os.path.join(ROOT, "evaluation/results/baseline_smooth50.csv"))["label"].values
braw = pd.read_csv(os.path.join(ROOT, "evaluation/results/baseline_raw.csv"))["label"].values

graded = m.half > 0
pub = gt != "X"
live = pub & np.isin(gt, ["H", "A"])
dead = pub & (gt == "D")

print("ball seen all %.3f graded %.3f" % (m.ball_seen.mean(), m.ball_seen[graded].mean()))
print("ball missing public live %.3f dead %.3f" % ((~m.ball_seen[live]).mean(), (~m.ball_seen[dead]).mean()))

# missing-gap lengths on graded frames
gaps = []
for _, a0, b0 in m.halves:
    gaps += [b - a for a, b in runs(~m.ball_seen, a0, b0 + 1)]
gaps = np.array(gaps)
bins = [0, 5, 25, 100, 250, 750, 10**9]
names = ["<0.2s", "0.2-1s", "1-4s", "4-10s", "10-30s", ">30s"]
tot = gaps.sum()
with open(os.path.join(OUT, "gaps.dat"), "w") as f:
    f.write("bin n share\n")
    for i, nm in enumerate(names):
        sel = (gaps >= bins[i]) & (gaps < bins[i + 1])
        f.write("%s %d %.4f\n" % (nm, sel.sum(), gaps[sel].sum() / tot))
        print("gap", nm, sel.sum(), "%.3f of missing frames" % (gaps[sel].sum() / tot))
print("n gaps", len(gaps), "median frames", np.median(gaps))

# tracked players per graded frame
nplay = (~np.isnan(m.px)).sum(axis=1)
print("players mean graded %.2f" % nplay[graded].mean())
h = np.bincount(nplay[graded], minlength=30)
with open(os.path.join(OUT, "players.dat"), "w") as f:
    f.write("n share\n")
    for k in range(10, 28):
        f.write("%d %.5f\n" % (k, h[k] / graded.sum()))

# activity (top-5 speed, 1 s mean): full-match histogram + public live/dead
act = deadball.activity(m)
edges = np.arange(0, 8.01, 0.2)
hg, _ = np.histogram(act[graded & ~np.isnan(act)], edges)
hl, _ = np.histogram(act[live & ~np.isnan(act)], edges)
hd, _ = np.histogram(act[dead & ~np.isnan(act)], edges)
with open(os.path.join(OUT, "activity.dat"), "w") as f:
    f.write("x all live dead\n")
    for i in range(len(hg)):
        f.write("%.2f %.5f %.5f %.5f\n" % (edges[i] + 0.1, hg[i] / hg.sum(), hl[i] / hl.sum(), hd[i] / hd.sum()))
print("activity median live %.2f dead %.2f" % (np.nanmedian(act[live]), np.nanmedian(act[dead])))

# contact evidence on public live frames
dh, da = possession.team_distances(m)
ev = possession.contact(dh, da)
truth = np.where(gt == "H", 0, np.where(gt == "A", 1, -1))
nearest = np.where(np.isfinite(dh) | np.isfinite(da), np.where(dh <= da, 0, 1), -1)
ls = live & m.ball_seen
print("public live frames %d, ball seen %d" % (live.sum(), ls.sum()))
sel = live & (nearest >= 0)
print("nearest-player right %.3f on %d frames" % ((nearest[sel] == truth[sel]).mean(), sel.sum()))
sel = live & (ev >= 0)
print("clear-contact right %.3f on %d frames (%.3f of live)" % ((ev[sel] == truth[sel]).mean(), sel.sum(), sel.sum() / live.sum()))
# contact share of graded frames full match
print("clear contact share of graded %.3f" % (ev[graded] >= 0).mean())

# distance of nearest player to ball, public live frames, ball seen
dmin = np.minimum(dh, da)
e2 = np.arange(0, 10.01, 0.5)
hdist, _ = np.histogram(np.clip(dmin[ls], 0, 9.99), e2)
with open(os.path.join(OUT, "dist.dat"), "w") as f:
    f.write("x share\n")
    for i in range(len(hdist)):
        f.write("%.2f %.5f\n" % (e2[i] + 0.25, hdist[i] / hdist.sum()))

# GT possessions in public window
first, last = 11185, 18685
spells = [(a, b, l) for a, b, l in label_runs(gt, first, last)]
print("GT segments", [(round((a - first) / 25, 1), round((b - a) / 25, 1), l) for a, b, l in spells])
durs = [(b - a) / 25 for a, b, l in spells if l in "HA"]
print("GT poss durations", sorted(round(d, 1) for d in durs), "median", np.median(durs))


def strip(labels, name):
    with open(os.path.join(OUT, "strip_%s.tex" % name), "w") as f:
        for a, b, l in label_runs(labels, first, last):
            f.write("\\seg{%.3f}{%.3f}{%s}\n" % ((a - first) / 25.0, (b - first) / 25.0, l))


for lab, nm in ((gt, "gt"), (sol, "sol"), (b50, "b50"), (braw, "raw")):
    strip(lab, nm)
    s = label_runs(lab, first, last)
    print(nm, "poss in window", sum(1 for *_, l in s if l in "HA"))

# full-match possessions of solution & baselines
for lab, nm in ((sol, "sol"), (b50, "b50"), (braw, "raw")):
    sp = [b - a for _, a0, b0 in m.halves for a, b, l in label_runs(lab, a0, b0 + 1) if l in "HA"]
    g = lab[graded]
    print(nm, "full poss", len(sp), "mean %.1fs" % (np.mean(sp) / 25), "D share %.3f" % (g == "D").mean())

# Our possession durations, full match
sp = np.array([b - a for _, a0, b0 in m.halves for a, b, l in label_runs(sol, a0, b0 + 1) if l in "HA"]) / 25
e3 = [0, 2, 5, 10, 20, 40, 80, 1e9]
hh, _ = np.histogram(sp, e3)
print("our durations hist", hh)
