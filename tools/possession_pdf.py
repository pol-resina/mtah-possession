"""A PDF comparing our solutions as possession bars, with every difference highlighted.

    python tools/possession_pdf.py            # -> evaluation/reports/possession_comparison.pdf

Each solution is a strip coloured by who has the ball (home blue, away red, dead grey). Under it,
a thin strip marks where it differs from the reference (the submitted solution): amber
where both say the ball is alive but name different teams, purple where one says alive and the
other dead. The public-window page compares every solution with the public ground truth instead.
Pages: summary table, whole match, public 5 minutes, then the match in 10-minute windows.
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
sys.path.insert(0, os.path.join(ROOT, "starter"))
import possession_metrics as pm  # noqa: E402
from load_smart_tagging import KICKOFF_VIDEO_FRAMES  # noqa: E402

RESULTS = os.path.join(ROOT, "evaluation", "results")
PUBLIC_GT = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
OUT = os.path.join(ROOT, "evaluation", "reports", "possession_comparison.pdf")
FPS = 25
REFERENCE = "solution"
# Our solutions, oldest first, with one line on what each is.
SOLUTIONS = (
    ("solution_pol_5e9d334", "Pol: rules (nearest player + dead ball)"),
    ("solution_adria", "Adrià: learned team model + Pol's dead ball"),
    ("solution", "SUBMITTED: Adrià's model + sharper dead ball"),
)
WINDOW_MIN = 10

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
COLORS = {"H": "#2f6fd6", "A": "#d6453d", "D": "#b7b9bb"}
TEAM_DIFF, LIVE_DIFF = "#e8a317", "#8a4fd1"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "text.color": INK,
                     "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": INK, "figure.facecolor": "white"})


def read_labels(path):
    frame = pd.read_csv(path, dtype={"label": "object"})
    frame.columns = [str(c).strip().lower() for c in frame.columns]
    return frame["label"].fillna("X").astype(str).str.strip().str.upper().values


def runs(mask):
    """(start, stop) of every run of True."""
    edges = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def clock(f, h2_start, k1, k2):
    """Match minutes: first half from its kickoff, second half from 45:00 at its kickoff."""
    return np.where(f >= h2_start, 45.0 + (f - k2) / FPS / 60.0, (f - k1) / FPS / 60.0)


def draw_window(ax, rows, ref, a, b, meta, compare_name="submitted"):
    """Bars for frames a..b: each row a label strip and, under it, its differences from ref."""
    y = 0.0
    ticks, names = [], []
    for name, lab in rows:
        seg = lab[a:b]
        for label, color in COLORS.items():
            ax.broken_barh([(s, e - s) for s, e in runs(seg == label)], (y, 0.62),
                           facecolors=color, linewidth=0)
        ticks.append(y + 0.31)
        names.append(name)
        if ref is not None and lab is not ref:
            other = ref[a:b]
            graded = (seg != "X") & (other != "X")
            live = graded & (seg != "D") & (other != "D")
            team = live & (seg != other)
            alive = graded & ((seg == "D") != (other == "D"))
            ax.broken_barh([(s, e - s) for s, e in runs(team)], (y + 0.68, 0.16),
                           facecolors=TEAM_DIFF, linewidth=0)
            ax.broken_barh([(s, e - s) for s, e in runs(alive)], (y + 0.68, 0.16),
                           facecolors=LIVE_DIFF, linewidth=0)
        y += 1.15
    ax.set_ylim(y - 0.2, -0.25)
    ax.set_yticks(ticks)
    ax.set_yticklabels(names)
    ax.tick_params(axis="y", length=0)
    n = b - a
    ax.set_xlim(0, n)
    span_min = n / FPS / 60.0
    step = next(s for s in (0.5, 1, 2, 5, 10, 15) if span_min / s <= 12)
    m0 = float(clock(np.array(a), *meta))
    m1 = float(clock(np.array(b), *meta))
    marks = np.arange(np.ceil(m0 / step) * step, m1 + 1e-9, step)
    pos = [(mm - m0) * 60 * FPS for mm in marks]
    ax.set_xticks(pos)
    ax.set_xticklabels(["{:d}:{:02d}".format(int(mm), int(round((mm % 1) * 60))) for mm in marks])
    ax.set_xlabel("match clock (minutes)")
    for x in pos:
        ax.axvline(x, color=GRID, linewidth=0.6, zorder=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)


def legend(fig, compare_name="the submitted solution"):
    handles = [Patch(color=COLORS["H"], label="home has the ball (H)"),
               Patch(color=COLORS["A"], label="away has the ball (A)"),
               Patch(color=COLORS["D"], label="dead ball (D)"),
               Patch(color=TEAM_DIFF, label="highlight: different team than " + compare_name),
               Patch(color=LIVE_DIFF, label="highlight: live vs dead differs from " + compare_name)]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=8,
               handlelength=1.2, columnspacing=1.4)


def header(fig, title, subtitle):
    fig.text(0.04, 0.955, title, fontsize=14, weight="bold")
    fig.text(0.04, 0.925, subtitle, fontsize=9, color=MUTED)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=OUT)
    args = parser.parse_args()

    labels = {}
    for name, _ in SOLUTIONS:
        path = os.path.join(RESULTS, name + ".csv")
        if os.path.isfile(path):
            labels[name] = read_labels(path)
        else:
            print("missing, skipped:", path)
    ref = labels[REFERENCE]
    gt = read_labels(PUBLIC_GT)
    graded = np.flatnonzero(ref != "X")
    gap = int(np.argmax(np.diff(graded)))
    h1 = (int(graded[0]), int(graded[gap]) + 1)
    h2 = (int(graded[gap + 1]), int(graded[-1]) + 1)
    k1, k2 = KICKOFF_VIDEO_FRAMES["1st Half"], KICKOFF_VIDEO_FRAMES["2nd Half"]
    meta = (h2[0], k1, k2)
    pub = np.flatnonzero(gt != "X")
    pub = (int(pub[0]), int(pub[-1]) + 1)

    # summary numbers
    gt_codes = pm.label_codes(gt)
    table = []
    for name, desc in SOLUTIONS:
        if name not in labels:
            continue
        lab = labels[name]
        score = pm.evaluate(gt_codes, pm.label_codes(lab), bootstrap_n=0)
        g = (lab != "X") & (ref != "X")
        live = g & (lab != "D") & (ref != "D")
        live_lab = lab[np.isin(lab, ("H", "A"))]
        poss = 0  # runs of one team, half by half
        for a, b in (h1, h2):
            seg = lab[a:b]
            change = np.flatnonzero(seg[1:] != seg[:-1]) + 1
            starts = np.concatenate([[0], change])
            poss += int(np.sum(np.isin(seg[starts], ("H", "A"))))
        table.append([
            name + (" *" if name == REFERENCE else ""),
            "{:.3f}".format(score["spell_f1_mean"]), "{:.3f}".format(score["macro_f1"]),
            str(poss), "{:.1%}".format(np.mean(lab[lab != "X"] == "D")),
            "{:.1%}".format(np.mean(live_lab == "H")) if len(live_lab) else "-",
            "-" if name == REFERENCE else "{:.1%}".format(np.mean(lab[live] != ref[live])),
            "-" if name == REFERENCE else "{:.1%}".format(np.mean((lab[g] == "D") != (ref[g] == "D"))),
        ])

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with PdfPages(args.out) as pdf:
        # 1. summary
        fig = plt.figure(figsize=(11.69, 8.27))
        header(fig, "La MasIA: possession by solution",
               "FC Barcelona (home) vs Athletic Club (away), whole match. * = the submitted solution, "
               "and the reference every highlight is measured against.")
        ax = fig.add_axes([0.04, 0.56, 0.92, 0.32])
        ax.axis("off")
        cols = ["solution", "public\nspellF1", "public\nmacroF1", "posses-\nsions",
                "dead\nshare", "home share\nof live", "different\nteam vs\nsubmitted",
                "live/dead\ndiffers vs\nsubmitted"]
        t = ax.table(cellText=table, colLabels=cols, loc="upper left", cellLoc="right",
                     colLoc="right", colWidths=[0.3, 0.09, 0.09, 0.09, 0.08, 0.1, 0.11, 0.11])
        t.auto_set_font_size(False)
        t.set_fontsize(9.5)
        t.scale(1, 2.4)
        for (r, c), cell in t.get_celld().items():
            cell.set_edgecolor(GRID)
            cell.set_linewidth(0.6)
            if c == 0:
                cell.set_text_props(ha="left")
                cell._loc = "left"
            if r == 0:
                cell.set_text_props(color=MUTED)
                cell.set_height(cell.get_height() * 1.6)
            if r > 0 and table[r - 1][0].endswith("*"):
                cell.set_text_props(weight="bold")
        fig.text(0.04, 0.47, "\n".join(
            ["What each solution is"] + ["  - {}: {}".format(n, d) for n, d in SOLUTIONS if n in labels]
            + ["", "How to read the next pages",
            "  - Each solution is one strip: blue = home has the ball, red = away, grey = dead ball, blank = not graded.",
            "  - The thin strip under each solution highlights where it differs from the submitted solution:",
            "    amber = both say the ball is alive but name different teams; purple = one says alive, the other dead.",
            "  - Page 3 is the only part with ground truth (the public 5 minutes); there the highlights are against the ground truth.",
            "  - Public scores cover 17 possessions only (spellF1 5-95% band of the submission: 0.48-0.84), so small gaps are noise.",
        ]), fontsize=10, va="top", linespacing=1.6)
        pdf.savefig(fig)
        plt.close(fig)

        rows = [(n, labels[n]) for n, _ in SOLUTIONS if n in labels]

        # 2. whole match, one panel per half
        fig, axes = plt.subplots(2, 1, figsize=(11.69, 8.27))
        fig.subplots_adjust(left=0.2, right=0.98, top=0.86, bottom=0.12, hspace=0.45)
        header(fig, "Whole match", "First half above, second half below. Highlights against the submitted solution.")
        for ax, (a, b), title in zip(axes, (h1, h2), ("1st half", "2nd half")):
            draw_window(ax, rows, ref, a, b, meta)
            ax.set_title(title, loc="left", fontsize=10, color=INK)
        legend(fig)
        pdf.savefig(fig)
        plt.close(fig)

        # 3. public window against the ground truth
        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        fig.subplots_adjust(left=0.2, right=0.98, top=0.86, bottom=0.18)
        header(fig, "Public 5 minutes, against the ground truth",
               "The only stretch with labels. Highlights here mark where each solution differs from the ground truth.")
        draw_window(ax, [("public ground truth", gt)] + rows, gt, pub[0], pub[1], meta)
        legend(fig, "the ground truth")
        pdf.savefig(fig)
        plt.close(fig)

        # 4+. 10-minute windows, two per page
        windows = []
        for a, b in (h1, h2):
            step = WINDOW_MIN * 60 * FPS
            for s in range(a, b, step):
                windows.append((s, min(b, s + step)))
        for i in range(0, len(windows), 2):
            pair = windows[i:i + 2]
            fig, axes = plt.subplots(2, 1, figsize=(11.69, 8.27))
            axes = np.atleast_1d(axes)
            fig.subplots_adjust(left=0.2, right=0.98, top=0.86, bottom=0.12, hspace=0.45)
            first = float(clock(np.array(pair[0][0]), *meta))
            last = float(clock(np.array(pair[-1][1]), *meta))
            header(fig, "Detail: {:.0f}' to {:.0f}'".format(np.floor(first), np.ceil(last)),
                   "10-minute windows. Highlights against the submitted solution.")
            for ax, (a, b) in zip(axes, pair):
                draw_window(ax, rows, ref, a, b, meta)
            for ax in axes[len(pair):]:
                ax.axis("off")
            legend(fig)
            pdf.savefig(fig)
            plt.close(fig)

    print("wrote", os.path.relpath(args.out, os.getcwd()))


if __name__ == '__main__':
    main()
