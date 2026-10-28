"""Check a submission without tuning against the five public minutes.

    python solution/diagnostics.py            # every check, with the current parameters
    python solution/diagnostics.py --sweep    # the same, with each parameter halved and doubled

The public ground truth has 17 possessions and 4 stoppages -- far too few to choose anything with.
It is reported with its bootstrap band so the noise is visible, but the checks that decide anything
cover the whole match and need no ground truth:

  D share        dead ball as a share of graded frames (real matches: 40-50%)
  possessions    how many, and how long (the public window averages 14 s)
  set pieces     share of Smart Tagging's 59 tagged restarts with one of our stoppages ending in it
  restart team   share of tagged restarts that name a team where our first live frames agree
  phase          share of our live frames inside Smart Tagging's in-possession phases that name the
                 tagged team (Smart Tagging is not an input to the pipeline, so it can check it)

To tell whether a change helped on the public window, use paired_spell_difference(old, new): it
scores both versions on the same bootstrap resamples, which pins the difference down far better
than two separate intervals.
"""

import argparse
import contextlib
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
sys.path.insert(0, os.path.join(ROOT, "starter"))
import load_smart_tagging  # noqa: E402
import possession_metrics as pm  # noqa: E402
from solution import broadcast, deadball, possession, run, tracking  # noqa: E402
from solution.segments import label_runs  # noqa: E402

# Params
PUBLIC_GT = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
BOOTSTRAP_N = 1000
SIDE = {"Barcelona": "H", "Atletic Club": "A"}
PHASE_CODES = ("BUILD UP", "PROGRESSION", "CREATING CHANCES", "ATTACKING TRANSITION")
TAG_SLACK = 50             # frames either side of a tagged restart that still count
RESTART_TEAM_FRAMES = 25   # first live frames after a restart compared with its tagged team
# Parameters swept by --sweep, one at a time: (module, name, values).
SWEEP = (
    (deadball, "ACTIVITY_MID", (2.2, 3.0)),
    (deadball, "ACTIVITY_SCALE", (0.25, 1.0)),
    (deadball, "MISSING_WEIGHT", (0.75, 3.0)),
    (deadball, "SWITCH_COST", (20.0, 80.0)),
    (deadball, "UNCONFIRMED_MIN", (75, 300)),
    (deadball, "SPOT_MARGIN", (1.5, 6.0)),
    (deadball, "RESTART_SPEED", (4.0, 7.0)),
    (deadball, "RESTART_LAG", (0, 20)),
    (possession, "CONTROL_RADIUS", (1.0, 2.0)),
    (possession, "SWITCH_COST", (4.0, 16.0)),
    (possession, "CONTROL_LAG", (10, 30)),
    (possession, "BROADCAST_WEIGHT", (0.36, 1.0)),
)
HEAD = ("{:<26} {:>7} {:>7} {:>4} | {:>5} {:>4} {:>5} {:>5} {:>5} {:>5} {:>5} {:>5} {:>5}\n"
        "{:<26} {:>20} | {:>53}").format(
    "", "spellF1", "macroF1", "poss", "Dshr", "stop", "poss", "mean", "med", "setpc", "rteam",
    "<2s", "phase", "", "public 5 minutes", "full match, no ground truth")
ROW = ("{name:<26} {spell:7.4f} {macro:7.4f} {n_public:4d} | {d_share:5.3f} {n_stoppages:4d} "
       "{n_possessions:5d} {mean_s:5.1f} {median_s:5.1f} {set_pieces:5.2f} {restart_team:5.2f} "
       "{short_share:5.2f} {phase:5.3f}")


@contextlib.contextmanager
def override(module, name, value):
    """Temporarily set a parameter. Parameters are read at call time, so this is all a sweep needs."""
    old = getattr(module, name)
    setattr(module, name, value)
    try:
        yield
    finally:
        setattr(module, name, old)


def public_scores(labels, bootstrap_n=0):
    """possession_metrics.evaluate on the public window, exactly as the scorer runs it."""
    gt = pm.label_codes(pm.read_label_csv(PUBLIC_GT))
    return pm.evaluate(gt, pm.label_codes(labels), bootstrap_n=bootstrap_n)


def paired_spell_difference(labels_a, labels_b, n_draws=BOOTSTRAP_N, seed=0):
    """Public spellF1 of b minus a: (difference, 5% bound, 95% bound, P(b better)).

    Both are scored on the same resamples of the ground-truth possessions, the way
    possession_metrics.bootstrap_spell_f1 resamples one submission.
    """
    gt = pm.label_codes(pm.read_label_csv(PUBLIC_GT))
    valid = pm.valid_runs(gt)
    gt_spells = pm.possession_spells(gt, valid)
    n_gt = len(gt_spells)
    flags, n_pred = [], []
    for labels in (labels_a, labels_b):
        pred = pm.possession_spells(pm.fill_abstentions(pm.label_codes(labels), valid), valid)
        overlaps = pm.spell_overlaps(gt_spells, pred)
        flags.append(np.vstack([pm.matched_gt_flags(overlaps, n_gt, t) for t in pm.SPELL_IOUS]))
        n_pred.append(len(pred))

    def score(which, picks):
        f1s = []
        for row in flags[which]:
            tp = int(row[picks].sum())
            fp, fn = max(0, n_pred[which] - tp), n_gt - tp
            f1s.append(2.0 * tp / (2 * tp + fp + fn) if tp else 0.0)
        return np.mean(f1s)

    rng = np.random.RandomState(seed)
    draws = []
    for _ in range(n_draws):
        picks = rng.randint(0, n_gt, n_gt)
        draws.append(score(1, picks) - score(0, picks))
    low, high = np.percentile(draws, (5, 95))
    everyone = np.arange(n_gt)
    return score(1, everyone) - score(0, everyone), low, high, float(np.mean(np.array(draws) > 0))


def segment_table(labels):
    """One line per public ground-truth segment: our segments over it, and the best IoU."""
    gt = pm.read_label_csv(PUBLIC_GT)["label"].values
    graded = np.flatnonzero(gt != "X")
    first, stop = int(graded[0]), int(graded[-1]) + 1
    ours = label_runs(labels, first, stop)
    lines = []
    for a, b, label in label_runs(gt, first, stop):
        over = [(c, d, x) for c, d, x in ours if d > a and c < b]
        iou = max([(min(b, d) - max(a, c)) / float(max(b, d) - min(a, c))
                   for c, d, x in over if x == label] or [0.0])
        lines.append("  {} {:>6}-{:<6} {:5.1f}s  IoU {:.2f}  <- {}".format(
            label, a, b, (b - a) / 25.0, iou,
            ", ".join("{} {}-{}".format(x, c, d) for c, d, x in over)))
    return "\n".join(lines)


def full_match_checks(match, labels, stoppages, events):
    """The whole-match checks described in the module docstring."""
    spells = [b - a for _, first, last in match.halves
              for a, b, label in label_runs(labels, first, last + 1) if label in "HA"]
    durations = np.array(spells) / float(match.fps)

    set_pieces = [e for e in events if (e["code"] or "").strip().upper() == "SET PIECES"]
    ends = np.array([s.stop for s in stoppages])
    found, team_n, team_ok = 0, 0, 0
    for event in set_pieces:
        inside = ends[(ends >= event["start_frame"] - TAG_SLACK)
                      & (ends <= event["end_frame"] + TAG_SLACK)]
        if not len(inside):
            continue
        found += 1
        side = SIDE.get(event["tags"].get("Team"))
        after = labels[inside[0]:inside[0] + RESTART_TEAM_FRAMES]
        after = after[np.isin(after, ("H", "A"))]
        if side is not None and len(after):
            team_n += 1
            team_ok += int(np.sum(after == side) * 2 > len(after))

    agree, total = 0, 0
    for event in events:
        side = SIDE.get(event["tags"].get("Team"))
        if side is None or (event["code"] or "").strip().upper() not in PHASE_CODES:
            continue
        window = labels[event["start_frame"]:event["end_frame"]]
        window = window[np.isin(window, ("H", "A"))]
        agree += int(np.sum(window == side))
        total += len(window)

    return {
        "d_share": float(np.mean(labels[labels != "X"] == "D")),
        "n_stoppages": len(stoppages),
        "n_possessions": len(spells),
        "mean_s": float(durations.mean()),
        "median_s": float(np.median(durations)),
        "short_share": float(np.mean(durations < 2.0)),
        "set_pieces": found / float(len(set_pieces)),
        "restart_team": team_ok / float(team_n) if team_n else float("nan"),
        "phase": agree / float(total) if total else float("nan"),
    }


def check(match, events, second):
    """Build with the current parameters: (labels, public scores, whole-match checks)."""
    labels, _, stoppages = run.build(match, second)
    return labels, public_scores(labels), full_match_checks(match, labels, stoppages, events)


def row(name, public, full):
    return ROW.format(name=name, spell=public["spell_f1_mean"], macro=public["macro_f1"],
                      n_public=public["n_pred_spells"], **full)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sweep", action="store_true", help="halve/double each parameter")
    args = parser.parse_args()

    match = tracking.load()
    events = load_smart_tagging.load()
    second = broadcast.contact_evidence(match)
    labels, public, full = check(match, events, second)

    boot = public_scores(labels, BOOTSTRAP_N)
    print("public window ({} possessions, mean {:.1f} s)".format(
        boot["n_gt_spells"], boot["gt_mean_spell_duration_s"]))
    print("  spellF1 {:.4f}  (bootstrap 5-95%: {:.4f} .. {:.4f})   macroF1 {:.4f}".format(
        boot["spell_f1_mean"], boot["spell_bootstrap_low"], boot["spell_bootstrap_high"],
        boot["macro_f1"]))
    print("  per IoU: " + "  ".join("{}: {:.3f}".format(k, v["f1"])
                                    for k, v in sorted(boot["spell_f1"].items())))
    print("\nground-truth segments and ours over them\n" + segment_table(labels))
    print("\nfull match")
    for key, value in full.items():
        print("  {:<14} {}".format(key, round(value, 4)))

    if args.sweep:
        print("\n" + HEAD)
        print(row("current", public, full))
        print(row("broadcast off", *check(match, events, None)[1:]))
        for module, name, values in SWEEP:
            for value in values:
                with override(module, name, value):
                    print(row("{}={}".format(name, value), *check(match, events, second)[1:]))
