"""Build the submission from tracking data alone: dead ball first, then possession.

    python solutions/solution_pol_5e9d334/run.py                 # evaluation/results/solution_pol_5e9d334.csv
    python solutions/solution_pol_5e9d334/run.py --no-broadcast  # tactical camera's tracking only
    python evaluation/run_evaluation.py       # scores it

Inputs: both cameras' tracking (ATD). No video pixels, no Smart Tagging, no ground truth. The same
inputs always give the same file, byte for byte, as RULES.md requires. README.md explains
the method.
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))  # solutions/<this folder>/ -> repository
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from baseline import write_submission  # noqa: E402
from load_smart_tagging import KICKOFF_VIDEO_FRAMES  # noqa: E402
import broadcast  # noqa: E402
import deadball  # noqa: E402
import possession  # noqa: E402
import tracking  # noqa: E402
from segments import label_runs  # noqa: E402

# Params
RESULTS_DIR = os.path.join(ROOT, "evaluation", "results")
# Kickoff frame of each half: the first graded frame, and 113,742 (FAQ.md) for the second half.
KICKOFFS = {1: KICKOFF_VIDEO_FRAMES["1st Half"], 2: KICKOFF_VIDEO_FRAMES["2nd Half"]}


def build(match, second=None):
    """(labels, dead mask, stoppages) for a loaded match.

    second: the broadcast camera's contact evidence (broadcast.contact_evidence), or None to use
    the tactical camera only.
    """
    dead, stoppages = deadball.detect(match, KICKOFFS)
    return possession.label(match, dead, stoppages, second), dead, stoppages


def summary(labels, match):
    """Label counts and possession durations over the graded frames."""
    graded = labels[labels != "X"]
    counts = {label: int(np.sum(graded == label)) for label in "HAD"}
    spells = [b - a for _, first, last in match.halves
              for a, b, label in label_runs(labels, first, last + 1) if label in "HA"]
    return ("graded frames  H {H}  A {A}  D {D}".format(**counts)
            + "   (D {:.1%} of graded)\n".format(counts["D"] / float(len(graded)))
            + "possessions    {}   mean {:.1f} s   median {:.1f} s".format(
                len(spells), np.mean(spells) / match.fps, np.median(spells) / match.fps))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", default=os.path.basename(HERE), help="file name on the board")
    parser.add_argument("--out", default=RESULTS_DIR, help="where to write")
    parser.add_argument("--no-broadcast", action="store_true",
                        help="use the tactical camera's tracking only")
    args = parser.parse_args()

    match = tracking.load()
    second = None if args.no_broadcast else broadcast.contact_evidence(match)
    labels, _, _ = build(match, second)
    path = write_submission(os.path.join(args.out, "{}.csv".format(args.name)), labels)
    print(summary(labels, match))
    print("-> {}".format(os.path.relpath(path, ROOT)))
