"""Entry point: turns the match tracking into the submission CSV (one H / A / D / X per video frame).

    python solutions/solution_pol_loop/run.py   # writes evaluation/results/solution_pol_loop.csv
    python evaluation/run_evaluation.py         # scores it

WHAT IT IS FOR
    Runs the whole pipeline; the other modules in this folder are the stages it calls. Inputs: both
    cameras' automatic tracking data (ATD); from the broadcast camera only two flags per frame
    (on the game shot, has the ball). No video, no Smart Tagging, no ground truth. Deterministic.
    README.md explains the method.

THE IMPORTANT PARTS
    - Two independent decisions: dead_ball() says WHEN the ball is alive, the team model says WHO
      has it. build() writes D on dead frames and H / A on live ones.
    - The team model runs on 5 Hz rows (every 5th frame of the 25 fps tracking); each frame carries
      1/5 of a row's evidence (ROW_FRAMES).
    - Team labels are smoothed per live stretch with a two-state Viterbi (segments.two_state); a
      change of team costs SWITCH_COST, so possession can change freely across a stoppage.
    - Frames outside the two halves given by KICKOFFS stay X.
"""

import argparse
import os
import sys
from types import SimpleNamespace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))  # solutions/<this folder>/ -> repository
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))

from baseline import write_submission                # noqa: E402 (ignore the "Module level import not at top of file" error)
from load_smart_tagging import KICKOFF_VIDEO_FRAMES  # noqa: E402
import games                                         # noqa: E402
import broadcast                                     # noqa: E402
import deadball                                      # noqa: E402
import features as FT                                # noqa: E402
import tracking                                      # noqa: E402
from segments import label_runs, runs, two_state     # noqa: E402

# Params
TEAM_MODEL = os.path.join(HERE, "team_model.txt")  # LightGBM, P(home | ball alive), 5 Hz rows
SWITCH_COST = 1.0       # to stop label flickering in noisy frames
ROW_FRAMES = 5          # 25 fps frames, 5 samples per row (5Hz): a frame carries 1/5 of a row's evidence
RESULTS_DIR = os.path.join(ROOT, "evaluation", "results")
# mark first/second halves explicitly; frames before the first, between the two and after the second are labelled "X"
KICKOFFS = {1: KICKOFF_VIDEO_FRAMES["1st Half"], 2: KICKOFF_VIDEO_FRAMES["2nd Half"]}


def as_obs(match):
    """The loaded match in the format features.py reads (positions normalised to the pitch)."""
    L, W = match.length, match.width
    n, s = match.px.shape # only need x,y coordinates for features.py
    players = np.full((n, s, 3), np.nan, dtype=np.float32)
    # divide per pitch size (L = length / W = width) to normalise
    players[:, :, 0] = match.px / L
    players[:, :, 1] = match.py / W
    ball = np.column_stack([match.bx / L, match.by / W]).astype(np.float32)
    meta_path, = games.require(["atd_meta"], match.game)
    team =np.asarray(tracking.team_of_each_track(meta_path)) # non-corrected tracking metadata (0 = home / 1 = away)
    return SimpleNamespace(track={"players": players, "ball": ball, "team": team},
                           pitch=(float(L), float(W)), halves=match.halves,
                           attacks_right=match.attacks_right, kickoffs=KICKOFFS,
                           fps=match.fps, name=match.game)


def team_probability(match):
    """P(home | ball alive) on every frame (5 Hz model, nearest row)."""
    import lightgbm as lgb
    obs = as_obs(match)
    _, X = FT.frame_features(obs)
    rows = FT.sample_rows(obs)
    with open(TEAM_MODEL) as f:  # text mode: a Windows checkout's CRLF would break LightGBM's parser
        booster = lgb.Booster(model_str=f.read())
    p = booster.predict(X[rows])
    return FT.expand(p, rows, match.n, match.halves)


def dead_ball(match):
    """(dead mask, stoppages): the dead-ball detector, with the broadcast camera as a second witness: its
    cut-aways are stoppage evidence, and a restart needs a ball in at least one camera."""
    other = broadcast.on_tactical_timeline(match) # broadcast tracking on the tactical frame index
    blind = ~match.ball_seen & np.isnan(other.bx) # when ball is missing in both cameras
    return deadball.detect(match, KICKOFFS, other.cut, blind)


def build(match):
    """(labels, dead mask, stoppages) for a loaded match.

    Labels, one per video frame: H = home has the ball, A = away has the ball, D = ball dead,
    X = not graded (outside the two halves).
    """
    dead, stoppages = dead_ball(match)
    p = np.clip(team_probability(match), 1e-6, 1 - 1e-6)
    cost_home, cost_away = -np.log(p) / ROW_FRAMES, -np.log(1 - p) / ROW_FRAMES
    out = np.full(match.n, "X", dtype="<U1")
    for _, first, last in match.halves:
        out[first:last + 1] = "D"
        # when the ball is dead, a home/away noisy record has no effect, the switch penalty doesn't affect it
        for a, b in runs(~dead, first, last + 1):
            state = two_state(cost_home[a:b], cost_away[a:b], SWITCH_COST, SWITCH_COST)
            out[a:b] = np.where(state == 0, "H", "A")
    return out, dead, stoppages


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
    parser.add_argument("--name", default="solution_pol_loop", help="file name on the board")
    parser.add_argument("--out", default=RESULTS_DIR, help="where to write")
    args = parser.parse_args()

    match = tracking.load()
    labels, _, _ = build(match)
    path = write_submission(os.path.join(args.out, "{}.csv".format(args.name)), labels)
    print(summary(labels, match))
    print("-> {}".format(os.path.relpath(path, ROOT)))
