"""solution_pol_loop: Adria's learned team model on a sharper dead ball (three additions to Pol's,
all in deadball.py: broadcast cut-aways, goals, false restarts).

    python solutions/solution_pol_loop/run.py   # writes evaluation/results/solution_pol_loop.csv
    python evaluation/run_evaluation.py         # scores it

Inputs: both cameras' tracking. From the broadcast camera only two flags per frame are used: is it
on the game shot, and does it have the ball. No video, no Smart Tagging, no ground truth.
Deterministic: same inputs, same file. README.md explains it.
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
from baseline import write_submission  # noqa: E402
from load_smart_tagging import KICKOFF_VIDEO_FRAMES  # noqa: E402
import games  # noqa: E402
import broadcast  # noqa: E402
import deadball  # noqa: E402
import features as FT  # noqa: E402
import tracking  # noqa: E402
from segments import label_runs, runs, two_state  # noqa: E402

# Params
TEAM_MODEL = os.path.join(HERE, "team_model.txt")  # LightGBM, P(home | ball alive), 5 Hz rows
SWITCH_COST = 1.0       # nats per change of team (the value chosen in every DFL bench fold)
ROW_FRAMES = 5          # 25 fps frames per 5 Hz row: a frame carries 1/5 of a row's evidence
RESULTS_DIR = os.path.join(ROOT, "evaluation", "results")
# Kickoff frame of each half: the first graded frame, and 113,742 (FAQ.md) for the second half.
KICKOFFS = {1: KICKOFF_VIDEO_FRAMES["1st Half"], 2: KICKOFF_VIDEO_FRAMES["2nd Half"]}


def as_obs(match):
    """The loaded match in the format features.py reads (positions normalised to the pitch)."""
    L, W = match.length, match.width
    n, s = match.px.shape
    players = np.full((n, s, 3), np.nan, dtype=np.float32)
    players[:, :, 0] = match.px / L
    players[:, :, 1] = match.py / W
    ball = np.column_stack([match.bx / L, match.by / W]).astype(np.float32)
    meta_path, = games.require(["atd_meta"], match.game)
    team = np.asarray(tracking.team_of_each_track(meta_path))
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
    """(dead mask, stoppages): Pol's dead ball, with the broadcast camera as a second witness: its
    cut-aways are stoppage evidence, and a restart needs a ball in at least one camera."""
    other = broadcast.on_tactical_timeline(match)
    blind = ~match.ball_seen & np.isnan(other.bx)
    return deadball.detect(match, KICKOFFS, other.cut, blind)


def build(match):
    """(labels, dead mask, stoppages) for a loaded match."""
    dead, stoppages = dead_ball(match)
    p = np.clip(team_probability(match), 1e-6, 1 - 1e-6)
    cost_home, cost_away = -np.log(p) / ROW_FRAMES, -np.log(1 - p) / ROW_FRAMES
    out = np.full(match.n, "X", dtype="<U1")
    for _, first, last in match.halves:
        out[first:last + 1] = "D"
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
