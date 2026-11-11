"""Load the tactical tracking (ATD) as numpy arrays in metres, one row per video frame.

Row i is video frame i, so it is also row i of the submission (the raw file counts from 1). The
file is parsed in one numpy call, so nothing is cached and nothing derived from the tracking is ever
written to disk -- RULES.md forbids publishing it.

One correction is made on the way in: the tracker swaps the two goalkeepers between their tracks
(Athletic's keeper track sometimes follows Barcelona's keeper), so a goalkeeper track is given the
team that defends the half it is standing in.
"""

import os
import sys
import xml.etree.ElementTree as ET
from types import SimpleNamespace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))  # solutions/<this folder>/ -> repository
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from games import GAME, GAMES, half_video_blocks, require, source_frames  # noqa: E402
from baseline import team_of_each_track                                   # noqa: E402

# Params
MAX_PLAYER_SPEED = 12.0       # (m/s); faster than this between two frames is a track jumping people
GK_MIN_SHARE = 0.25           # a goalkeeper track is tracked for at least this share of a half...
GK_BOX_SHARE = 0.6            # ...and spends at least this share of that time in one penalty area
PENALTY_AREA = (16.5, 20.16)  # depth and half-width (m)


def parse_raw(raw_path, n_slots):
    """(frame ids, players (frames, slots, 3), ball (frames, 3)) from a raw ATD text file.

    Every line is "frame:x,y,s;x,y,s;...:x,y,s", so reading ':' and ';' as ',' turns the whole file
    into one flat array of floats ("NaN" where something is not tracked).
    """
    with open(raw_path) as handle:
        text = handle.read()
    n_lines = text.count("\n") + (0 if text.endswith("\n") else 1)  # tolerate a missing final newline
    flat = np.fromstring(text.replace(":", ",").replace(";", ",").replace("\n", ","),
                         dtype=np.float64, sep=",")
    width = 1 + 3 * n_slots + 3  # frame id + (x, y, speed) per player slot + (x, y, speed) of the ball
    if flat.size != n_lines * width:
        raise ValueError("{}: expected {} fields per line".format(raw_path, width))
    table = flat.reshape(n_lines, width)
    return table[:, 0].astype(np.int64), table[:, 1:-3].reshape(n_lines, n_slots, 3), table[:, -3:]


def home_attacks_right(meta_path):
    """{half: True if the home side attacks towards x = 105 m}, from the ATD metadata."""
    home = next(ET.parse(meta_path).getroot().iter("Team"))
    for parameter in home.iter("ProviderParameter"):
        value = parameter.findtext("Value")
        if (parameter.findtext("Name") == "attack_direction_first_half"
                and value in ("left_to_right", "right_to_left")):
            first = value == "left_to_right"
            return {1: first, 2: not first}
    raise ValueError("{}: no attack_direction_first_half for the home team".format(meta_path))


def team_per_frame(px, py, team, halves, attacks_right, length, width):
    """(frames, slots) team of each track on each frame: its metadata team, except that a
    goalkeeper track gets the team defending the half it is in."""
    out = np.repeat(team[None, :], len(px), axis=0)
    for half, first, last in halves:
        x, y = px[first:last + 1], py[first:last + 1]
        seen = ~np.isnan(x)
        with np.errstate(invalid="ignore"):
            # in either penalty area (the one at x = 0 or the one at x = length).
            in_box = (np.abs(y - width / 2.0) <= PENALTY_AREA[1]) & (
                (x <= PENALTY_AREA[0]) | (x >= length - PENALTY_AREA[0]))
        tracked = seen.sum(axis=0)
        # keeper tracker: present for a large enough share of the half on the same area
        keeper = ((tracked >= GK_MIN_SHARE * len(x))
                  & ((in_box & seen).sum(axis=0) >= GK_BOX_SHARE * np.maximum(tracked, 1)))
        # the home side defends x = 0 when it attacks right, so the defender of that goal is 0 in that case, else 1.
        left_defender = 0 if attacks_right[half] else 1
        with np.errstate(invalid="ignore"):
            on_left = x[:, keeper] < length / 2.0
        block = out[first:last + 1]
        block[:, keeper] = np.where(seen[:, keeper],
                                    np.where(on_left, left_defender, 1 - left_defender),
                                    team[keeper][None, :])
    return out


def load(game=GAME, data_dir=None):
    """The tactical tracking of a game. Every per-frame array is indexed by video frame."""
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir)
    length, width = GAMES[game]["pitch"]
    fps = GAMES[game]["fps"]
    n = source_frames("tactical", game)

    team = team_of_each_track(meta_path)
    frame_ids, players, ball = parse_raw(raw_path, len(team))
    # raw ids must be exactly 1..n -> ensures [row i = video frame i]
    if len(frame_ids) != n or frame_ids[0] != 1 or np.any(np.diff(frame_ids) != 1):
        raise ValueError("{}: expected frame ids 1..{}".format(raw_path, n))

    px = (players[:, :, 0] * length).astype(np.float32)
    py = (players[:, :, 1] * width).astype(np.float32)
    bx, by = ball[:, 0] * length, ball[:, 1] * width
    # a NaN position also blanks the speed of its two neighbouring frames
    speed = np.hypot(np.gradient(px, axis=0) * fps, np.gradient(py, axis=0) * fps)
    speed[speed > MAX_PLAYER_SPEED] = np.nan

    halves = half_video_blocks(game)     # [(half, first frame, last frame)] as graded
    half = np.zeros(n, dtype=np.int8)    # 1 or 2 on graded frames, 0 elsewhere
    for number, first, last in halves:
        half[first:last + 1] = number
    attacks_right = home_attacks_right(meta_path)

    return SimpleNamespace(
        game=game, n=n, fps=fps, length=length, width=width,
        halves=halves, half=half, attacks_right=attacks_right,
        px=px, py=py, speed=speed,
        team_at=team_per_frame(px, py, team, halves, attacks_right, length, width),
        bx=bx, by=by, ball_seen=~np.isnan(bx),
        bspeed=np.hypot(np.gradient(bx) * fps, np.gradient(by) * fps),
    )
