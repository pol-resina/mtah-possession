"""The one data format every model on the bench consumes: an observed match.

It mirrors what pipeline/atd.load_tracking returns for the challenge match, plus the match facts
that the challenge gives in its metadata, so a model written for the challenge runs on a degraded
DFL match without knowing the difference:

    track       {'players': (n, S, 3) x/length, y/width, NaN speed;  'ball': (n, 2) x/length,
                 y/width;  'team': (S,) 0 home / 1 away}  -- normalised, NaN when not tracked
    pitch       (length, width) metres
    halves      [(half, first row, last row)] graded rows, inclusive
    attacks_right {1: bool, 2: bool}  home attacks x = length (ATD metadata gives this)
    kickoffs    {half: kickoff row}
    fps         25
    name        short id for reports

Truth labels never travel inside an Obs; the harness keeps them apart.
"""

from types import SimpleNamespace

import numpy as np

FPS = 25


def make(players_m, ball_m, team, pitch, halves, attacks_right, kickoffs=None, name=""):
    """Build an Obs from metric arrays (players (n, S, 2) metres, ball (n, 2) metres)."""
    length, width = float(pitch[0]), float(pitch[1])
    n, s = players_m.shape[:2]
    players = np.full((n, s, 3), np.nan, dtype=np.float32)
    players[:, :, 0] = players_m[:, :, 0] / length
    players[:, :, 1] = players_m[:, :, 1] / width
    ball = np.column_stack([ball_m[:, 0] / length, ball_m[:, 1] / width]).astype(np.float32)
    halves = [(int(h), int(a), int(b)) for h, a, b in halves]
    return SimpleNamespace(
        track={"players": players, "ball": ball, "team": np.asarray(team, dtype=np.int64)},
        pitch=(length, width), halves=halves,
        attacks_right={int(k): bool(v) for k, v in dict(attacks_right).items()},
        kickoffs=kickoffs or {h: a for h, a, _ in halves}, fps=FPS, name=name)


def challenge(game="BAR-ATH"):
    """The real challenge match as an Obs (no labels)."""
    import os
    import sys
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, root)
    from games import GAMES, half_video_blocks, require
    from pipeline.atd import load_tracking
    from starter.load_smart_tagging import KICKOFF_VIDEO_FRAMES
    import xml.etree.ElementTree as ET

    track = load_tracking(game)
    meta_path, = require(["atd_meta"], game)
    home = next(ET.parse(meta_path).getroot().iter("Team"))
    first = None
    for p in home.iter("ProviderParameter"):
        if p.findtext("Name") == "attack_direction_first_half":
            first = p.findtext("Value") == "left_to_right"
    halves = half_video_blocks(game)
    kick = {1: KICKOFF_VIDEO_FRAMES["1st Half"], 2: KICKOFF_VIDEO_FRAMES["2nd Half"]}
    return SimpleNamespace(track=track, pitch=tuple(GAMES[game]["pitch"]), halves=halves,
                           attacks_right={1: first, 2: not first}, kickoffs=kick, fps=FPS,
                           name=game)


def graded_mask(obs):
    n = len(obs.track["ball"])
    g = np.zeros(n, dtype=bool)
    for _, a, b in obs.halves:
        g[a:b + 1] = True
    return g
