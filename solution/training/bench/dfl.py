"""Load the Sportec Open DFL matches (Bassek et al. 2025, CC BY 4.0) as clean, labelled tracking.

Seven Bundesliga matches of TRACAB optical tracking at 25 fps, corrected by the provider, with a
per-frame ball status (alive/dead) and ball possession (home/away). That is the same nature as the
challenge's ground truth ("D where the corrected tracking marks the ball dead, otherwise the team
it names as owning it", CHALLENGE.md), so the labels convert directly:

    BallStatus == 0                -> D
    BallPossession == 1 (home)     -> H
    BallPossession == 2 (guest)    -> A

D always overrides the owner, as in the challenge.

Coordinates arrive centred on the centre spot; they are moved to a corner origin (0..length,
0..width), the convention of the challenge ATD. Each half becomes one block of rows; the two blocks
are concatenated with HALFTIME_GAP ungraded rows (label X) between them and listed in `halves`, as in
games.half_video_blocks.

Parsing a 400 MB XML takes ~1 min, so the result is cached under cache/dfl/ (gitignored).
"""

import os
import re
import sys
import xml.etree.ElementTree as ET
from glob import glob

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(os.path.dirname(ROOT))  # the repository (ROOT is solution/training/)
DFL_DIR = os.path.join(REPO, "cache", "dfl", "raw")  # where ../download.py puts the data
CACHE_DIR = os.path.join(REPO, "cache", "team_model", "dfl")
FPS = 25

PLAYER_FRAME = re.compile(r'<Frame N="(\d+)" T="[^"]*" X="([-\d.]+)" Y="([-\d.]+)"')
BALL_FRAME = re.compile(r'<Frame N="(\d+)" T="[^"]*" X="([-\d.]+)" Y="([-\d.]+)" Z="([-\d.]+)"'
                        r'[^>]*?BallPossession="(\d)" BallStatus="(\d)"')
HEADER = re.compile(r'GameSection="(\w+)" MatchId="[^"]*" TeamId="([^"]*)" PersonId="([^"]*)"')
SECTIONS = ("firstHalf", "secondHalf")
HALFTIME_GAP = 250  # ungraded rows between the halves, so no metric runs across half-time


def match_ids(dfl_dir=DFL_DIR):
    """Every match with a positions file, sorted."""
    paths = glob(os.path.join(dfl_dir, "DFL_04_03_positions_raw_observed_*.xml"))
    return sorted(re.search(r"(DFL-MAT-\w+)\.xml$", p).group(1) for p in paths)


def _path(kind, match_id, dfl_dir):
    found = glob(os.path.join(dfl_dir, "DFL_{}_*{}.xml".format(kind, match_id)))
    if len(found) != 1:
        raise FileNotFoundError("{} for {} in {}".format(kind, match_id, dfl_dir))
    return found[0]


def match_info(match_id, dfl_dir=DFL_DIR):
    """{'home': team id, 'away': team id, 'keepers': set of person ids, 'names': (home, away)}."""
    root = ET.parse(_path("02_01_matchinformation", match_id, dfl_dir)).getroot()
    teams = {t.get("Role"): t for t in root.iter("Team")}
    keepers = {p.get("PersonId") for p in root.iter("Player") if p.get("PlayingPosition") == "TW"}
    return {"home": teams["home"].get("TeamId"), "away": teams["guest"].get("TeamId"),
            "keepers": keepers,
            "names": (teams["home"].get("TeamName"), teams["guest"].get("TeamName"))}


def parse(match_id, dfl_dir=DFL_DIR):
    """Parse one match into a dict of numpy arrays (see load())."""
    info = match_info(match_id, dfl_dir)
    with open(_path("04_03_positions_raw_observed", match_id, dfl_dir), encoding="utf-8") as fh:
        text = fh.read()
    pitch = re.search(r'<PitchSize X="([\d.]+)" Y="([\d.]+)"', text)
    length, width = float(pitch.group(1)), float(pitch.group(2))

    ball = {}      # section -> (N, x, y, z, possession, status)
    people = {}    # (section, person) -> (team, N, x, y)
    for chunk in text.split("<FrameSet ")[1:]:
        section, team_id, person = HEADER.match(chunk).groups()
        if team_id == "BALL":
            rows = np.array(BALL_FRAME.findall(chunk), dtype=np.float64)
            ball[section] = rows
        else:
            rows = np.array(PLAYER_FRAME.findall(chunk), dtype=np.float64)
            people[(section, person)] = (team_id, rows)
    del text

    persons = sorted({p for _, p in people})
    index = {p: k for k, p in enumerate(persons)}
    team = np.array([0 if next(t for (s, q), (t, _) in people.items() if q == p) == info["home"]
                     else 1 for p in persons], dtype=np.int8)
    keeper = np.array([p in info["keepers"] for p in persons])

    blocks, halves, start = [], [], 0
    for h, section in enumerate(SECTIONS, start=1):
        rows = ball[section]
        n0, n1 = int(rows[0, 0]), int(rows[-1, 0])
        n = n1 - n0 + 1
        if len(rows) != n or np.any(np.diff(rows[:, 0]) != 1):
            raise ValueError("{} {}: ball frames are not contiguous".format(match_id, section))
        pl = np.full((n, len(persons), 2), np.nan, dtype=np.float32)
        for (s, person), (_, prow) in people.items():
            if s != section or not len(prow):
                continue
            k = (prow[:, 0] - n0).astype(np.int64)
            ok = (k >= 0) & (k < n)
            pl[k[ok], index[person], 0] = prow[ok, 1] + length / 2.0
            pl[k[ok], index[person], 1] = prow[ok, 2] + width / 2.0
        bl = np.column_stack([rows[:, 1] + length / 2.0, rows[:, 2] + width / 2.0, rows[:, 3]])
        lab = np.where(rows[:, 5] == 0, "D", np.where(rows[:, 4] == 1, "H", "A"))
        if blocks:  # half-time: ungraded rows, as in the challenge video
            blocks.append((np.full((HALFTIME_GAP, len(persons), 2), np.nan, dtype=np.float32),
                           np.full((HALFTIME_GAP, 3), np.nan, dtype=np.float32),
                           np.full(HALFTIME_GAP, "X")))
            start += HALFTIME_GAP
        blocks.append((pl, bl.astype(np.float32), lab))
        halves.append((h, start, start + n - 1))
        start += n

    players = np.concatenate([b[0] for b in blocks])
    ball_xyz = np.concatenate([b[1] for b in blocks])
    labels = np.concatenate([b[2] for b in blocks]).astype("<U1")

    # Attack direction: the home keeper's mean x in each half says which goal home defends.
    attacks_right = {}
    for h, a, b in halves:
        hk = keeper & (team == 0)
        mean_x = np.nanmean(players[a:b + 1][:, hk, 0])
        attacks_right[h] = bool(mean_x < length / 2.0)

    return {"match_id": match_id, "players": players, "team": team, "keeper": keeper,
            "persons": np.array(persons), "ball": ball_xyz, "labels": labels,
            "halves": np.array(halves, dtype=np.int64), "pitch": np.array([length, width]),
            "attacks_right": np.array([attacks_right[1], attacks_right[2]]),
            "names": np.array(info["names"])}


def load(match_id, refresh=False):
    """Clean labelled match, cached.

    players (n, P, 2) metres, NaN when the person is not on the pitch; team (P,) 0 home 1 away;
    keeper (P,) bool; ball (n, 3) x, y, z metres; labels (n,) 'H'/'A'/'D'; halves (2, 3) rows of
    (half, first row, last row); pitch (length, width); attacks_right (2,) home attacks x = length
    in half 1 / half 2.
    """
    path = os.path.join(CACHE_DIR, match_id + ".npz")
    if os.path.isfile(path) and not refresh:
        with np.load(path) as data:
            return {k: data[k] for k in data.files}
    data = parse(match_id)
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez_compressed(path, **data)
    return data


if __name__ == "__main__":
    ids = sys.argv[1:] or match_ids()
    for mid in ids:
        d = load(mid, refresh=True)
        lab = d["labels"]
        print("{} {} vs {}: {} frames, pitch {}, D {:.1%}, H {:.1%}, A {:.1%}, attacks_right {}".format(
            mid, d["names"][0], d["names"][1], len(lab), d["pitch"], np.mean(lab == "D"),
            np.mean(lab == "H"), np.mean(lab == "A"), d["attacks_right"]))
