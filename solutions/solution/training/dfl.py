"""The open DFL matches as numpy arrays, one compact file per match.

    python solutions/solution/training/dfl.py                 # cache/dfl/raw/*.xml -> cache/dfl/npz/<match>.npz

Each match becomes (25 Hz, both halves one after the other, pitch metres with the origin in a corner
like our own tracking):

    px, py      (frames, slots)  player positions, NaN when the player is not on the pitch
    team        (slots,)         0 home, 1 away
    keeper      (slots,)         True for goalkeepers
    ball        (frames, 3)      x, y, z
    home_ball   (frames,)        True when the home team has the ball (DFL BallPossession)
    alive       (frames,)        True when the ball is in play (DFL BallStatus)
    halves      [(half, first frame, last frame)]
    attacks_right  (2,)          per half: does the home team attack towards x = length?
    pitch       (2,)             length, width

The XML is read with regular expressions, not an XML parser: the files are 400 MB of one repeated
line, and nothing in them is evaluated.
"""

import glob
import mmap
import multiprocessing
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RAW = os.path.join(ROOT, "cache", "dfl", "raw")
NPZ = os.path.join(ROOT, "cache", "dfl", "npz")
MATCHES = ("J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9")
FRAMESET = re.compile(rb'<FrameSet GameSection="(\w+)" MatchId="[^"]*" TeamId="([^"]*)" PersonId="([^"]*)">')
PLAYER = re.compile(rb'<Frame N="(\d+)" T="[^"]*" X="([-\d.]+)" Y="([-\d.]+)"')
BALL = re.compile(rb'<Frame N="(\d+)" T="[^"]*" X="([-\d.]+)" Y="([-\d.]+)" Z="([-\d.]+)"[^>]*'
                  rb'BallPossession="(\d)" BallStatus="(\d)"')
PITCH = re.compile(rb'<PitchSize X="([\d.]+)" Y="([\d.]+)"')
TEAM = re.compile(r'<Team TeamId="([^"]*)"[^>]*Role="(\w+)"(.*?)</Team>', re.S)
PLAYER_INFO = re.compile(r'<Player PersonId="([^"]*)"[^>]*PlayingPosition="([^"]*)"')
SECTIONS = {b"firstHalf": 1, b"secondHalf": 2}


def one(pattern, name):
    found = glob.glob(os.path.join(RAW, pattern.format(name)))
    if len(found) != 1:
        raise FileNotFoundError("{}: run solutions/solution/training/download.py first".format(pattern.format(name)))
    return found[0]


def roster(name):
    """({team id: 0 home / 1 away}, {person id: True if goalkeeper})."""
    with open(one("DFL_02_01_matchinformation*{}.xml", name), encoding="utf-8") as handle:
        text = handle.read()
    teams, keepers = {}, {}
    for team_id, role, body in TEAM.findall(text):
        teams[team_id] = 0 if role == "home" else 1
        for person, position in PLAYER_INFO.findall(body):
            keepers[person] = position == "TW"
    return teams, keepers


def parse(name):
    """One match as a dict of arrays (see the module docstring)."""
    teams, keepers = roster(name)
    with open(one("DFL_04_03_positions*{}.xml", name), "rb") as handle:
        data = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
        length, width = (float(v) for v in PITCH.search(data).groups())
        heads = list(FRAMESET.finditer(data))
        ends = [h.start() for h in heads[1:]] + [len(data)]
        tracks, balls = [], {}
        for head, end in zip(heads, ends):
            section = SECTIONS[head.group(1)]
            team, person = head.group(2).decode(), head.group(3).decode()
            if team == "BALL":
                rows = np.array(BALL.findall(data, head.end(), end), dtype=np.float64)
                balls[section] = rows
            elif team in teams:  # the referees have tracks too; they are left out
                rows = np.array(PLAYER.findall(data, head.end(), end), dtype=np.float64)
                tracks.append((section, team, person, rows))
        data.close()

    first = {s: int(balls[s][:, 0].min()) for s in balls}
    count = {s: int(balls[s][:, 0].max()) - first[s] + 1 for s in balls}
    start = {1: 0, 2: count[1]}
    n = count[1] + count[2]
    people = sorted({(teams[t], p) for _, t, p, _ in tracks})
    slot = {p: i for i, (_, p) in enumerate(people)}
    px = np.full((n, len(people)), np.nan, dtype=np.float32)
    py = np.full((n, len(people)), np.nan, dtype=np.float32)
    for section, _, person, rows in tracks:
        at = rows[:, 0].astype(np.int64) - first[section] + start[section]
        ok = (at >= start[section]) & (at < start[section] + count[section])
        px[at[ok], slot[person]] = rows[ok, 1] + length / 2.0
        py[at[ok], slot[person]] = rows[ok, 2] + width / 2.0
    ball = np.full((n, 3), np.nan, dtype=np.float32)
    home_ball = np.zeros(n, dtype=bool)
    alive = np.zeros(n, dtype=bool)
    for section, rows in balls.items():
        at = rows[:, 0].astype(np.int64) - first[section] + start[section]
        ball[at, 0], ball[at, 1], ball[at, 2] = rows[:, 1] + length / 2.0, rows[:, 2] + width / 2.0, rows[:, 3]
        home_ball[at] = rows[:, 4] == 1
        alive[at] = rows[:, 5] == 1
    team = np.array([t for t, _ in people], dtype=np.int8)
    keeper = np.array([keepers.get(p, False) for _, p in people])
    halves = np.array([(1, 0, count[1] - 1), (2, count[1], n - 1)])
    attacks_right = []
    for _, a, b in halves:  # the home keeper stands in the half the home team defends
        home_keeper = np.nanmean(px[a:b + 1][:, (team == 0) & keeper])
        attacks_right.append(bool(home_keeper < length / 2.0))
    return dict(px=px, py=py, team=team, keeper=keeper, ball=ball, home_ball=home_ball, alive=alive,
                halves=halves, attacks_right=np.array(attacks_right), pitch=np.array([length, width]))


def convert(name):
    path = os.path.join(NPZ, name + ".npz")
    if not os.path.isfile(path):
        np.savez_compressed(path + ".part.npz", **parse(name))
        os.replace(path + ".part.npz", path)
    return name


def load(name):
    with np.load(os.path.join(NPZ, name + ".npz")) as data:
        return {key: data[key] for key in data.files}


if __name__ == '__main__':
    os.makedirs(NPZ, exist_ok=True)
    names = sys.argv[1:] or MATCHES
    with multiprocessing.Pool(3) as pool:
        for done in pool.imap_unordered(convert, names):
            m = load(done)
            n = len(m["alive"])
            spells = int(np.sum(np.diff(m["home_ball"][m["alive"]].astype(np.int8)) != 0)) + 1
            print("{}: {} frames ({:.0f} min), {} slots, pitch {:.0f} x {:.0f}, alive {:.1%}, home ball {:.1%} "
                  "of alive, about {} changes of team while alive".format(
                      done, n, n / 25 / 60, len(m["team"]), m["pitch"][0], m["pitch"][1], m["alive"].mean(),
                      m["home_ball"][m["alive"]].mean(), spells))
            sys.stdout.flush()
