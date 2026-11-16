"""A DFL match made to look like our tracking (ATD), so a model can be trained and tested on it.

The DFL tracking is clean: every player on every frame, a ball on every frame. Ours is not. What is
taken away here, and how much, is set from aggregate measurements of our own match's tracking (no
labels involved); each number says where it comes from.

    python solution/training/sim.py            # the degraded matches' statistics next to our tracking's
"""

import os
import sys
from types import SimpleNamespace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SOLUTION = os.path.dirname(HERE)                   # solution/
ROOT = os.path.dirname(SOLUTION)                   # the repository
for path in (SOLUTION, HERE):
    if path not in sys.path:
        sys.path.insert(0, path)
import dfl  # noqa: E402
import tracking  # noqa: E402  (the solution's: team_per_frame, MAX_PLAYER_SPEED)

FPS = 25
# Params. Targets measured on our tracking (graded frames of BAR-ATH), in brackets.
# -- ball not tracked --------------------------------------------------------------------------
# In our tracking the ball is lost for most of a stoppage in 21% of those under 5 s, about 63% of
# those of 5-20 s and about 85% of longer ones (stoppages as our rule-based step finds them); when
# it is lost, that is from the start (median 0.0 s) until just before the restart (0.4-0.8 s).
DEAD_HIDDEN = ((5.0, 0.20), (20.0, 0.63), (1e9, 0.85))   # (stoppages shorter than, share hidden)
DEAD_MIN = 1.0            # seconds; shorter stoppages keep their ball
DEAD_HEAD = (0.0, 0.5)    # seconds of ball still tracked after a stoppage starts...
DEAD_TAIL = (0.2, 1.0)    # ...and tracked again before it ends
LOST_PER_MIN = 1.3        # ball lost while in play, times per minute of play [7% of live frames,
LOST_MEDIAN = 1.5         # runs of median 1.8 s, 90th percentile 8 s]
LOST_SIGMA = 1.15         # spread of the log-normal run length
LOST_RANGE = (0.3, 16.0)  # seconds
# -- ball tracked in the wrong place -----------------------------------------------------------
# The tracker bridges short gaps with a straight line, which is wrong whenever the ball was touched
# meanwhile. Our two cameras put the ball over 3 m apart on 21% of frames, mostly along the line of
# sight; if the tactical one is the wrong one in about a third of those, its ball is off by more
# than 3 m on about 7% of the frames it reports.
BRIDGE_PER_MIN = 6.0      # bridged gaps per minute of play
BRIDGE_RANGE = (0.4, 4.0)  # seconds; 4 s is the longest gap our tracker bridges
BRIDGE_CROWD = 3.0        # times likelier to start with two or more players within 3 m (occlusion)
AIR = 2.0                 # metres: above this the ball is not detected (flights are bridged)
# -- players -----------------------------------------------------------------------------------
DROP_PER_MIN = 0.42       # short losses of one player, per player-minute [in play 21.5 of 22
DROP_MEAN = 3.0           # are tracked, keepers included: the camera sees the whole pitch]; seconds
# Few players tracked happens in long stoppages only [fewer than 18 on 13% of the frames of
# stoppages of 40-80 s and 49% of longer ones, about 0 in shorter ones]: the picture closes in.
ZOOM = ((40.0, 0.0), (80.0, 0.30), (1e9, 0.75))   # (stoppages shorter than, share that close in)
ZOOM_RADIUS = 25.0        # metres still in view
SMOOTH_PLAYERS = 6.0      # frames (sigma): our tracks are smooth, speed changes 0.07 m/s per frame
SMOOTH_BALL = 7.0         # frames: our ball's speed changes by under 1 m/s per frame 99% of the time
NOISE = 0.15              # metres of slowly varying position error (homography)
NOISE_SMOOTH = 2.0        # seconds
SWAPS = 12                # times per match two opponents next to the ball trade tracks [a
SWAP_MEAN = 8.0           # re-teamed player is within 3 m of the ball on 0.7% of live frames]


def runs(mask):
    edges = np.diff(np.concatenate(([0], np.asarray(mask, dtype=np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def smooth_noise(rng, shape, sigma, seconds):
    """Gaussian noise of standard deviation sigma, correlated over about `seconds`."""
    width = int(seconds * FPS)
    raw = rng.standard_normal((shape[0] + 2 * width,) + shape[1:]).astype(np.float32)
    kernel = np.ones(width, dtype=np.float32) / width
    csum = np.cumsum(raw, axis=0)
    out = (csum[width:] - csum[:-width])[:shape[0]] / width
    return out * (sigma * np.sqrt(width))


def smooth(values, sigma):
    """Gaussian smoothing along time that ignores NaN (a NaN stays NaN)."""
    from scipy.ndimage import gaussian_filter1d
    ok = ~np.isnan(values)
    total = gaussian_filter1d(np.where(ok, values, 0.0).astype(np.float64), sigma, axis=0, mode="nearest")
    weight = gaussian_filter1d(ok.astype(np.float64), sigma, axis=0, mode="nearest")
    return np.where(ok, total / np.maximum(weight, 1e-9), np.nan)


def degrade_ball(m, rng, in_play=True):
    """(bx, by) with losses (NaN) and straight-line bridges; also the mask of bridged frames.
    in_play=False leaves the ball untouched while it is in play (a perfect ball tracker), to
    measure what the ball's faults cost."""
    n = len(m["alive"])
    bx, by, bz = (m["ball"][:, i].astype(np.float64) for i in range(3))
    alive = m["alive"] if in_play else np.zeros(n, dtype=bool)
    lost = np.zeros(n, dtype=bool)
    bridged = np.zeros(n, dtype=bool)

    # bridges: flights, and random short gaps (likelier in a crowd)
    spans = [(a, b) for a, b in runs(alive & (bz > AIR)) if b - a <= BRIDGE_RANGE[1] * FPS]
    d = np.hypot(m["px"] - bx[:, None], m["py"] - by[:, None])
    crowd = (np.where(np.isnan(d), np.inf, d) < 3.0).sum(axis=1) >= 2
    weight = np.where(alive, np.where(crowd, BRIDGE_CROWD, 1.0), 0.0)
    count = rng.poisson(BRIDGE_PER_MIN * alive.sum() / FPS / 60.0)
    starts = rng.choice(n, size=count, p=weight / weight.sum()) if alive.any() else []
    lengths = (rng.uniform(*BRIDGE_RANGE, size=count) * FPS).astype(int)
    spans += [(int(s), int(s + k)) for s, k in zip(starts, lengths)]
    for a, b in spans:
        a, b = max(a, 1), min(b, n - 1)
        if b <= a or not alive[a - 1:b + 1].all() or bridged[a - 1:b + 1].any():
            continue
        t = (np.arange(a, b) - (a - 1)) / float(b - a + 1)
        bx[a:b] = bx[a - 1] + t * (bx[b] - bx[a - 1])
        by[a:b] = by[a - 1] + t * (by[b] - by[a - 1])
        bridged[a:b] = True

    # long flights are simply lost
    for a, b in runs(alive & (bz > AIR)):
        if b - a > BRIDGE_RANGE[1] * FPS:
            lost[a:b] = True
    # lost while in play
    count = rng.poisson(LOST_PER_MIN * alive.sum() / FPS / 60.0)
    starts = rng.choice(np.flatnonzero(alive), size=count) if alive.any() else []
    lengths = np.clip(rng.lognormal(np.log(LOST_MEDIAN), LOST_SIGMA, size=count), *LOST_RANGE)
    for s, k in zip(starts, (lengths * FPS).astype(int)):
        lost[s:s + k] = True
    # lost during stoppages
    for a, b in runs(~m["alive"]):
        hidden = next(share for limit, share in DEAD_HIDDEN if (b - a) / FPS < limit)
        if b - a >= DEAD_MIN * FPS and rng.uniform() < hidden:
            head = int(rng.uniform(*DEAD_HEAD) * FPS)
            tail = int(rng.uniform(*DEAD_TAIL) * FPS)
            if a + head < b - tail:
                lost[a + head:b - tail] = True
    bx, by = smooth(bx, SMOOTH_BALL), smooth(by, SMOOTH_BALL)
    bx[lost], by[lost] = np.nan, np.nan
    return bx, by, bridged & ~lost


def degrade_players(m, bx, by, rng):
    """(px, py) with players out of view, short losses, position error and a few track swaps."""
    n, slots = m["px"].shape
    px, py = m["px"].astype(np.float32).copy(), m["py"].astype(np.float32).copy()
    # track swaps next to the ball
    alive = m["alive"]
    d = np.hypot(px - m["ball"][:, 0:1], py - m["ball"][:, 1:2])
    d = np.where(np.isnan(d), np.inf, d)
    home, away = m["team"] == 0, m["team"] == 1
    done = 0
    for start in rng.choice(np.flatnonzero(alive), size=SWAPS * 40):
        i = int(np.argmin(np.where(home, d[start], np.inf)))
        j = int(np.argmin(np.where(away, d[start], np.inf)))
        if not (d[start, i] < 3.0 and d[start, j] < 3.0):
            continue  # a swap needs two opponents next to the ball
        stop = min(n, start + int(rng.exponential(SWAP_MEAN) * FPS) + 1)
        px[start:stop, [i, j]] = px[start:stop, [j, i]]
        py[start:stop, [i, j]] = py[start:stop, [j, i]]
        d[start:stop, [i, j]] = d[start:stop, [j, i]]
        done += 1
        if done == SWAPS:
            break
    # position error, then the tracker's smoothing
    px += smooth_noise(rng, px.shape, NOISE, NOISE_SMOOTH)
    py += smooth_noise(rng, py.shape, NOISE, NOISE_SMOOTH)
    px = smooth(px, SMOOTH_PLAYERS).astype(np.float32)
    py = smooth(py, SMOOTH_PLAYERS).astype(np.float32)
    hidden = np.zeros(px.shape, dtype=bool)
    # short losses of single players
    for s in range(slots):
        count = rng.poisson(DROP_PER_MIN * n / FPS / 60.0)
        for a, k in zip(rng.randint(0, n, size=count), (rng.exponential(DROP_MEAN, size=count) * FPS).astype(int)):
            hidden[a:a + k + 1, s] = True
    # stoppages in which the picture closes in on the incident
    for a, b in runs(~alive):
        if rng.uniform() < next(share for limit, share in ZOOM if (b - a) / FPS < limit):
            lo = a + int(rng.uniform(0.0, 0.3) * (b - a))
            hi = b - int(rng.uniform(0.0, 0.3) * (b - a))
            focus = m["ball"][lo, :2]
            hidden[lo:hi] |= np.hypot(px[lo:hi] - focus[0], py[lo:hi] - focus[1]) > ZOOM_RADIUS
    px[hidden], py[hidden] = np.nan, np.nan
    return px, py


def controlled(truth, seconds):
    """The labels with brief touches removed: a spell shorter than `seconds` that has the other team
    on both sides, with no stoppage in between, goes to that team (repeated until nothing changes).

    DFL changes the team in possession at every touch: 27-37% of its spells last under 2 s, 8 to 11
    spells per minute of play. Our ground truth asks for controlled possession (CHALLENGE.md) and has
    12% under 2 s and 4.2 per minute. Merging sandwiched spells under 2 s brings the DFL labels to
    13-19% under 2 s.
    """
    out = truth.copy()
    if seconds <= 0:
        return out
    while True:
        bounds = np.concatenate(([0], np.flatnonzero(out[1:] != out[:-1]) + 1, [len(out)]))
        changed = False
        for i in range(1, len(bounds) - 2):
            a, b = bounds[i], bounds[i + 1]
            if (out[a] in "HA" and b - a < seconds * FPS and out[a - 1] in "HA" and out[b] in "HA"
                    and out[a - 1] == out[b] != out[a]):
                out[a:b] = out[a - 1]
                changed = True
        if not changed:
            return out


def truth_of(name, control=0.0):
    """Letters H / A / D per frame of a DFL match, as labelled or with brief touches removed."""
    m = dfl.load(name)
    return controlled(np.where(m["alive"], np.where(m["home_ball"], "H", "A"), "D").astype("<U1"), control)


def atd_like(name, seed, control=0.0, clean_ball=False):
    """The match `name`, degraded with this seed: an object shaped like tracking.load()'s, plus
    `truth` (letters H / A / D per frame; control > 0: brief touches removed, see controlled()),
    `kickoffs` and `bridged`. clean_ball: the ball is never lost or bridged while in play."""
    m = dfl.load(name)
    rng = np.random.RandomState(seed)
    length, width = float(m["pitch"][0]), float(m["pitch"][1])
    n = len(m["alive"])
    bx, by, bridged = degrade_ball(m, rng, in_play=not clean_ball)
    px, py = degrade_players(m, bx, by, rng)
    halves = [(int(h), int(a), int(b)) for h, a, b in m["halves"]]
    half = np.zeros(n, dtype=np.int8)
    for number, first, last in halves:
        half[first:last + 1] = number
    attacks_right = {1: bool(m["attacks_right"][0]), 2: bool(m["attacks_right"][1])}
    speed = np.hypot(np.gradient(px, axis=0) * FPS, np.gradient(py, axis=0) * FPS)
    speed[speed > tracking.MAX_PLAYER_SPEED] = np.nan
    truth = controlled(np.where(m["alive"], np.where(m["home_ball"], "H", "A"), "D").astype("<U1"),
                       control)
    kickoffs = {h: int(a + np.argmax(m["alive"][a:b + 1])) for h, a, b in halves}
    return SimpleNamespace(
        game=name, n=n, fps=FPS, length=length, width=width, halves=halves, half=half,
        attacks_right=attacks_right, px=px, py=py, speed=speed, team=m["team"].astype(np.int64),
        team_at=tracking.team_per_frame(px, py, m["team"].astype(np.int64), halves, attacks_right, length, width),
        bx=bx, by=by, ball_seen=~np.isnan(bx),
        bspeed=np.hypot(np.gradient(bx) * FPS, np.gradient(by) * FPS),
        truth=truth, kickoffs=kickoffs, bridged=bridged, true_ball=m["ball"], alive=m["alive"])


def statistics(match):
    """Aggregate statistics comparable with our tracking's."""
    alive = match.alive
    tracked = ~np.isnan(match.px)
    count = tracked.sum(axis=1)
    lost = ~match.ball_seen
    off = np.hypot(match.bx - match.true_ball[:, 0], match.by - match.true_ball[:, 1])
    seen_alive = alive & match.ball_seen
    lost_runs = np.array([b - a for a, b in runs(lost & alive)]) / float(FPS)
    dv = np.diff(np.gradient(match.px, axis=0) * FPS, axis=0)
    dbv = np.abs(np.diff(np.gradient(match.bx) * FPS))
    return {
        "tracked players, in play": count[alive].mean(),
        "all 22 tracked, in play": np.mean(count[alive] >= 22),
        "tracked players, stoppages": count[~alive].mean(),
        "under 18 tracked, stoppages": np.mean(count[~alive] < 18),
        "ball lost, in play": lost[alive].mean(),
        "ball lost, stoppages": lost[~alive].mean(),
        "mostly lost, stoppages 5-20 s": np.mean([lost[a:b].mean() > 0.5 for a, b in runs(~alive)
                                                   if 5 * FPS <= b - a < 20 * FPS]),
        "mostly lost, stoppages 20 s+": np.mean([lost[a:b].mean() > 0.5 for a, b in runs(~alive)
                                                  if b - a >= 20 * FPS]),
        "lost runs in play, median s": np.median(lost_runs),
        "lost runs in play, p90 s": np.percentile(lost_runs, 90),
        "ball over 3 m off, in play": np.mean(off[seen_alive] > 3.0),
        "ball bridged, in play": match.bridged[seen_alive].mean(),
        "player speed change / frame": np.nanstd(dv[alive[1:]]),
        "ball speed change, median": np.nanmedian(dbv[alive[1:]]),
        "ball speed change, p99": np.nanpercentile(dbv[alive[1:]], 99),
    }


OURS = {"tracked players, in play": 21.53, "all 22 tracked, in play": 0.64,
        "tracked players, stoppages": 19.48, "under 18 tracked, stoppages": 0.177,
        "ball lost, in play": 0.071, "ball lost, stoppages": 0.733,
        "mostly lost, stoppages 5-20 s": 0.63, "mostly lost, stoppages 20 s+": 0.85,
        "lost runs in play, median s": 1.8, "lost runs in play, p90 s": 7.9,
        "ball over 3 m off, in play": 0.07, "ball bridged, in play": float("nan"),
        "player speed change / frame": 0.067, "ball speed change, median": 0.132,
        "ball speed change, p99": 0.92}

if __name__ == '__main__':
    rows = [statistics(atd_like(name, 0)) for name in (sys.argv[1:] or dfl.MATCHES[:3])]
    print("{:<30} {:>8} {:>8}   per match".format("", "ours", "bench"))
    for key in OURS:
        values = [r[key] for r in rows]
        print("{:<30} {:>8.3f} {:>8.3f}   {}".format(key, OURS[key], np.mean(values),
                                                    " ".join("{:.3f}".format(v) for v in values)))
