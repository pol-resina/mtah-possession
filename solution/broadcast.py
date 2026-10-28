"""The broadcast camera's own tracking (ATD, not video) as a second witness of ball contact.

The broadcast camera is zoomed in on the ball, so it sees duels the wide tactical camera blurs.

  1. align    a live broadcast runs in real time, so one constant offset per half maps it onto the
              tactical frames: the offset whose ball positions match the tactical ball most often,
              refined to the frame
  2. gate     a broadcast frame is used only when its players stand where the tactical players
              are, which rules out replays, close-ups and any misalignment
  3. contact  clear contacts are read from it exactly as from the tactical camera, goalkeeper fix
              included
"""

from types import SimpleNamespace

import numpy as np
import pandas as pd

from . import possession, tracking

# Params
SOURCE = "broadcast"
MATCH_DISTANCE = 2.0      # metres: two ball positions this close are the same ball
REFINE_FRAMES = 50        # frame-level search either side of the best whole-second offset
CONSISTENT_WITHIN = 1.5   # metres: median distance from broadcast players to tactical players
MIN_PLAYERS = 6           # broadcast players needed on a frame to judge that
CHUNK = 4000              # frames per block when comparing players, to bound memory


def load_raw(game=tracking.GAME, data_dir=None):
    """(players (frames, slots, 3), ball (frames, 3), team per slot, homography valid per frame)."""
    meta_path, raw_path, homography_path = tracking.require(
        ["atd_meta", "atd_raw", "homography"], game, data_dir, SOURCE)
    team = tracking.team_of_each_track(meta_path)
    _, players, ball = tracking.parse_raw(raw_path, len(team))
    valid = pd.read_csv(homography_path)["valid"].values == 1
    return players, ball, team, valid


def per_second(values, fps):
    """Mean of each whole second, ignoring NaN; NaN where a second has nothing."""
    k = len(values) // fps
    block = values[:k * fps].reshape(k, fps)
    seen = ~np.isnan(block)
    total = np.where(seen, block, 0.0).sum(axis=1)
    count = seen.sum(axis=1)
    return np.where(count > 0, total / np.maximum(count, 1), np.nan)


def find_offset(match, bx, by, first, last):
    """Step 1 for one half: frames to add to a broadcast frame to get the tactical frame.

    No sanity threshold is needed here: with a wrong offset the players never line up, and step 2
    then keeps no frames at all.
    """
    fps = match.fps
    tx, ty = per_second(match.bx, fps), per_second(match.by, fps)
    sx, sy = per_second(bx, fps), per_second(by, fps)
    seconds = np.arange(first // fps, (last + 1) // fps)

    def share(offset):  # share of the half's seconds on which both balls are in the same place
        source = seconds - offset
        inside = (source >= 0) & (source < len(sx))
        gap = np.hypot(sx[source[inside]] - tx[seconds[inside]],
                       sy[source[inside]] - ty[seconds[inside]])
        return np.sum(gap < MATCH_DISTANCE) / float(len(seconds))

    candidates = np.arange(-len(sx), len(tx))
    coarse = candidates[int(np.argmax([share(offset) for offset in candidates]))]

    frames = np.arange(first, last + 1)
    best, best_gap = None, np.inf
    for offset in range(coarse * fps - REFINE_FRAMES, coarse * fps + REFINE_FRAMES + 1):
        source = frames - offset
        inside = (source >= 0) & (source < len(bx))
        gap = np.hypot(bx[source[inside]] - match.bx[frames[inside]],
                       by[source[inside]] - match.by[frames[inside]])
        gap = gap[~np.isnan(gap)]
        if len(gap) and np.median(gap) < best_gap:
            best, best_gap = offset, np.median(gap)
    return best


def lines_up(x, y, tactical_x, tactical_y):
    """Step 2: per frame, do the broadcast players (x, y) stand where the tactical players are?"""
    gap = np.hypot(x[:, :, None] - tactical_x[:, None, :], y[:, :, None] - tactical_y[:, None, :])
    nearest = np.where(np.isnan(gap), np.inf, gap).min(axis=2)
    seen = ~np.isnan(x)
    counted = seen.sum(axis=1)
    with np.errstate(invalid="ignore"):
        ranked = np.sort(np.where(seen, nearest, np.inf), axis=1)
    median = ranked[np.arange(len(x)), np.maximum(counted - 1, 0) // 2]
    return (counted >= MIN_PLAYERS) & (median <= CONSISTENT_WITHIN)


def on_tactical_timeline(match, game=tracking.GAME, data_dir=None):
    """The broadcast tracking re-indexed by tactical frame, NaN wherever it is not usable."""
    players, ball, team, valid = load_raw(game, data_dir)
    ball_x, ball_y = ball[:, 0] * match.length, ball[:, 1] * match.width
    px = np.full((match.n, len(team)), np.nan, dtype=np.float32)
    py = np.full((match.n, len(team)), np.nan, dtype=np.float32)
    bx = np.full(match.n, np.nan)
    by = np.full(match.n, np.nan)
    for _, first, last in match.halves:
        offset = find_offset(match, ball_x, ball_y, first, last)
        if offset is None:  # the two ball tracks never overlap: nothing to align
            continue
        frames = np.arange(first, last + 1)
        source = frames - offset
        inside = (source >= 0) & (source < len(ball))
        frames, source = frames[inside], source[inside]
        calibrated = valid[source]  # the broadcast homography exists on that frame
        frames, source = frames[calibrated], source[calibrated]
        for lo in range(0, len(frames), CHUNK):
            f, s = frames[lo:lo + CHUNK], source[lo:lo + CHUNK]
            x = (players[s, :, 0] * match.length).astype(np.float32)
            y = (players[s, :, 1] * match.width).astype(np.float32)
            live = lines_up(x, y, match.px[f], match.py[f])
            px[f[live]], py[f[live]] = x[live], y[live]
            bx[f[live]], by[f[live]] = ball_x[s[live]], ball_y[s[live]]
    team_at = tracking.team_per_frame(px, py, team, match.halves, match.attacks_right,
                                      match.length, match.width)
    return SimpleNamespace(px=px, py=py, bx=bx, by=by, team_at=team_at)


def contact_evidence(match, game=tracking.GAME, data_dir=None):
    """Step 3: broadcast clear contact per tactical frame (0 home, 1 away, -1 none or unusable)."""
    return possession.contact(*possession.team_distances(on_tactical_timeline(match, game, data_dir)))
