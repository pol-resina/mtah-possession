"""Touch features: who last changed the ball's motion.

Possession changes when a player of the other team plays the ball. The shipped features see who is
near the ball; these add who made it change speed or direction. A touch is a frame where the ball's
velocity changes by more than KINK between TOUCH_SPAN frames before and after, with a player within
REACH of it; it belongs to the nearer team when that team is clearly nearer.

Used through core.FEATURES["touch"]; in the solution, through features.py.
"""

import numpy as np

import features as FT
import tracking

FPS = 25
TOUCH_SPAN = 2        # frames either side over which the velocity change is measured
KINK = 3.0            # m/s: a change of ball velocity this large is a touch, not rolling friction
REACH = 2.5           # metres: a player this close could have made it
MARGIN = 0.5          # metres by which one team must be nearer than the other to own the touch
LONG = 30.0           # seconds; "never" for the time-since features
WINDOWS_S = (1.0, 2.0, 5.0, 10.0)


def _since(event):
    """Seconds since the last True (LONG before the first)."""
    idx = np.arange(len(event))
    last = np.maximum.accumulate(np.where(event, idx, -10 ** 9))
    return np.minimum((idx - last) / float(FPS), LONG)


def _count(event, seconds, future):
    """Number of True in the past (or future) `seconds`, this frame included."""
    w = int(seconds * FPS)
    c = np.concatenate(([0], np.cumsum(event)))
    i = np.arange(len(event))
    if future:
        return (c[np.minimum(i + w, len(event))] - c[i]).astype(np.float64)
    return (c[i + 1] - c[np.maximum(i + 1 - w, 0)]).astype(np.float64)


def touches(px, py, team_at, bx, by):
    """(home touch, away touch, any kink) per frame, booleans."""
    seen = ~np.isnan(bx)
    vx, vy = np.gradient(bx) * FPS, np.gradient(by) * FPS
    k = TOUCH_SPAN
    dvx = np.full(len(bx), np.nan)
    dvy = np.full(len(bx), np.nan)
    dvx[k:-k] = vx[2 * k:] - vx[:-2 * k]
    dvy[k:-k] = vy[2 * k:] - vy[:-2 * k]
    with np.errstate(invalid="ignore"):
        kink = np.hypot(dvx, dvy) > KINK
    kink &= seen
    # keep one frame per burst: the first of each run of kink frames
    first = kink & ~np.concatenate(([False], kink[:-1]))
    d = np.hypot(px - bx[:, None], py - by[:, None])
    d = np.where(np.isnan(d), np.inf, d)
    dh = np.where(team_at == 0, d, np.inf).min(axis=1)
    da = np.where(team_at == 1, d, np.inf).min(axis=1)
    home = first & (dh < REACH) & (dh < da - MARGIN)
    away = first & (da < REACH) & (da < dh - MARGIN)
    return home, away, first


def frame_features(match, obs, swap=False):
    """(names, matrix) at 25 fps: the shipped features plus the touch features."""
    names, X = FT.frame_features(obs, swap=swap)
    team = np.asarray(obs.track["team"]).astype(np.int64)
    attacks_right = dict(obs.attacks_right)
    if swap:
        team = 1 - team
        attacks_right = {h: not v for h, v in attacks_right.items()}
    team_at = tracking.team_per_frame(match.px, match.py, team, obs.halves, attacks_right,
                                      match.length, match.width)
    home, away, kink = touches(match.px, match.py, team_at, match.bx, match.by)
    f = {}
    f["touch_since_home"] = _since(home)
    f["touch_since_away"] = _since(away)
    f["touch_since_diff"] = f["touch_since_away"] - f["touch_since_home"]   # + : home touched last
    f["touch_until_home"] = _since(home[::-1])[::-1]
    f["touch_until_away"] = _since(away[::-1])[::-1]
    f["touch_until_diff"] = f["touch_until_away"] - f["touch_until_home"]   # + : home touches next
    f["touch_last"] = np.sign(f["touch_since_diff"])
    f["touch_next"] = np.sign(f["touch_until_diff"])
    f["kink_since"] = _since(kink)
    for w in WINDOWS_S:
        for label, future in (("p", False), ("f", True)):
            h, a = _count(home, w, future), _count(away, w, future)
            f["touch_n_home_{}{}".format(label, w)] = h
            f["touch_n_away_{}{}".format(label, w)] = a
            f["touch_n_diff_{}{}".format(label, w)] = h - a
    # run of consecutive touches by the same team up to now: + home, - away
    run = np.zeros(len(home))
    count, side = 0, 0
    which = np.where(home, 1, np.where(away, -1, 0))
    for i in np.flatnonzero(which):
        if which[i] == side:
            count += 1
        else:
            side, count = which[i], 1
        run[i] = side * count
    idx = np.maximum.accumulate(np.where(which != 0, np.arange(len(which)), 0))
    f["touch_run"] = np.where(np.cumsum(which != 0) > 0, run[idx], 0.0)
    extra = list(f)
    X = np.column_stack([X] + [np.asarray(f[k], dtype=np.float32) for k in extra])
    return names + extra, X
