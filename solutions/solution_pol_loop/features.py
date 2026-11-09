"""Copied from FCB_challenges/bench/features.py (top_speed from bench/stats.py): keep in step.

Per-frame features for the learned models, from tracking only, computed at 25 fps and sampled
at 5 Hz.

Nothing follows a player: every feature is collective (per team or global) or about the ball, so
identity switches and fragmented slots barely matter. Goalkeeper tracks get the team defending the
half they stand in (the fix from solution/tracking.py), since that is the one team error we know the
ATD makes.

All coordinates are in the home team's attacking frame: x grows towards the goal home attacks, so a
feature means the same in both halves. Swapping H and A (augmentation) then only needs the team
array flipped and the frame mirrored, which build() does when swap=True.

Groups (the plan's table):
    ball         seen, position, distance to lines, speed, turn, time since / until seen, still
    ball-team    nearest player per team, their difference, players within 3 and 5 m
    no ball      per team: centroid, spread, mean / p90 speed, forward velocity, count; global speed
    context      mean / min / max over past and future windows of 0.4, 1, 2, 5, 10 s
"""

import numpy as np
from scipy.ndimage import maximum_filter1d, minimum_filter1d, uniform_filter1d

TOP_K = 5  # fastest players in top_speed


def top_speed(players_m, fps=25, half_width=12):
    """Mean speed of the TOP_K fastest tracked players per frame, 1 s centred mean, m/s.

    Slot jumps (> 12 m/s) are ignored, as in solution/tracking.py.
    """
    sp = np.hypot(np.gradient(players_m[:, :, 0], axis=0),
                  np.gradient(players_m[:, :, 1], axis=0)) * fps
    sp[sp > 12.0] = np.nan
    sp = np.where(np.isnan(sp), -1.0, sp)
    fastest = np.sort(sp, axis=1)[:, -TOP_K:]
    ok = fastest >= 0
    n = ok.sum(1)
    per = np.where(n > 0, np.where(ok, fastest, 0).sum(1) / np.maximum(n, 1), np.nan)
    seen = ~np.isnan(per)
    cs = np.concatenate([[0], np.cumsum(np.where(seen, per, 0))])
    cn = np.concatenate([[0], np.cumsum(seen)])
    i = np.arange(len(per))
    lo, hi = np.maximum(0, i - half_width), np.minimum(len(per), i + half_width + 1)
    c = cn[hi] - cn[lo]
    return np.where(c > 0, (cs[hi] - cs[lo]) / np.maximum(c, 1), np.nan)

STEP = 5                       # 25 fps -> 5 Hz
WINDOWS_S = (0.4, 1.0, 2.0, 5.0, 10.0)
FPS = 25


def _team_at(px, py, team, halves, attacks_right, L, W):
    from tracking import team_per_frame
    return team_per_frame(px, py, team, halves, attacks_right, L, W)


def _since(seen):
    """Frames since the last True (0 on True frames); large before the first."""
    idx = np.arange(len(seen))
    last = np.maximum.accumulate(np.where(seen, idx, -10 ** 6))
    return (idx - last).astype(np.float32)


def _hold(x, seen):
    """Last seen value carried forward (NaN before the first)."""
    idx = np.where(seen, np.arange(len(x)), 0)
    idx = np.maximum.accumulate(idx)
    out = x[idx].copy()
    out[np.cumsum(seen) == 0] = np.nan
    return out


def _nanmean_window(x, w):
    ok = ~np.isnan(x)
    s = uniform_filter1d(np.where(ok, x, 0.0), w, mode="nearest")
    c = uniform_filter1d(ok.astype(np.float64), w, mode="nearest")
    return np.where(c > 0, s / np.maximum(c, 1e-9), np.nan)


def _shift(x, k):
    """y[t] = x[t + k], edge-padded."""
    if k == 0:
        return x
    out = np.empty_like(x)
    if k > 0:
        out[:-k] = x[k:]
        out[-k:] = x[-1]
    else:
        out[-k:] = x[:k]
        out[:-k] = x[0]
    return out


def _row_p90(a):
    """np.nanpercentile(a, 90, axis=1) (linear interpolation), via one sort."""
    s = np.sort(a, axis=1)  # NaN last
    c = (~np.isnan(a)).sum(1)
    pos = 0.9 * np.maximum(c - 1, 0)
    lo = np.floor(pos).astype(np.int64)
    hi = np.minimum(lo + 1, np.maximum(c - 1, 0))
    r = np.arange(len(a))
    v = s[r, lo] + (s[r, hi] - s[r, lo]) * (pos - lo)
    return np.where(c > 0, v, np.nan)


def context(x, name):
    """{name_stat_dir_w: array}: mean, min, max over past / future windows."""
    out = {}
    finite = ~np.isnan(x)
    big = 1e6
    for ws in WINDOWS_S:
        w = int(round(ws * FPS)) | 1  # odd
        half = w // 2
        m = _nanmean_window(x, w)
        lo = minimum_filter1d(np.where(finite, x, big), w, mode="nearest")
        hi = maximum_filter1d(np.where(finite, x, -big), w, mode="nearest")
        lo = np.where(lo >= big, np.nan, lo)
        hi = np.where(hi <= -big, np.nan, hi)
        for d, k in (("p", -half), ("f", half)):
            out["{}_mean_{}{}".format(name, d, ws)] = _shift(m, k)
            out["{}_min_{}{}".format(name, d, ws)] = _shift(lo, k)
            out["{}_max_{}{}".format(name, d, ws)] = _shift(hi, k)
    return out


def frame_features(obs, swap=False):
    """(names, matrix (n, F) float32) at 25 fps. swap=True: H and A exchanged, frame mirrored."""
    L, W = obs.pitch
    P, B = obs.track["players"], obs.track["ball"]
    team = np.asarray(obs.track["team"]).astype(np.int64)
    attacks_right = dict(obs.attacks_right)
    if swap:
        team = 1 - team
        attacks_right = {h: not v for h, v in attacks_right.items()}
    n = len(B)
    px = (P[:, :, 0] * L).astype(np.float64)
    py = (P[:, :, 1] * W).astype(np.float64)
    bx = B[:, 0].astype(np.float64) * L
    by = B[:, 1].astype(np.float64) * W
    if swap:  # mirror across the long axis too, so the swapped copy is a different-looking match
        py, by = W - py, W - by
    team_at = _team_at(px.astype(np.float32), py.astype(np.float32), team, obs.halves,
                       attacks_right, L, W)

    # home attacking frame: x grows towards the goal home attacks
    flip = np.zeros(n, dtype=bool)
    for h, a, b in obs.halves:
        flip[a:b + 1] = not attacks_right[h]
    fx = np.where(flip[:, None], L - px, px)
    fbx = np.where(flip, L - bx, bx)

    f = {}
    seen = ~np.isnan(bx)
    f["ball_seen"] = seen.astype(np.float64)
    f["ball_x"] = fbx
    f["ball_y_c"] = np.abs(by - W / 2.0)
    f["ball_line_dist"] = np.minimum.reduce([bx, L - bx, by, W - by])  # negative outside
    f["ball_touch_dist"] = np.minimum(by, W - by)
    f["ball_goal_dist"] = np.minimum(bx, L - bx)
    vx, vy = np.gradient(bx) * FPS, np.gradient(by) * FPS
    sp = np.hypot(vx, vy)
    f["ball_speed"] = _nanmean_window(sp, 5)
    ang = np.arctan2(vy, vx)
    turn = np.abs(np.angle(np.exp(1j * (_shift(ang, 5) - _shift(ang, -5)))))
    f["ball_turn"] = np.where(sp > 1.0, turn, np.nan)
    since = _since(seen)
    until = _since(seen[::-1])[::-1]
    f["ball_since_s"] = np.minimum(since, 3000) / FPS
    f["ball_until_s"] = np.minimum(until, 3000) / FPS
    hx, hy = _hold(fbx, seen), _hold(by, seen)
    f["held_x"] = hx
    f["held_y_c"] = np.abs(hy - W / 2.0)
    hbx = _hold(bx, seen)
    f["held_line_dist"] = np.minimum.reduce([hbx, L - hbx, hy, W - hy])
    f["ball_still"] = np.where(seen, (f["ball_speed"] < 0.5).astype(np.float64), np.nan)

    # ball-team (with the held ball when it is lost; ball_since_s says how stale it is)
    ux = np.where(seen, bx, hbx)
    uy = np.where(seen, by, hy)
    d = np.hypot(px - ux[:, None], py - uy[:, None])
    home = team_at == 0
    away = team_at == 1
    dh = np.where(home & ~np.isnan(d), d, np.inf).min(1)
    da = np.where(away & ~np.isnan(d), d, np.inf).min(1)
    dh[~np.isfinite(dh)] = np.nan
    da[~np.isfinite(da)] = np.nan
    f["d_home"], f["d_away"] = dh, da
    f["d_diff"] = dh - da
    f["d_min"] = np.fmin(dh, da)
    for r in (3.0, 5.0):
        f["n_home_{}m".format(int(r))] = np.nansum(home & (d < r), axis=1).astype(np.float64)
        f["n_away_{}m".format(int(r))] = np.nansum(away & (d < r), axis=1).astype(np.float64)
    f["contact"] = np.where(f["d_min"] < 1.5, np.sign(da - dh), 0.0)  # +1 home closer, -1 away

    # without the ball
    pspeed = np.hypot(np.gradient(px, axis=0), np.gradient(py, axis=0)) * FPS
    pspeed[pspeed > 12.0] = np.nan
    pvx = np.gradient(fx, axis=0) * FPS
    pvx[np.abs(pvx) > 12.0] = np.nan
    for k, mask in (("home", home), ("away", away)):
        ok = mask & ~np.isnan(px)
        cnt = ok.sum(1).astype(np.float64)
        f["{}_n".format(k)] = cnt
        with np.errstate(invalid="ignore", divide="ignore"):
            mx = np.where(ok, fx, np.nan)
            my = np.where(ok, py, np.nan)
            f["{}_cx".format(k)] = np.nanmean(mx, 1)
            f["{}_cy".format(k)] = np.nanmean(my, 1)
            f["{}_sx".format(k)] = np.nanstd(mx, 1)
            f["{}_sy".format(k)] = np.nanstd(my, 1)
            f["{}_xmax".format(k)] = np.nanmax(np.where(ok, fx, -np.inf), 1)
            f["{}_xmin".format(k)] = np.nanmin(np.where(ok, fx, np.inf), 1)
            sp_k = np.where(ok, pspeed, np.nan)
            f["{}_speed".format(k)] = np.nanmean(sp_k, 1)
            f["{}_speed_p90".format(k)] = _row_p90(sp_k)
            f["{}_vx".format(k)] = np.nanmean(np.where(ok, pvx, np.nan), 1)
        for c in ("xmax", "xmin"):
            v = f["{}_{}".format(k, c)]
            v[~np.isfinite(v)] = np.nan
    f["cx_diff"] = f["home_cx"] - f["away_cx"]
    f["vx_diff"] = f["home_vx"] - f["away_vx"]
    f["top5_speed"] = top_speed(np.stack([px, py], -1))
    f["n_visible"] = f["home_n"] + f["away_n"]
    f["ball_minus_home_cx"] = np.where(seen, fbx, hx) - f["home_cx"]
    f["ball_minus_away_cx"] = np.where(seen, fbx, hx) - f["away_cx"]

    # time since the half started (kickoff routines, pre-kickoff dead ball)
    t_half = np.full(n, np.nan)
    for h, a, b in obs.halves:
        t_half[a:b + 1] = (np.arange(b - a + 1) - (obs.kickoffs.get(h, a) - a)) / FPS
    f["t_from_kickoff"] = np.clip(t_half, -200, 3000)

    ctx_signals = ("ball_seen", "ball_speed", "ball_line_dist", "held_line_dist", "d_diff", "d_min",
                   "contact", "top5_speed", "cx_diff", "vx_diff", "home_speed", "away_speed",
                   "n_visible", "ball_x", "n_home_3m", "n_away_3m")
    for s in ctx_signals:
        f.update(context(f[s], s))
    names = list(f)
    X = np.column_stack([np.asarray(f[k], dtype=np.float32) for k in names])
    return names, X


def sample_rows(obs):
    """5 Hz rows inside the graded halves: indices into the 25 fps arrays."""
    rows = []
    for _, a, b in obs.halves:
        rows.append(np.arange(a + STEP // 2, b + 1, STEP))
    return np.concatenate(rows)


def expand(values_5hz, rows, n, halves):
    """Back to 25 fps: each graded frame takes its nearest 5 Hz row in the same half."""
    out = np.zeros((n,) + values_5hz.shape[1:], dtype=values_5hz.dtype)
    for _, a, b in halves:
        sel = np.flatnonzero((rows >= a) & (rows <= b))
        r = rows[sel]
        frames = np.arange(a, b + 1)
        k = np.clip(np.searchsorted(r, frames - STEP // 2), 0, len(r) - 1)
        out[a:b + 1] = values_5hz[sel[k]]
    return out
