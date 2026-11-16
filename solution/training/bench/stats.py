"""Label-free statistics of an observed match: the numbers the degrader is calibrated to.

Every statistic here is computed from tracking alone, so it can be measured on the real challenge
match (no labels used) and on a degraded DFL match, and the two compared. That is the whole
calibration: the degrader is tuned until a degraded DFL match looks, through these numbers, like
our ATD. No ground truth of the challenge match is involved anywhere.
"""

import numpy as np

from .obs import graded_mask

SPEED_BINS = (0.0, 1.5, 2.5, 3.5, 5.0, 8.0, 99.0)
TOP_K = 5


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


def run_lengths(mask):
    x = np.concatenate([[0], mask.astype(np.int8), [0]])
    d = np.diff(x)
    return np.flatnonzero(d == -1) - np.flatnonzero(d == 1)


def metric_arrays(obs):
    L, W = obs.pitch
    P = obs.track["players"]
    pm = np.stack([P[:, :, 0] * L, P[:, :, 1] * W], axis=-1)
    B = obs.track["ball"]
    return pm, B[:, 0] * L, B[:, 1] * W


def describe(obs):
    """Dict of label-free statistics over the graded frames."""
    g = graded_mask(obs)
    pm, bx, by = metric_arrays(obs)
    L, W = obs.pitch
    seen = ~np.isnan(bx)
    out = {}
    out["ball_missing"] = float(np.mean(~seen[g]))
    top = top_speed(pm)
    for lo, hi in zip(SPEED_BINS[:-1], SPEED_BINS[1:]):
        m = g & (top >= lo) & (top < hi)
        out["share_speed_{}-{}".format(lo, hi)] = float(m.sum() / g.sum())
        out["missing_speed_{}-{}".format(lo, hi)] = float(np.mean(~seen[m])) if m.any() else np.nan
    gaps = np.concatenate([run_lengths(~seen[a:b + 1]) for _, a, b in obs.halves])
    shows = np.concatenate([run_lengths(seen[a:b + 1]) for _, a, b in obs.halves])
    for q in (10, 25, 50, 75, 90):
        out["gap_len_p{}".format(q)] = float(np.percentile(gaps, q)) if len(gaps) else np.nan
        out["seen_len_p{}".format(q)] = float(np.percentile(shows, q)) if len(shows) else np.nan
    out["gaps_per_min"] = float(len(gaps) / (g.sum() / 25 / 60))
    s = seen & g
    with np.errstate(invalid="ignore"):
        out["ball_out1_far_touch"] = float(np.sum(s & (by > W + 1)) / s.sum())
        out["ball_out1_near_touch"] = float(np.sum(s & (by < -1)) / s.sum())
        out["ball_out1_goal"] = float(np.sum(s & ((bx < -1) | (bx > L + 1))) / s.sum())
        out["ball_out5"] = float(np.sum(s & ((bx < -5) | (bx > L + 5) | (by < -5) | (by > W + 5)))
                                 / s.sum())
    with np.errstate(invalid="ignore"):
        outm = s & ((bx < -1) | (bx > L + 1) | (by < -1) | (by > W + 1))
    rl = run_lengths(outm)
    out["out_run_p50"] = float(np.median(rl)) if len(rl) else 0.0
    out["out_runs_per_min"] = float(len(rl) / (g.sum() / 25 / 60))
    out["out_frac_in_long_runs"] = float(rl[rl >= 50].sum() / max(rl.sum(), 1))
    bsp = np.hypot(np.gradient(bx), np.gradient(by)) * 25
    out["out_speed_p50"] = float(np.nanmedian(bsp[outm])) if outm.any() else 0.0
    step = np.hypot(np.diff(bx), np.diff(by))
    ok = s[1:] & s[:-1]
    out["ball_step_p50"], out["ball_step_p99"] = (float(v) for v in np.percentile(step[ok], [50, 99]))
    vis = (~np.isnan(pm[:, :, 0]))[g].sum(1)
    for q in (1, 5, 25, 50, 95):
        out["visible_p{}".format(q)] = float(np.percentile(vis, q))
    segs = []
    v = ~np.isnan(pm[:, :, 0])
    for j in range(v.shape[1]):
        segs.extend(run_lengths(v[g, j]))
    segs = np.array(segs)
    for q in (10, 50, 90):
        out["slot_seg_p{}".format(q)] = float(np.percentile(segs, q))
    out["slot_segs_per_min"] = float(len(segs) / (g.sum() / 25 / 60))
    out["top_speed_mean"] = float(np.nanmean(top[g]))
    return out


def compare(rows):
    """Print {name: describe(obs)} side by side."""
    names = list(rows)
    keys = list(rows[names[0]])
    print("{:<26}".format("stat") + "".join("{:>14}".format(n[:13]) for n in names))
    for k in keys:
        print("{:<26}".format(k) + "".join("{:>14.3f}".format(rows[n][k]) for n in names))
