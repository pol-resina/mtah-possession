"""Turn a clean DFL match into one that looks like our automatic ATD.

The clean DFL tracking is corrected: every player and the ball on every frame, ball height known.
Our ATD is automatic: the ball is lost for long stretches (mostly in stoppages), lofted balls are
projected onto the grass from the camera, players drop out and their tracks fragment into anonymous
slots, and a slot sometimes carries the wrong team (the two goalkeepers swap tracks). Each of those
is simulated here, with its own knob, so that a model can be trained and scored on data that fails
the way ours fails, while the truth is known.

Every knob is either set from label-free statistics of our ATD (bench/stats.py; see
bench/calibrate.py) or swept for sensitivity. None comes from the challenge's ground truth.

    obs, labels = degrade(clean_match, seed=0, ball_missing=0.41)

Same input, same seed, same knobs -> same output, byte for byte.
"""

from dataclasses import dataclass, field, replace

import numpy as np
from scipy.signal import lfilter
from scipy.special import ndtri

from . import obs as obs_mod
from .stats import SPEED_BINS, top_speed

# P(ball missing | five fastest players' speed bin) on our ATD (bench/stats.py on BAR-ATH, label-free).
ATD_MISSING_BY_SPEED = (0.846, 0.625, 0.248, 0.123, 0.117, 0.093)


@dataclass(frozen=True)
class Knobs:
    # ball missing
    ball_missing: float = 0.41          # target share of graded frames with no ball
    missing_by_speed: tuple = ATD_MISSING_BY_SPEED
    speed_gain: float = 1.0             # scales the speed-bin probits (the AR noise flattens them)
    missing_tau_s: float = 6.0          # correlation time of the latent loss process, seconds
    lofted_z: float = 2.5               # metres; above this the ball is "high"
    lofted_weight: float = 1.0          # extra probit for a high ball
    congestion_weight: float = 0.3      # extra probit per player within 2 m of the ball beyond 1
    out_weight: float = 1.5             # extra probit for a ball more than 1 m beyond a line
    out_episode_visible: float = -1.0   # >= 0: each true out-of-play episode is seen whole with
                                        # this probability, else lost whole (overrides the latent)
    dead_coupling: float = 0.0          # extra probit when the ball is truly dead (sensitivity)
    min_gap: int = 100                  # frames; shorter losses are interpolated by the tracker
    min_seen: int = 25                  # frames; shorter sightings are dropped
    # ball position
    camera: tuple = (None, -45.0, 30.0)  # x (None = halfway), y, height of the tactical camera, m
    ball_noise: float = 0.25            # metres, low-passed
    ball_smooth: int = 5                # frames of centred moving average (the ATD ball is smooth)
    false_per_min: float = 0.3          # episodes of the ball "found" on a player while lost
    false_len: tuple = (10, 60)
    # players
    player_noise: float = 0.05          # metres, low-passed over ~2 s (the ATD players are smooth)
    drop_per_min: float = 0.6           # per player: rate of starting a dropout
    drop_len_s: float = 3.0             # mean dropout length
    crowd_weight: float = 2.0           # dropout rate multiplier per extra player within 1.5 m
    blackout_per_min: float = 0.25      # episodes when many players are lost at once
    blackout_len: tuple = (25, 250)
    blackout_share: tuple = (0.3, 0.9)
    break_per_min: float = 0.35         # per player: identity switches (new slot) while tracked
    team_error: float = 0.01            # share of new track segments put in the other team's slots
    keeper_swap: float = 0.3            # share of goalkeeper segments put in the other team's slots
    slots: tuple = (24, 22)             # slot pool per team, as in the ATD metadata


def load_knobs(path=None):
    """Knobs fitted by bench/calibrate.py (cache/degrade_knobs.json), or the defaults."""
    import json
    import os
    path = path or os.path.join(os.path.dirname(os.path.abspath(__file__)), "degrade_knobs.json")
    if not os.path.isfile(path):
        return Knobs()
    raw = json.load(open(path))
    return Knobs(**{k: tuple(v) if isinstance(v, list) else v for k, v in raw.items()})


def _ar1(n, tau_frames, rng):
    rho = np.exp(-1.0 / max(tau_frames, 1e-6))
    eps = rng.standard_normal(n) * np.sqrt(1 - rho ** 2)
    out = lfilter([1.0], [1.0, -rho], eps)
    return out


def _lowpass_noise(shape, sigma, rng, tau=10):
    """Gaussian noise with std sigma, AR(1) along axis 0 (a tracker's smooth wobble)."""
    rho = np.exp(-1.0 / tau)
    eps = rng.standard_normal(shape) * np.sqrt(1 - rho ** 2) * sigma
    return lfilter([1.0], [1.0, -rho], eps, axis=0)


def _runs(mask):
    x = np.concatenate([[0], mask.astype(np.int8), [0]])
    d = np.diff(x)
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def _clean_runs(missing, min_gap, min_seen):
    """Drop short sightings (-> lost), then short losses (-> seen, interpolated)."""
    m = missing.copy()
    for a, b in _runs(~m):
        if b - a < min_seen and a > 0 and b < len(m):
            m[a:b] = True
    for a, b in _runs(m):
        if b - a < min_gap and a > 0 and b < len(m):
            m[a:b] = False
    return m


def _bin_probit(top, table):
    p = np.full(len(top), table[2])
    for k, (lo, hi) in enumerate(zip(SPEED_BINS[:-1], SPEED_BINS[1:])):
        p[(top >= lo) & (top < hi)] = table[k]
    return ndtri(np.clip(p, 1e-3, 1 - 1e-3))


def _beyond_lines(ball, pitch, margin):
    L, W = pitch
    return ((ball[:, 0] < -margin) | (ball[:, 0] > L + margin)
            | (ball[:, 1] < -margin) | (ball[:, 1] > W + margin))


def ball_missing_mask(clean, knobs, rng, top):
    """Which frames lose the ball. Calibrated to knobs.ball_missing over the graded frames."""
    ball, players = clean["ball"], clean["players"]
    n = len(ball)
    d = np.hypot(players[:, :, 0] - ball[:, None, 0], players[:, :, 1] - ball[:, None, 1])
    crowd = np.maximum(np.sum(d < 2.0, axis=1) - 1, 0)
    base = (knobs.speed_gain * _bin_probit(top, knobs.missing_by_speed)
            + knobs.lofted_weight * (ball[:, 2] > knobs.lofted_z)
            + knobs.congestion_weight * crowd
            + knobs.out_weight * _beyond_lines(ball, clean["pitch"], 1.0)
            + knobs.dead_coupling * (clean["labels"] == "D"))
    noise = np.zeros(n)
    for _, a, b in clean["halves"]:
        noise[a:b + 1] = _ar1(b - a + 1, knobs.missing_tau_s * 25, rng)

    def make(offset):
        m = np.zeros(n, dtype=bool)
        for _, a, b in clean["halves"]:
            m[a:b + 1] = _clean_runs(base[a:b + 1] + offset + noise[a:b + 1] > 0,
                                     knobs.min_gap, knobs.min_seen)
        return m

    if knobs.out_episode_visible >= 0:
        # whole episodes: the ball beyond a line, gaps under 1 s bridged
        beyond = _beyond_lines(ball, clean["pitch"], 0.5)
        ep = beyond.copy()
        for a, b in _runs(~beyond):
            if b - a < 25 and a > 0 and b < n:
                ep[a:b] = True
        force_seen = np.zeros(n, dtype=bool)
        force_lost = np.zeros(n, dtype=bool)
        for a, b in _runs(ep):
            if rng.random() < knobs.out_episode_visible:
                force_seen[a:b] = True
            else:
                force_lost[a:b] = True
        base = base.copy()
        base[force_seen] = -50.0
        base[force_lost] = 50.0

    lo, hi = -6.0, 6.0
    for _ in range(30):
        mid = (lo + hi) / 2
        if make(mid).mean() < knobs.ball_missing:
            lo = mid
        else:
            hi = mid
    return make((lo + hi) / 2)


def project(ball, pitch, camera):
    """Where a 2-D detector puts a ball at height z: the ray from the camera through it, on the
    grass. A lofted ball lands beyond its true spot, away from the camera."""
    L, W = pitch
    cx = L / 2.0 if camera[0] is None else camera[0]
    cy, h = camera[1], camera[2]
    z = np.clip(ball[:, 2], 0.0, 0.8 * h)
    k = h / (h - z)
    return np.column_stack([cx + (ball[:, 0] - cx) * k, cy + (ball[:, 1] - cy) * k])


def _interpolate_gaps(xy, filled_mask):
    """Linear interpolation over frames in filled_mask from their seen neighbours."""
    out = xy.copy()
    idx = np.arange(len(xy))
    good = ~filled_mask & ~np.isnan(xy[:, 0])
    for c in range(2):
        out[filled_mask, c] = np.interp(idx[filled_mask], idx[good], xy[good, c])
    return out


def degrade_ball(clean, knobs, rng, top):
    ball = clean["ball"].astype(np.float64)
    n = len(ball)
    missing = ball_missing_mask(clean, knobs, rng, top)
    xy = project(ball, clean["pitch"], knobs.camera)
    xy = xy + _lowpass_noise((n, 2), knobs.ball_noise, rng)
    # Short losses the tracker bridges: the raw detector would have missed some frames there; the
    # tracker draws a straight line over them. Simulate the misses at the high-ball frames.
    bridged = (ball[:, 2] > knobs.lofted_z) & ~missing
    xy = _interpolate_gaps(xy, bridged)
    if knobs.ball_smooth > 1:
        k = np.ones(knobs.ball_smooth) / knobs.ball_smooth
        pad = knobs.ball_smooth // 2
        for c in range(2):
            xy[:, c] = np.convolve(np.pad(xy[:, c], pad, mode="edge"), k, mode="valid")
    out = np.where(missing[:, None], np.nan, xy)
    # False sightings: while lost, the detector locks on a player for a moment.
    graded_minutes = sum(b - a + 1 for _, a, b in clean["halves"]) / 25.0 / 60.0
    n_false = rng.poisson(knobs.false_per_min * graded_minutes)
    lost = np.flatnonzero(missing)
    players = clean["players"]
    for _ in range(n_false if len(lost) else 0):
        t0 = int(rng.choice(lost))
        length = int(rng.integers(*knobs.false_len))
        t1 = min(n, t0 + length)
        here = np.flatnonzero(~np.isnan(players[t0, :, 0]))
        if not len(here):
            continue
        j = int(rng.choice(here))
        seg = slice(t0, t1)
        ok = missing[seg] & ~np.isnan(players[seg, j, 0])
        out[seg][ok] = players[seg, j][ok] + rng.normal(0, 0.3, (ok.sum(), 2))
    return out, missing


def degrade_players(clean, knobs, rng):
    """(players (n, S, 2) metres in anonymous slots, team per slot)."""
    P = clean["players"].astype(np.float64)
    n, n_p = P.shape[:2]
    team, keeper = clean["team"], clean["keeper"]
    on = ~np.isnan(P[:, :, 0])
    P = P + _lowpass_noise(P.shape, knobs.player_noise, rng, tau=50)

    # crowding: players within 1.5 m of this one
    crowd = np.zeros((n, n_p))
    step = 5  # computed every 5 frames and held, enough for a rate
    for t in range(0, n, step):
        x = P[t]
        dd = np.hypot(x[:, None, 0] - x[None, :, 0], x[:, None, 1] - x[None, :, 1])
        crowd[t:t + step] = np.nansum(dd < 1.5, axis=1) - 1
    rate = knobs.drop_per_min / (25 * 60) * (1 + knobs.crowd_weight * np.maximum(crowd, 0))
    starts = rng.random((n, n_p)) < rate
    lost = np.zeros((n, n_p), dtype=bool)
    mean_len = knobs.drop_len_s * 25
    for t, j in zip(*np.nonzero(starts)):
        lost[t:t + 1 + int(rng.exponential(mean_len)), j] = True
    minutes = n / 25.0 / 60.0
    for _ in range(rng.poisson(knobs.blackout_per_min * minutes)):
        t0 = int(rng.integers(0, n))
        t1 = min(n, t0 + int(rng.integers(*knobs.blackout_len)))
        who = rng.random(n_p) < rng.uniform(*knobs.blackout_share)
        lost[t0:t1, who] = True
    visible = on & ~lost

    # Split each player's visible time into track segments: a new segment after every loss, and
    # spontaneous identity breaks while tracked.
    brk = rng.random((n, n_p)) < knobs.break_per_min / (25 * 60)
    segments = []  # (start, stop, person, slot team)
    for j in range(n_p):
        for a, b in _runs(visible[:, j]):
            cuts = [a] + [int(t) for t in np.flatnonzero(brk[a:b, j]) + a if t > a] + [b]
            for s0, s1 in zip(cuts[:-1], cuts[1:]):
                slot_team = int(team[j])
                p_wrong = knobs.keeper_swap if keeper[j] else knobs.team_error
                if rng.random() < p_wrong:
                    slot_team = 1 - slot_team
                segments.append((s0, s1, j, slot_team))
    segments.sort()

    pools = [list(range(knobs.slots[0])),
             list(range(knobs.slots[0], knobs.slots[0] + knobs.slots[1]))]
    slot_team = [0] * knobs.slots[0] + [1] * knobs.slots[1]
    free_at = {s: 0 for s in range(sum(knobs.slots))}
    out = [np.full((n, 2), np.nan) for _ in range(sum(knobs.slots))]
    for s0, s1, j, tm in segments:
        free = [s for s in pools[tm] if free_at[s] <= s0]
        if not free:  # pool exhausted: open a new slot for this team
            s = len(out)
            out.append(np.full((n, 2), np.nan))
            pools[tm].append(s)
            slot_team.append(tm)
            free_at[s] = 0
            free = [s]
        s = min(free, key=lambda q: (free_at[q], q))  # the slot idle longest, as a tracker reuses ids
        out[s][s0:s1] = P[s0:s1, j]
        free_at[s] = s1
    return np.stack(out, axis=1), np.array(slot_team)


def kickoffs(clean):
    """First alive frame of each half: the challenge gives these as metadata."""
    out = {}
    for h, a, b in clean["halves"]:
        alive = np.flatnonzero(clean["labels"][a:b + 1] != "D")
        out[int(h)] = int(a + alive[0]) if len(alive) else int(a)
    return out


def degrade(clean, seed=0, knobs=None, **overrides):
    """(Obs, labels) for a clean DFL match. overrides replace single knobs, e.g. ball_missing=0.3."""
    knobs = replace(knobs or Knobs(), **overrides)
    mid = str(clean["match_id"])  # seed from a stable hash (Python's str hash is salted)
    rng = np.random.default_rng([seed, sum(ord(c) * 31 ** i for i, c in enumerate(mid)) % (2 ** 31)])
    top = top_speed(clean["players"])
    ball, missing = degrade_ball(clean, knobs, rng, top)
    players, team = degrade_players(clean, knobs, rng)
    ar = {1: bool(clean["attacks_right"][0]), 2: bool(clean["attacks_right"][1])}
    o = obs_mod.make(players, ball, team, clean["pitch"], clean["halves"], ar,
                     kickoffs=kickoffs(clean), name=mid)
    return o, clean["labels"].copy()


def clean_obs(clean):
    """The clean match as an Obs (no degradation): the ceiling every model is also scored on."""
    ar = {1: bool(clean["attacks_right"][0]), 2: bool(clean["attacks_right"][1])}
    return obs_mod.make(clean["players"], clean["ball"][:, :2], clean["team"], clean["pitch"],
                        clean["halves"], ar, kickoffs=kickoffs(clean), name=str(clean["match_id"]))
