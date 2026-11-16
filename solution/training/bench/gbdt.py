"""M2 / M3: two LightGBM classifiers (alive/dead, then H/A on alive frames) at 5 Hz, combined into
P(H, A, D), decoded by argmax (M2) or a three-state Viterbi (M3).

Training data: the other six DFL matches, degraded with training seeds (never the evaluation
seed), each also as an H/A-swapped, mirrored copy so nothing team-specific can be learned.
Evaluation: the held-out match, degraded with the evaluation seed. Out-of-fold probabilities are
cached, then the decoder's three costs are chosen per held-out match on the OTHER six matches'
out-of-fold probabilities (log grid, official spellF1; the largest costs within one standard error
of the best), so the held-out labels touch nothing.

    python -m bench.gbdt                   # OOF probabilities for every match, then M2 / M3 table
    python -m bench.gbdt --final           # train on all 7, label the challenge match

Hyper-parameters are fixed in PARAMS before any result was seen (experiments/registry.csv).
"""

import argparse
import hashlib
import itertools
import json
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(os.path.dirname(ROOT))  # the repository (ROOT is solution/training/)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(REPO, "evaluation"))
import possession_metrics as pm  # noqa: E402

from bench import features as FT  # noqa: E402
from bench.harness import Bench, breakdown, knobs_id, score  # noqa: E402

CACHE = os.path.join(REPO, "cache", "team_model", "bench")
TRAIN_SEEDS = (100, 101)
EVAL_SEED = 0
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200,
              feature_fraction=0.5, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              max_bin=255, deterministic=True, force_row_wise=True, seed=0, verbose=-1,
              num_threads=8)
N_TREES = 400
GRID = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
LETTERS = np.array(["H", "A", "D"])


# -- data ---------------------------------------------------------------------------------------

def _feature_path(name, seed, swap, kid):
    return os.path.join(CACHE, "feat_{}_{}_{}_{}.npz".format(kid, name[-6:], seed, int(swap)))


def match_rows(bench, mid, seed, swap=False):
    """(names, X (rows, F), y labels at rows, rows) for one degraded match, cached on disk."""
    kid = knobs_id(bench.knobs)
    path = _feature_path(mid, seed, swap, kid)
    obs, truth = bench.get(mid, seed)
    rows = FT.sample_rows(obs)
    if os.path.isfile(path):
        with np.load(path, allow_pickle=False) as d:
            names, X = list(d["names"]), d["X"]
    else:
        names, Xf = FT.frame_features(obs, swap=swap)
        X = Xf[rows]
        os.makedirs(CACHE, exist_ok=True)
        np.savez_compressed(path, names=np.array(names), X=X)
    y = truth[rows]
    if swap:
        y = np.where(y == "H", "A", np.where(y == "A", "H", y))
    if STACK:
        pn, PX = pol_features(bench, obs, mid, seed, swap, rows)
        names, X = names + pn, np.concatenate([X, PX], 1)
    return names, X, y, rows


STACK = False  # True: Pol's labels (and their context windows) become features ("gbdt_pol")


def pol_features(bench, obs, mid, seed, swap, rows):
    """Pol's solution run on this observed match, as features: +1 H / -1 A / 0 D, and dead."""
    from bench.models import pol
    path = os.path.join(CACHE, "polfeat_{}_{}_{}_{}.npz".format(knobs_id(bench.knobs), mid[-6:],
                                                                seed, int(swap)))
    if os.path.isfile(path):
        with np.load(path, allow_pickle=False) as d:
            return list(d["names"]), d["X"]
    lab = pol(obs)
    team = np.where(lab == "H", 1.0, np.where(lab == "A", -1.0, 0.0))
    if swap:
        team = -team
    f = {"pol_team": team, "pol_dead": (lab == "D").astype(np.float64)}
    f.update(FT.context(f["pol_team"], "pol_team"))
    f.update(FT.context(f["pol_dead"], "pol_dead"))
    names = list(f)
    X = np.column_stack([np.asarray(f[k], dtype=np.float32)[rows] for k in names])
    os.makedirs(CACHE, exist_ok=True)
    np.savez_compressed(path, names=np.array(names), X=X)
    return names, X


def training_set(bench, matches):
    Xs, ys = [], []
    for m in matches:
        for s in TRAIN_SEEDS:
            for swap in (False, True):
                names, X, y, _ = match_rows(bench, m, s, swap)
                keep = y != "X"
                Xs.append(X[keep])
                ys.append(y[keep])
    X = np.concatenate(Xs).astype(np.float32, copy=False)
    del Xs
    return names, X, np.concatenate(ys)


# -- model --------------------------------------------------------------------------------------

def fit(X, y):
    import lightgbm as lgb
    dead = lgb.train(PARAMS, lgb.Dataset(X, label=(y == "D").astype(np.float32)), N_TREES)
    alive = y != "D"
    team = lgb.train(PARAMS, lgb.Dataset(X[alive], label=(y[alive] == "H").astype(np.float32)),
                     N_TREES)
    return dead, team


def proba(models, X):
    """(rows, 3) P(H), P(A), P(D)."""
    dead, team = models
    pd_ = dead.predict(X)
    ph = team.predict(X)
    return np.column_stack([(1 - pd_) * ph, (1 - pd_) * (1 - ph), pd_])


def oof(bench, refresh=False):
    """{match: (rows, P (rows, 3))} out-of-fold, cached."""
    kid = knobs_id(bench.knobs)
    tag = hashlib.sha256(json.dumps([PARAMS, N_TREES, TRAIN_SEEDS, sorted(bench.matches)],
                                    sort_keys=True).encode()).hexdigest()[:8]
    path = os.path.join(CACHE, "oof_{}_{}{}.npz".format(kid, tag, "_pol" if STACK else ""))
    if os.path.isfile(path) and not refresh:
        with np.load(path) as d:
            return {m: (d[m + "_rows"], d[m + "_p"]) for m in bench.matches}
    out = {}
    for m in bench.matches:
        fold = path.replace(".npz", "_{}.npz".format(m[-6:]))
        if os.path.isfile(fold) and not refresh:  # each fold is saved as soon as it is done
            with np.load(fold) as d:
                out[m] = (d["rows"], d["p"])
            continue
        t0 = time.time()
        _, X, y = training_set(bench, [o for o in bench.matches if o != m])
        models = fit(X, y)
        del X, y
        _, Xe, _, rows = match_rows(bench, m, EVAL_SEED)
        out[m] = (rows, proba(models, Xe))
        os.makedirs(CACHE, exist_ok=True)
        np.savez_compressed(fold, rows=rows, p=out[m][1])
        print("oof {} trained ({:.0f} s)".format(m[-6:], time.time() - t0), flush=True)
    np.savez_compressed(path, **{m + "_rows": r for m, (r, _) in out.items()},
                        **{m + "_p": p for m, (_, p) in out.items()})
    return out


# -- decoding -----------------------------------------------------------------------------------

def _viterbi_py(cost, trans):
    n = len(cost)
    acc = cost[0].copy()
    new = np.zeros(3)
    back = np.zeros((n, 3), dtype=np.int8)
    for t in range(1, n):
        for j in range(3):
            best, arg = acc[0] + trans[0, j], 0
            for i in range(1, 3):
                v = acc[i] + trans[i, j]
                if v < best:
                    best, arg = v, i
            new[j] = best + cost[t, j]
            back[t, j] = arg
        acc[:] = new
    path = np.zeros(n, dtype=np.int8)
    path[-1] = int(np.argmin(acc))
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


try:
    from numba import njit
    _viterbi = njit(cache=True)(_viterbi_py)
except ImportError:  # pragma: no cover
    _viterbi = _viterbi_py


def transitions(c_ha, c_ad, c_da):
    """3x3 switch costs, states H, A, D."""
    return np.array([[0.0, c_ha, c_ad],
                     [c_ha, 0.0, c_ad],
                     [c_da, c_da, 0.0]])


def decode(P, rows, n, halves, costs=None):
    """Labels at 25 fps from 5 Hz probabilities. costs None -> argmax (M2)."""
    eps = 1e-6
    out = np.full(n, "X", dtype="<U1")
    for _, a, b in halves:
        sel = (rows >= a) & (rows <= b)
        p = P[sel]
        if costs is None:
            path = np.argmax(p, axis=1).astype(np.int8)
        else:
            path = _viterbi(-np.log(np.clip(p, eps, 1.0)), transitions(*costs))
        lab = FT.expand(LETTERS[path][:, None], rows[sel], n, [(0, a, b)])[a:b + 1, 0]
        out[a:b + 1] = lab
    return out


def fast_spell(truth, pred):
    gt, pr = pm.label_codes(truth), pm.label_codes(pred)
    runs = pm.valid_runs(gt)
    return pm.spell_scores(gt, pm.fill_abstentions(pr, runs), runs)["mean_f1"]


class FastScorer:
    """Official spellF1 for many decodings of one match: truth codes and the 5 Hz -> 25 fps map
    computed once, the prediction kept as integer codes (possession_metrics order H, A, D, X)."""

    def __init__(self, truth, rows, halves):
        self.gt = pm.label_codes(truth)
        self.runs = pm.valid_runs(self.gt)
        self.n = len(truth)
        self.parts = []
        for _, a, b in halves:
            sel = np.flatnonzero((rows >= a) & (rows <= b))
            r = rows[sel]
            k = np.clip(np.searchsorted(r, np.arange(a, b + 1) - FT.STEP // 2), 0, len(r) - 1)
            self.parts.append((a, b, sel, k))

    def codes(self, P, costs):
        out = np.full(self.n, 3, dtype=np.int8)
        for a, b, sel, k in self.parts:
            p = P[sel]
            if costs is None:
                path = np.argmax(p, axis=1).astype(np.int8)
            else:
                path = _viterbi(-np.log(np.clip(p, 1e-6, 1.0)), transitions(*costs))
            out[a:b + 1] = path[k]
        return out

    def spell(self, P, costs):
        return pm.spell_scores(self.gt, self.codes(P, costs), self.runs)["mean_f1"]

    def spell_path(self, P, path_fn):
        """spellF1 of path_fn(cost (rows, 3)) -> 5 Hz states, per half."""
        out = np.full(self.n, 3, dtype=np.int8)
        for a, b, sel, k in self.parts:
            out[a:b + 1] = path_fn(-np.log(np.clip(P[sel], 1e-6, 1.0)))[k]
        return pm.spell_scores(self.gt, out, self.runs)["mean_f1"]


COARSE = (1.0, 4.0, 16.0)


def choose_costs(bench, probs, matches, verbose=False):
    """Costs maximising mean spellF1 over `matches` (their OOF probabilities); the largest costs
    (by sum of logs) within one standard error of the best. Coarse grid {1, 4, 16}^3, then every
    neighbour of the best on the full GRID (factor 2 steps)."""
    scorers = []
    for m in matches:
        obs, truth = bench.get(m, EVAL_SEED)
        rows, P = probs[m]
        scorers.append((FastScorer(truth, rows, obs.halves), P))
    seen = {}

    def evaluate(c):
        if c not in seen:
            seen[c] = np.array([sc.spell(P, c) for sc, P in scorers])
        return seen[c].mean()

    for c in itertools.product(COARSE, COARSE, COARSE):
        evaluate(c)
    for _ in range(3):  # refine around the current best
        best = max(seen, key=lambda c: seen[c].mean())
        idx = [GRID.index(v) for v in best]
        for d in itertools.product((-1, 0, 1), repeat=3):
            j = [i + e for i, e in zip(idx, d)]
            if all(0 <= x < len(GRID) for x in j):
                evaluate(tuple(GRID[x] for x in j))
    combos = list(seen)
    table = np.array([seen[c] for c in combos])
    mean = table.mean(1)
    se = table.std(1, ddof=1) / np.sqrt(len(matches))
    best = int(np.argmax(mean))
    ok = np.flatnonzero(mean >= mean[best] - se[best])
    pick = ok[np.argmax([np.sum(np.log(combos[i])) for i in ok])]
    if verbose:
        print("  best {} {:.3f}; picked {} {:.3f}".format(combos[best], mean[best], combos[pick],
                                                          mean[pick]))
    return combos[pick], table, combos


def bench_rows(bench, probs, cost_cache=None, prefix="gbdt", extra=True):
    """Harness-style rows: m2 argmax, m3 Viterbi (costs chosen without the held-out match), and
    with extra=True m4 HSMM and m5 Viterbi at 25 fps (bench/hsmm.py), same rule for their knobs."""
    from bench import hsmm
    rows_out = []
    cost_cache = {} if cost_cache is None else cost_cache
    for m in bench.matches:
        obs, truth = bench.get(m, EVAL_SEED)
        rows, P = probs[m]
        n = len(truth)
        others = [o for o in bench.matches if o != m]
        if m not in cost_cache:
            cost_cache[m] = choose_costs(bench, probs, others)[0]
        costs = cost_cache[m]
        variants = [("m2", lambda: decode(P, rows, n, obs.halves, None), ""),
                    ("m3", lambda: decode(P, rows, n, obs.halves, costs), str(costs))]
        if extra:
            # durations from the training matches' labels only
            durations = hsmm.fit_durations([bench.clean(o)["labels"] for o in others])
            w = choose_hsmm_weight(bench, probs, others, costs, durations)
            variants.append(("m4", lambda: hsmm.decode_hsmm(P, rows, n, obs.halves, costs,
                                                            durations, w), "{} w={}".format(costs, w)))
            variants.append(("m5", lambda: hsmm.decode_fine(P, rows, n, obs.halves, costs),
                             str(costs)))
        for tag, fn, info in variants:
            pred = fn()
            name = "{}_{}".format(prefix, tag)
            r = {"model": name, "match": m, "seed": EVAL_SEED, "condition": "degraded",
                 "costs": info}
            r.update(score(truth, pred))
            r.update(breakdown(truth, pred, obs))
            rows_out.append(r)
            print("{:<10} {} spellF1 {:.3f} macroF1 {:.3f} {}".format(
                name, m[-6:], r["spell_f1_mean"], r["macro_f1"], info), flush=True)
    return pd.DataFrame(rows_out)


HSMM_WEIGHTS = (0.0, 0.25, 0.5, 1.0, 2.0)


def choose_hsmm_weight(bench, probs, matches, costs, durations):
    """Duration weight maximising mean spellF1 on `matches`; the smallest within one SE."""
    from bench import hsmm
    (pm_, ps_), (dm_, ds_) = durations
    dp, dd = hsmm.duration_table(pm_, ps_), hsmm.duration_table(dm_, ds_)
    trans = transitions(*costs)
    table = np.zeros((len(HSMM_WEIGHTS), len(matches)))
    for j, m in enumerate(matches):
        obs, truth = bench.get(m, EVAL_SEED)
        rows, P = probs[m]
        sc = FastScorer(truth, rows, obs.halves)
        for i, w in enumerate(HSMM_WEIGHTS):
            dur = w * np.stack([dp, dp, dd])
            table[i, j] = sc.spell_path(P, lambda c: hsmm._hsmm(c, dur, trans, hsmm.MAX_SEG))
    mean = table.mean(1)
    se = table.std(1, ddof=1) / np.sqrt(len(matches))
    best = int(np.argmax(mean))
    ok = np.flatnonzero(mean >= mean[best] - se[best])
    return HSMM_WEIGHTS[int(ok[0])]


def final(bench, out_csv):
    """Train on all seven, pick costs on all seven OOF, label the challenge match."""
    from bench import obs as obs_mod
    probs = oof(bench)
    costs, _, _ = choose_costs(bench, probs, bench.matches, verbose=True)
    _, X, y = training_set(bench, bench.matches)
    models = fit(X, y)
    o = obs_mod.challenge()
    names, Xf = FT.frame_features(o)
    rows = FT.sample_rows(o)
    Xr = Xf[rows]
    if STACK:
        from bench.models import pol
        lab = pol(o)
        team = np.where(lab == "H", 1.0, np.where(lab == "A", -1.0, 0.0))
        f = {"pol_team": team, "pol_dead": (lab == "D").astype(np.float64)}
        f.update(FT.context(f["pol_team"], "pol_team"))
        f.update(FT.context(f["pol_dead"], "pol_dead"))
        Xr = np.concatenate([Xr, np.column_stack([np.asarray(f[k], dtype=np.float32)[rows]
                                                  for k in f])], 1)
    P = proba(models, Xr)
    labels = decode(P, rows, len(o.track["ball"]), o.halves, costs)
    from starter.baseline import write_submission
    write_submission(out_csv, labels)
    np.savez_compressed(out_csv.replace(".csv", "_proba.npz"), rows=rows, P=P)
    print("costs {} -> {}".format(costs, out_csv))
    return labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", default="")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--stack", action="store_true", help="Pol's labels as extra features")
    args = ap.parse_args()
    global STACK
    STACK = args.stack
    prefix = "gbdtpol" if STACK else "gbdt"
    bench = Bench()
    if args.final:
        final(bench, args.final)
        return
    probs = oof(bench, args.refresh)
    df = bench_rows(bench, probs, prefix=prefix)
    path = os.path.join(ROOT, "bench", "results", "{}_{}.csv".format(prefix, knobs_id(bench.knobs)))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(df.groupby("model")[["spell_f1_mean", "macro_f1"]].agg(["mean", "std"]).round(3))
    print("->", os.path.relpath(path, ROOT))


if __name__ == "__main__":
    main()


# -- the law of the Laws: a ball that stays outside the lines is out of play ----------------------

LAW_MARGIN = 1.0       # metres beyond a line
LAW_LONG = 50          # frames: an outside run this long is a ball out of play
LAW_SLOW = 3.0         # m/s: a slow ball outside for LAW_SLOW_FRAMES is out of play too
LAW_SLOW_FRAMES = 5


def law_dead(obs):
    """25 fps mask: ball seen beyond a line for >= LAW_LONG frames, or slow there for
    >= LAW_SLOW_FRAMES (v4's rule; fast short outside runs are lofted balls and stay alive)."""
    from bench.stats import run_lengths  # noqa: F401
    L, W = obs.pitch
    B = obs.track["ball"]
    bx, by = B[:, 0].astype(np.float64) * L, B[:, 1].astype(np.float64) * W
    with np.errstate(invalid="ignore"):
        out = (bx < -LAW_MARGIN) | (bx > L + LAW_MARGIN) | (by < -LAW_MARGIN) | (by > W + LAW_MARGIN)
        slow = np.hypot(np.gradient(bx), np.gradient(by)) * 25 < LAW_SLOW
    mask = np.zeros(len(B), dtype=bool)
    x = np.concatenate([[0], out.astype(np.int8), [0]])
    d = np.diff(x)
    for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
        if b - a >= LAW_LONG or np.sum(slow[a:b]) >= LAW_SLOW_FRAMES:
            mask[a:b] = True
    return mask


def apply_law(P, rows, obs, strength=0.999):
    """P with rows inside a law-dead run forced towards D."""
    m = law_dead(obs)[rows]
    Q = P.copy()
    Q[m] = [(1 - strength) / 2, (1 - strength) / 2, strength]
    return Q
