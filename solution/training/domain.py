"""Does the degraded DFL data look like our tracking? A classifier is asked to tell them apart.

    python solution/training/domain.py

Rows of our match's features (live frames by our own dead-ball step; no labels involved) against
rows of the bench's test matches (frames in play). An AUC near 0.5 means the model cannot tell where
a row comes from; the features that give it away show what the degradation still gets wrong.
"""

import os
import re
import sys

import numpy as np

import core
import dfl

sys.path.insert(0, os.path.join(core.ROOT, "starter"))


def ours():
    """Feature rows of our match on the frames our pipeline calls live."""
    import run          # the solution's run.py
    import tracking
    match = tracking.load()
    dead, _ = run.dead_ball(match)
    obs = run.as_obs(match)
    names, X = core.FT.frame_features(obs)
    rows = core.FT.sample_rows(obs)
    rows = rows[~dead[rows]]
    return names, X[rows].astype(np.float32)


def base(name):
    return re.sub(r"_(mean|min|max)_[pf][0-9.]+$", "", name)


if __name__ == '__main__':
    import lightgbm as lgb
    names, A = ours()
    B = np.vstack([t["X"][t["alive"]] for t in (core.table(n, core.TEST_SEED) for n in dfl.MATCHES)])
    rng = np.random.RandomState(0)
    B = B[rng.choice(len(B), size=min(len(B), 4 * len(A)), replace=False)]
    # time-in-half and the absolute clock are not about the tracking: leave them out
    keep = [i for i, n in enumerate(names) if base(n) != "t_from_kickoff"]
    X = np.vstack([A, B])[:, keep]
    y = np.concatenate([np.ones(len(A)), np.zeros(len(B))])
    # blocked folds: consecutive rows are near-duplicates, so split by position, not at random
    fold = np.concatenate([np.arange(len(A)) * 5 // len(A), np.arange(len(B)) * 5 // len(B)])
    aucs, gain = [], np.zeros(len(keep))
    for k in range(5):
        tr, te = fold != k, fold == k
        model = lgb.train(dict(objective="binary", learning_rate=0.1, num_leaves=15, min_data_in_leaf=200,
                               feature_fraction=0.5, verbosity=-1, num_threads=16, seed=0),
                          lgb.Dataset(X[tr], label=y[tr]), num_boost_round=60)
        p = model.predict(X[te])
        order = np.argsort(p)
        ranks = np.empty(len(p))
        ranks[order] = np.arange(1, len(p) + 1)
        pos = y[te] == 1
        aucs.append((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2.0) / (pos.sum() * (~pos).sum()))
        gain += model.feature_importance("gain")
    print("rows: ours {}  bench {}   AUC (5 blocked folds): {:.3f}  ({})".format(
        len(A), len(B), np.mean(aucs), " ".join("{:.3f}".format(a) for a in aucs)))
    by = {}
    for i, g in zip(keep, gain):
        by[base(names[i])] = by.get(base(names[i]), 0.0) + g
    total = sum(by.values())
    print("what gives it away (share of the classifier's gain, by signal):")
    for k, v in sorted(by.items(), key=lambda kv: -kv[1])[:12]:
        j = [i for i, n in enumerate(names) if n == k]
        a = np.nanmean(A[:, j[0]]) if j else float("nan")
        b = np.nanmean(B[:, j[0]]) if j else float("nan")
        print("  {:<22} {:.3f}   mean ours {:8.3f}  bench {:8.3f}".format(k, v / total, a, b))
