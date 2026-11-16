"""Shared, memory-lean helpers for team-model experiments on Adrià's DFL bench (leave-one-match-out,
team only: the true dead ball, our decoder with switch cost 1.0)."""
import os, sys, time
from types import SimpleNamespace
import numpy as np
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
OUT = os.path.join(REPO, "cache", "team_model", "network")  # out-of-fold probabilities and results
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, os.path.join(REPO, "solution", "training"))
sys.path.insert(0, os.path.join(REPO, "solution"))
from bench import gbdt, degrade                      # noqa: E402
from bench import features as BFT                    # noqa: E402
from bench.harness import Bench, knobs_id            # noqa: E402
import possession_metrics as pm                      # noqa: E402
from segments import runs, two_state                 # noqa: E402

SWITCH = 1.0
ROW_FRAMES = 5
BENCH = Bench(knobs=degrade.load_knobs())
KID = knobs_id(BENCH.knobs)
MATCHES = list(BENCH.matches)


def meta(m):
    """(rows, labels at rows, halves, all frame labels) from the clean match; no degradation."""
    c = BENCH.clean(m)
    halves = [tuple(int(v) for v in h) for h in c["halves"]]
    rows = BFT.sample_rows(SimpleNamespace(halves=halves))
    return rows, c["labels"][rows], halves, c["labels"]


def table(m, seed, swap):
    """(X float32 (rows, F), y labels at rows) for one degraded copy, from the feature cache."""
    path = gbdt._feature_path(m, seed, swap, KID)
    if not os.path.isfile(path):
        gbdt.match_rows(BENCH, m, seed, swap)   # degrades and caches the table
        BENCH._obs.clear()
    with np.load(path, allow_pickle=False) as d:
        X = d["X"].astype(np.float32, copy=False)
    _, y, _, _ = meta(m)
    if swap:
        y = np.where(y == "H", "A", np.where(y == "A", "H", y))
    return X, y


def decode_team(p_rows, m):
    """Our decoder: true dead ball, two-state Viterbi on P(home) inside every live stretch."""
    rows, _, halves, labels = meta(m)
    n = len(labels)
    p = np.clip(BFT.expand(np.asarray(p_rows, dtype=np.float64), rows, n, halves), 1e-6, 1 - 1e-6)
    ch, ca = -np.log(p) / ROW_FRAMES, -np.log(1 - p) / ROW_FRAMES
    dead = labels == "D"
    out = np.full(n, "X", dtype="<U1")
    for _, first, last in halves:
        out[first:last + 1] = "D"
        for a, b in runs(~dead, first, last + 1):
            st = two_state(ch[a:b], ca[a:b], SWITCH, SWITCH)
            out[a:b] = np.where(st == 0, "H", "A")
    return out


def score(pred, m):
    _, _, _, labels = meta(m)
    r = pm.evaluate(pm.label_codes(labels), pm.label_codes(pred), bootstrap_n=0)
    return {"spell": r["spell_f1_mean"], "macro": r["macro_f1"], "team_acc": r["team_acc_gt_alive"],
            "n_pred": r["n_pred_spells"], "n_gt": r["n_gt_spells"]}
