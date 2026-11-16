"""Score models on the DFL bench: one row per (model, match), the official metric unchanged.

    python -m bench.harness --models baseline_smooth50,pol,v1,v3,v4
    python -m bench.harness --models pol,gbdt_m3 --missing 0.2,0.41,0.6 --seeds 0,1

Each match is degraded (bench/degrade.py, knobs from bench/calibrate.py) and every model predicts
it. Learned models are trained leave-one-match-out: the held-out match is never seen in training.
Scoring is evaluation/possession_metrics.evaluate, untouched. Results go to bench/results/.

Acceptance rule (the plan): a change is accepted when its spellF1 gains more than 0.04 on average
AND it wins on all 7 matches (one-sided sign test p = 1/128 ~ 0.008; 6 of 7 gives 0.0625).
"""

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import asdict

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(os.path.dirname(ROOT))  # the repository (ROOT is solution/training/)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(REPO, "evaluation"))
import possession_metrics as pm  # noqa: E402

from bench import degrade, dfl  # noqa: E402
from bench.obs import graded_mask  # noqa: E402

RESULTS = os.path.join(REPO, "cache", "team_model", "results")
SPELL_BINS = (0, 2, 5, 15, 1e9)  # seconds


def score(truth, pred):
    """The official numbers, plus a few for the error breakdown."""
    gt, pr = pm.label_codes(truth), pm.label_codes(pred)
    r = pm.evaluate(gt, pr)
    out = {k: r[k] for k in ("macro_f1", "spell_f1_mean", "n_gt_spells", "n_pred_spells",
                             "alive_dead_f1", "team_acc_gt_alive", "team_acc_both_alive",
                             "mean_spell_duration_s", "gt_mean_spell_duration_s")}
    for c in "HAD":
        out["f1_" + c] = r["per_class"][c]["f1"]
    out["trans_f1_25"] = r["transitions"][25]["f1"]
    return out


def breakdown(truth, pred, obs):
    """Where the errors are: dead ball, team with/without the ball, short possessions."""
    g = graded_mask(obs) & (truth != "X")
    seen = ~np.isnan(obs.track["ball"][:, 0])
    t, p = truth[g], pred[g]
    alive_t, alive_p = t != "D", p != "D"
    both = alive_t & alive_p
    s = seen[g]
    out = {
        "missed_dead": float(np.mean(~alive_t & alive_p)),       # share of graded frames
        "false_dead": float(np.mean(alive_t & ~alive_p)),
        "team_err_seen": float(np.mean(t[both & s] != p[both & s])) if (both & s).any() else np.nan,
        "team_err_unseen": float(np.mean(t[both & ~s] != p[both & ~s])) if (both & ~s).any() else np.nan,
        "share_both_alive_unseen": float(np.mean(both & ~s)),
    }
    # recall of true possessions by duration, at IoU 0.5
    gt, pr = pm.label_codes(truth), pm.label_codes(pred)
    runs = pm.valid_runs(gt)
    gs = pm.possession_spells(gt, runs)
    ps = pm.possession_spells(pm.fill_abstentions(pr, runs), runs)
    flags = pm.matched_gt_flags(pm.spell_overlaps(gs, ps), len(gs), 0.5)
    dur = np.array([(b - a) / 25.0 for a, b, *_ in gs])
    for lo, hi in zip(SPELL_BINS[:-1], SPELL_BINS[1:]):
        m = (dur >= lo) & (dur < hi)
        key = "recall_{}-{}s".format(lo, int(hi) if hi < 1e8 else "inf")
        out[key] = float(flags[m].mean()) if m.any() else np.nan
        out["n_" + key[7:]] = int(m.sum())
    return out


def knobs_id(knobs):
    return hashlib.sha256(json.dumps(asdict(knobs), sort_keys=True, default=str).encode()).hexdigest()[:10]


class Bench:
    """Degraded matches, generated once per (match, seed, knobs) and kept in memory."""

    def __init__(self, matches=None, knobs=None):
        self.matches = list(matches or dfl.match_ids())
        self.knobs = knobs or degrade.load_knobs()
        self._clean, self._obs = {}, {}

    def clean(self, mid):
        if mid not in self._clean:
            self._clean[mid] = dfl.load(mid)
        return self._clean[mid]

    def get(self, mid, seed=0, condition="degraded", **overrides):
        key = (mid, seed, condition, tuple(sorted(overrides.items())))
        if key not in self._obs:
            c = self.clean(mid)
            if condition == "clean":
                self._obs[key] = (degrade.clean_obs(c), c["labels"].copy())
            else:
                self._obs[key] = degrade.degrade(c, seed=seed, knobs=self.knobs, **overrides)
        return self._obs[key]


def run(bench, model_names, seeds=(0,), condition="degraded", train_seeds=(100, 101),
        verbose=True, **overrides):
    """DataFrame, one row per (model, match, seed)."""
    from bench.models import REGISTRY
    rows = []
    for name in model_names:
        model = REGISTRY[name]
        learned = hasattr(model, "fit")
        for seed in seeds:
            for mid in bench.matches:
                t0 = time.time()
                obs, truth = bench.get(mid, seed, condition, **overrides)
                if learned:
                    train = [bench.get(m, s, condition, **overrides)
                             for m in bench.matches if m != mid for s in train_seeds]
                    fitted = model.fit(train)
                    pred = fitted.predict(obs)
                else:
                    pred = model(obs)
                row = {"model": name, "match": mid, "seed": seed, "condition": condition,
                       **{k: v for k, v in overrides.items()}}
                row.update(score(truth, pred))
                row.update(breakdown(truth, pred, obs))
                row["seconds"] = round(time.time() - t0, 1)
                rows.append(row)
                os.makedirs(RESULTS, exist_ok=True)  # partial results survive a killed run
                pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "_partial.csv"), index=False)
                if verbose:
                    print("{:<20} {} seed {} spellF1 {:.3f} macroF1 {:.3f} ({:.0f} s)".format(
                        name, mid[-6:], seed, row["spell_f1_mean"], row["macro_f1"],
                        row["seconds"]), flush=True)
    return pd.DataFrame(rows)


def summary(df, ref=None):
    """Mean over matches per model, and paired wins against ref."""
    keys = ["spell_f1_mean", "macro_f1", "f1_H", "f1_A", "f1_D", "alive_dead_f1",
            "team_acc_both_alive", "n_pred_spells", "n_gt_spells", "missed_dead", "false_dead",
            "team_err_seen", "team_err_unseen", "recall_0-2s", "recall_2-5s", "recall_5-15s",
            "recall_15-infs"]
    keys = [k for k in keys if k in df]
    group = [c for c in ("condition", "model") if c in df]
    table = df.groupby(group, sort=False)[keys].mean()
    table.insert(1, "spell_sd", df.groupby(group, sort=False)["spell_f1_mean"].std())
    if ref is not None and ref in set(df["model"]):
        wins, diffs, pvals = [], [], []
        for idx in table.index:
            model = idx[-1] if isinstance(idx, tuple) else idx
            a = df[df.model == ref].set_index(["match", "seed"])["spell_f1_mean"]
            b = df[df.model == model].set_index(["match", "seed"])["spell_f1_mean"]
            d = (b - a).dropna()
            w = int((d > 0).sum())
            wins.append("{}/{}".format(w, len(d)))
            diffs.append(d.mean())
            pvals.append(sign_test(w, len(d)))
        table["d_spell_vs_" + ref] = diffs
        table["wins"] = wins
        table["p_sign"] = pvals
    return table


def sign_test(wins, n):
    """One-sided P(at least `wins` wins out of n | no difference)."""
    from math import comb
    return sum(comb(n, k) for k in range(wins, n + 1)) / 2.0 ** n if n else np.nan


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models", default="baseline_smooth50,pol,v1,v3,v4")
    ap.add_argument("--matches", default="")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--missing", default="", help="comma list of ball_missing rates to sweep")
    ap.add_argument("--condition", default="degraded", choices=("degraded", "clean"))
    ap.add_argument("--ref", default="pol")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    from bench import gbdt  # noqa: F401  (registers the learned models)
    bench = Bench(args.matches.split(",") if args.matches else None)
    seeds = [int(s) for s in args.seeds.split(",")]
    models = args.models.split(",")
    frames = []
    rates = [float(r) for r in args.missing.split(",")] if args.missing else [None]
    for rate in rates:
        extra = {} if rate is None else {"ball_missing": rate}
        frames.append(run(bench, models, seeds, args.condition, **extra))
    df = pd.concat(frames, ignore_index=True)
    os.makedirs(RESULTS, exist_ok=True)
    tag = args.tag or "{}_{}_{}".format(args.condition, knobs_id(bench.knobs),
                                        time.strftime("%Y%m%d-%H%M%S"))
    path = os.path.join(RESULTS, tag + ".csv")
    df.to_csv(path, index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    group_cols = ["ball_missing"] if "ball_missing" in df else []
    if group_cols:
        for rate, sub in df.groupby("ball_missing"):
            print("\nball_missing = {}".format(rate))
            print(summary(sub, args.ref).round(3).to_string())
    else:
        print(summary(df, args.ref).round(3).to_string())
    print("-> {}".format(os.path.relpath(path, ROOT)))


if __name__ == "__main__":
    main()
