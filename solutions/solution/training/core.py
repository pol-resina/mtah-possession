"""The bench: train a team model on six degraded DFL matches, test the whole pipeline on the seventh.

    python solutions/solution/training/core.py              # leave-one-match-out for the shipped recipe

Everything here runs on other matches. Our own match and its public ground truth are never read.
"""

import multiprocessing
import os
import sys
from types import SimpleNamespace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SOLUTION = os.path.dirname(HERE)                   # solutions/solution/
ROOT = os.path.dirname(os.path.dirname(SOLUTION))  # the repository
for path in (os.path.join(ROOT, "evaluation"), SOLUTION, HERE):
    if path not in sys.path:
        sys.path.insert(0, path)
import possession_metrics as pm  # noqa: E402
import deadball  # noqa: E402
import features as FT  # noqa: E402
from segments import runs, two_state  # noqa: E402
import dfl  # noqa: E402
import sim  # noqa: E402

TABLES = os.path.join(ROOT, "cache", "dfl", "tables")
MODELS = os.path.join(ROOT, "cache", "dfl", "models")
TRAIN_SEEDS = (100, 101)   # two degraded copies of every training match...
TEST_SEED = 0              # ...and another one of the test match
SWITCH_COST = 1.0          # nats per change of team, as in the solution
ROW_FRAMES = 5
# The settings stored in the shipped model, ../team_model.txt
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200,
              bagging_fraction=0.8, bagging_freq=1, feature_fraction=0.5, lambda_l2=1.0, seed=0,
              deterministic=True, force_row_wise=True, verbosity=-1, num_threads=16)
ROUNDS = 400


def as_obs(match):
    """A degraded match in the format features.py reads."""
    n, slots = match.px.shape
    players = np.full((n, slots, 3), np.nan, dtype=np.float32)
    players[:, :, 0] = match.px / match.length
    players[:, :, 1] = match.py / match.width
    ball = np.column_stack([match.bx / match.length, match.by / match.width]).astype(np.float32)
    return SimpleNamespace(track={"players": players, "ball": ball, "team": match.team},
                           pitch=(match.length, match.width), halves=match.halves,
                           attacks_right=match.attacks_right, kickoffs=match.kickoffs,
                           fps=match.fps, name=match.game)


def base_features(match, obs, swap):
    return FT.frame_features(obs, swap=swap)


FEATURES = {"base": base_features}
# A training recipe: which features, and which labels (seconds of sim.controlled; 0 = as labelled).
KINDS = {"base": ("base", 0.0), "base_c2": ("base", 2.0)}
CLEAN_BALL = ("cleanball",)   # feature kinds computed with a ball that is never lost or bridged in play
FEATURES["cleanball"] = base_features


_truth = {}


def table(name, seed, swap=False, kind="base"):
    """{X: features at the 5 Hz rows, rows, y: home has the ball, alive, names}. The features are
    cached on disk; a recipe with other labels reuses them and only replaces y."""
    features, control = KINDS.get(kind, (kind, 0.0))
    if control > 0:
        t = dict(table(name, seed, swap, features))
        if (name, control) not in _truth:
            _truth[(name, control)] = sim.truth_of(name, control)
        t["y"] = _truth[(name, control)][t["rows"]] == ("A" if swap else "H")
        return t
    kind = features
    path = os.path.join(TABLES, "{}_{}_{}_{}.npz".format(kind, name, seed, int(swap)))
    if os.path.isfile(path):
        with np.load(path, allow_pickle=False) as data:
            return {key: data[key] for key in data.files}
    match = sim.atd_like(name, seed, clean_ball=kind in CLEAN_BALL)
    obs = as_obs(match)
    names, X = FEATURES[kind](match, obs, swap)
    rows = FT.sample_rows(obs)
    truth = match.truth[rows]
    out = dict(X=X[rows].astype(np.float32), rows=rows, y=(truth == ("A" if swap else "H")),
               alive=truth != "D", names=np.array(names))
    os.makedirs(TABLES, exist_ok=True)
    np.savez(path + ".part.npz", **out)
    os.replace(path + ".part.npz", path)
    return out


def _build(job):
    table(*job)
    return job


def build_tables(kind="base", names=dfl.MATCHES, workers=3):
    kind = KINDS.get(kind, (kind, 0.0))[0]
    jobs = [(n, s, w, kind) for n in names for s in TRAIN_SEEDS for w in (False, True)]
    jobs += [(n, TEST_SEED, False, kind) for n in names]
    jobs = [j for j in jobs if not os.path.isfile(
        os.path.join(TABLES, "{}_{}_{}_{}.npz".format(j[3], j[0], j[1], int(j[2]))))]
    if jobs:
        with multiprocessing.Pool(workers) as pool:
            for _ in pool.imap_unordered(_build, jobs):
                pass


def training_set(names, kind="base", columns=None, target="team"):
    """(X, y) from every training table of these matches. target "team": the alive rows, y = home
    has the ball. target "dead": every row, y = the ball is dead."""
    Xs, ys = [], []
    for name in names:
        for seed in TRAIN_SEEDS:
            for swap in (False, True):
                t = table(name, seed, swap, kind)
                keep = t["alive"] if target == "team" else np.ones(len(t["alive"]), dtype=bool)
                X = t["X"][keep]
                Xs.append(X if columns is None else X[:, columns])
                ys.append(t["y"][keep] if target == "team" else ~t["alive"])
    return np.vstack(Xs), np.concatenate(ys)


def train(names, kind="base", params=None, rounds=ROUNDS, columns=None, target="team", tag=None):
    """A LightGBM model on these matches. tag: cache it in cache/dfl/models/ under this name."""
    import lightgbm as lgb
    path = None if tag is None else os.path.join(MODELS, tag + ".txt")
    if path is not None and os.path.isfile(path):
        with open(path) as handle:
            return lgb.Booster(model_str=handle.read())
    X, y = training_set(names, kind, columns, target)
    data = lgb.Dataset(X, label=y.astype(np.float32), free_raw_data=True)
    booster = lgb.train(dict(PARAMS, **(params or {})), data, num_boost_round=rounds)
    if path is not None:
        os.makedirs(MODELS, exist_ok=True)
        booster.save_model(path)
    return booster


def frame_probability(match, t, booster, columns=None):
    """P(home) per frame of a test match from its table."""
    X = t["X"] if columns is None else t["X"][:, columns]
    return FT.expand(booster.predict(X), t["rows"], match.n, match.halves)


def decode(match, p, dead, switch=SWITCH_COST):
    """Letters per frame: D where dead, else the two-state Viterbi over P(home)."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    cost_home, cost_away = -np.log(p) / ROW_FRAMES, -np.log(1 - p) / ROW_FRAMES
    out = np.full(match.n, "X", dtype="<U1")
    for _, first, last in match.halves:
        out[first:last + 1] = "D"
        for a, b in runs(~dead, first, last + 1):
            state = two_state(cost_home[a:b], cost_away[a:b], switch, switch)
            out[a:b] = np.where(state == 0, "H", "A")
    return out


def detected_dead(match):
    return deadball.detect(match, match.kickoffs)[0]


def learned_dead(match, p_dead, switch):
    """Dead mask from P(dead) per frame: a two-state Viterbi per half, dead before the kickoff."""
    p = np.clip(p_dead, 1e-6, 1 - 1e-6)
    cost_live, cost_dead = -np.log(1 - p) / ROW_FRAMES, -np.log(p) / ROW_FRAMES
    dead = np.zeros(match.n, dtype=bool)
    for half, first, last in match.halves:
        kickoff = match.kickoffs.get(half, first)
        dead[first:kickoff] = True
        dead[kickoff:last + 1] = two_state(cost_live[kickoff:last + 1], cost_dead[kickoff:last + 1],
                                           switch, switch, start_cost=(0.0, 1e6)).astype(bool)
    return dead


def score(match, labels):
    s = pm.evaluate(pm.label_codes(match.truth), pm.label_codes(labels), bootstrap_n=0)
    return {"spell": s["spell_f1_mean"], "macro": s["macro_f1"], "team": s["team_acc_gt_alive"],
            "n_pred": s["n_pred_spells"], "n_gt": s["n_gt_spells"]}


def lomo(kind="base", params=None, rounds=ROUNDS, columns=None, names=dfl.MATCHES, verbose=True,
         keep=None):
    """Leave one match out: {match: {"true dead ball": scores, "our dead ball": scores}}."""
    build_tables(kind, names)
    out = {}
    cacheable = params is None and columns is None and rounds == ROUNDS
    for held in names:
        booster = train([n for n in names if n != held], kind, params, rounds, columns,
                        tag="team_{}_{}".format(kind, held) if cacheable else None)
        match = sim.atd_like(held, TEST_SEED, KINDS.get(kind, (kind, 0.0))[1],
                             clean_ball=KINDS.get(kind, (kind, 0.0))[0] in CLEAN_BALL)
        p = frame_probability(match, table(held, TEST_SEED, False, kind), booster, columns)
        if keep is not None:
            keep[held] = (match, p)
        true_dead = match.truth == "D"
        out[held] = {"true dead ball": score(match, decode(match, p, true_dead)),
                     "our dead ball": score(match, decode(match, p, detected_dead(match)))}
        if verbose:
            a, b = out[held]["true dead ball"], out[held]["our dead ball"]
            print("  {}  true dead ball: spellF1 {:.3f} macroF1 {:.3f} team {:.3f} ({} vs {} spells)   "
                  "our dead ball: spellF1 {:.3f} macroF1 {:.3f}".format(
                      held, a["spell"], a["macro"], a["team"], a["n_pred"], a["n_gt"], b["spell"], b["macro"]))
            sys.stdout.flush()
    return out


def summary(result, label=""):
    line = "{:<34}".format(label)
    for mode in ("true dead ball", "our dead ball"):
        for key in ("spell", "macro"):
            line += "  {} {} {:.4f}".format(mode.split()[0], key, np.mean([r[mode][key] for r in result.values()]))
    return line


def paired(a, b, mode="our dead ball", key="spell"):
    """b minus a per match: (mean difference, matches where b is better, number of matches)."""
    d = np.array([b[m][mode][key] - a[m][mode][key] for m in a])
    return float(d.mean()), int((d > 0).sum()), len(d)


if __name__ == '__main__':
    result = lomo()
    print(summary(result, "shipped recipe, leave one out"))
