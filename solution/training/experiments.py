"""Experiments on the bench, each leave-one-match-out and compared match by match.

    python solution/training/experiments.py base      # the shipped recipe
    python solution/training/experiments.py dead      # a learned dead-ball step against the rule-based one
    python solution/training/experiments.py touch     # the shipped features plus touch features
    python solution/training/experiments.py switch    # the decoder's switch cost

Results are appended to cache/dfl/results.json so later runs can compare against earlier ones.
"""

import json
import os
import sys

import numpy as np

import core
import dfl
import sim
import touch

RESULTS = os.path.join(core.ROOT, "cache", "dfl", "results.json")
core.FEATURES["touch"] = touch.frame_features
core.KINDS.update({"touch": ("touch", 0.0), "touch_c2": ("touch", 2.0)})


def remember(name, result):
    data = json.load(open(RESULTS)) if os.path.isfile(RESULTS) else {}
    data[name] = result
    with open(RESULTS, "w") as handle:
        json.dump(data, handle, indent=1)


def recall(name):
    return json.load(open(RESULTS))[name] if os.path.isfile(RESULTS) else None


def compare(a_name, b_name, a, b):
    for mode in ("true dead ball", "our dead ball"):
        for key in ("spell", "macro"):
            if mode not in next(iter(a.values())) or mode not in next(iter(b.values())):
                continue
            d, wins, n = core.paired(a, b, mode, key)
            print("  {} vs {}, {} {}F1: {:+.4f}, better on {} of {} matches".format(b_name, a_name, mode, key, d, wins, n))


def full_score(match, labels):
    s = core.pm.evaluate(core.pm.label_codes(match.truth), core.pm.label_codes(labels), bootstrap_n=0)
    return {"spell": s["spell_f1_mean"], "macro": s["macro_f1"], "team": s["team_acc_gt_alive"],
            "alive_dead": s["alive_dead_f1"], "n_pred": s["n_pred_spells"], "n_gt": s["n_gt_spells"]}


def run_base():
    result = core.lomo("base")
    print(core.summary(result, "shipped recipe"))
    remember("base", result)


def run_touch():
    base = recall("base")
    result = core.lomo("touch")
    print(core.summary(result, "with touch features"))
    remember("touch", result)
    if base:
        print(core.summary(base, "shipped recipe"))
        compare("shipped", "touch", base, result)


def run_switch():
    """The decoder's switch cost, on the cached leave-one-out models (no retraining)."""
    for kind in ("base",):
        rows = {}
        for held in dfl.MATCHES:
            booster = core.train([n for n in dfl.MATCHES if n != held], kind, tag="team_{}_{}".format(kind, held))
            match = sim.atd_like(held, core.TEST_SEED)
            p = core.frame_probability(match, core.table(held, core.TEST_SEED, False, kind), booster)
            dead = core.detected_dead(match)
            for c in (0.25, 0.5, 1.0, 2.0, 4.0):
                rows.setdefault(c, []).append((core.score(match, core.decode(match, p, match.truth == "D", c))["spell"],
                                               core.score(match, core.decode(match, p, dead, c))["spell"]))
        for c, v in rows.items():
            v = np.array(v)
            print("  switch cost {:<5} spellF1: true dead ball {:.4f}   our dead ball {:.4f}".format(c, v[:, 0].mean(), v[:, 1].mean()))


def run_dead(kind="base", costs=(1.0, 2.0, 5.0, 10.0, 20.0)):
    """A learned dead-ball step (LightGBM P(dead) on the same features + Viterbi) against the
    rule-based detector, end to end with the leave-one-out team model."""
    out = {"rules": {}}
    for c in costs:
        out["learned {}".format(c)] = {}
    for held in dfl.MATCHES:
        others = [n for n in dfl.MATCHES if n != held]
        team = core.train(others, kind, tag="team_{}_{}".format(kind, held))
        dead_model = core.train(others, kind, target="dead", tag="dead_{}_{}".format(kind, held))
        match = sim.atd_like(held, core.TEST_SEED)
        t = core.table(held, core.TEST_SEED, False, kind)
        p = core.frame_probability(match, t, team)
        p_dead = core.FT.expand(dead_model.predict(t["X"]), t["rows"], match.n, match.halves)
        out["rules"][held] = full_score(match, core.decode(match, p, core.detected_dead(match)))
        line = "  {}  rules: spell {:.3f} macro {:.3f} alive/dead {:.3f}".format(
            held, out["rules"][held]["spell"], out["rules"][held]["macro"], out["rules"][held]["alive_dead"])
        for c in costs:
            s = full_score(match, core.decode(match, p, core.learned_dead(match, p_dead, c)))
            out["learned {}".format(c)][held] = s
            line += " | {}: {:.3f} {:.3f} {:.3f}".format(c, s["spell"], s["macro"], s["alive_dead"])
        print(line)
        sys.stdout.flush()
    print("\n{:<16} {:>8} {:>8} {:>10}   spell better than rules on".format("", "spellF1", "macroF1", "alive/dead"))
    for name, rows in out.items():
        wins = sum(rows[m]["spell"] > out["rules"][m]["spell"] for m in rows)
        print("{:<16} {:>8.4f} {:>8.4f} {:>10.4f}   {} of {}".format(
            name, *(np.mean([r[k] for r in rows.values()]) for k in ("spell", "macro", "alive_dead")), wins, len(rows)))
    remember("dead_" + kind, out)


def run_convention(kinds=("base", "base_c2"), costs=(0.5, 1.0, 2.0)):
    """Models trained on the labels as given and on the labels with brief touches removed, each
    scored against both, team step only (true dead ball)."""
    rows = {}
    for held in dfl.MATCHES:
        others = [n for n in dfl.MATCHES if n != held]
        match = sim.atd_like(held, core.TEST_SEED)
        truths = {"as labelled": match.truth, "controlled": sim.controlled(match.truth, 2.0)}
        dead = match.truth == "D"
        for kind in kinds:
            core.build_tables(kind)
            booster = core.train(others, kind, tag="team_{}_{}".format(kind, held))
            p = core.frame_probability(match, core.table(held, core.TEST_SEED, False, kind), booster)
            for c in costs:
                labels = core.decode(match, p, dead, c)
                for tname, truth in truths.items():
                    s = core.pm.evaluate(core.pm.label_codes(truth), core.pm.label_codes(labels), bootstrap_n=0)
                    rows.setdefault((kind, c, tname), {})[held] = (s["spell_f1_mean"], s["macro_f1"], s["n_pred_spells"], s["n_gt_spells"])
        line = "  {}".format(held)
        for kind in kinds:
            line += "   {} vs controlled: {:.3f}".format(kind, rows[(kind, 1.0, "controlled")][held][0])
        print(line)
        sys.stdout.flush()
    print("\n{:<10} {:>6} {:<12} {:>8} {:>8} {:>7} {:>7}".format("trained", "switch", "scored vs", "spellF1", "macroF1", "spells", "truth"))
    for (kind, c, tname), per in sorted(rows.items(), key=lambda kv: (kv[0][2], kv[0][0], kv[0][1])):
        v = np.array(list(per.values()))
        print("{:<10} {:>6} {:<12} {:>8.4f} {:>8.4f} {:>7.0f} {:>7.0f}".format(kind, c, tname, *v.mean(axis=0)))
    remember("convention_" + "_".join(kinds), {"{}|{}|{}".format(*k): v for k, v in rows.items()})
    a, b = (kinds[0], 1.0, "controlled"), (kinds[1], 1.0, "controlled")
    d = np.array([rows[b][m][0] - rows[a][m][0] for m in rows[a]])
    print("{} vs {} on controlled labels, switch 1: {:+.4f}, better on {} of {}".format(kinds[1], kinds[0], d.mean(), int((d > 0).sum()), len(d)))


def run_cleanball():
    """What do the ball's faults cost? The same recipe, trained and tested with a ball that is never
    lost or bridged while in play."""
    base = recall("base")
    result = core.lomo("cleanball")
    print(core.summary(result, "perfect ball in play"))
    remember("cleanball", result)
    if base:
        print(core.summary(base, "shipped recipe"))
        compare("shipped", "perfect ball", base, result)


if __name__ == '__main__':
    {"base": run_base, "touch": run_touch, "switch": run_switch, "dead": run_dead,
     "convention": run_convention, "cleanball": run_cleanball}[sys.argv[1]]()
