"""Train the v7 team model (GBDT P(H | alive)) on all 7 degraded DFL matches: the shipped
solution/team_model.txt and team_model_features.txt. Same data, PARAMS and trees as `gbdt --final`.

    python solution/training/bench/export_team.py [out.txt]   # default cache/team_model/team_model.txt

It never overwrites the solution's model: compare with `cmp`, or copy by hand.
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from bench import degrade, gbdt  # noqa: E402
from bench.harness import Bench  # noqa: E402

REPO = os.path.dirname(os.path.dirname(ROOT))  # the repository (ROOT is solution/training/)
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "cache", "team_model", "team_model.txt")

if __name__ == "__main__":
    import lightgbm as lgb
    bench = Bench(knobs=degrade.load_knobs())
    names, X, y = gbdt.training_set(bench, bench.matches)
    alive = y != "D"
    model = lgb.train(gbdt.PARAMS, lgb.Dataset(X[alive], label=(y[alive] == "H").astype(np.float32)),
                      gbdt.N_TREES)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    model.save_model(OUT)
    with open(OUT.replace(".txt", "_features.txt"), "w") as f:
        f.write("\n".join(names) + "\n")
    print("->", OUT, "knobs", gbdt.knobs_id(bench.knobs))
