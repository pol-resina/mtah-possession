"""Train the team model on all seven bench matches.

    python solution/training/export.py [--out DIR] [--kind base|touch] [--dead]

Writes team_model.txt and team_model_features.txt (and, with --dead, dead_model.txt) into DIR,
cache/dfl/export/ by default. It never overwrites the solution's own model: copy the files next to
run.py by hand if you mean to ship them. The same seeds, the same degraded copies and the same
settings as the leave-one-match-out runs in experiments.py, only with no match held out.
"""

import os
import sys

import core
import dfl
import experiments  # noqa: F401  (registers the feature kinds)

if __name__ == '__main__':
    out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else \
        os.path.join(core.ROOT, "cache", "dfl", "export")
    kind = sys.argv[sys.argv.index("--kind") + 1] if "--kind" in sys.argv else "base"
    os.makedirs(out, exist_ok=True)
    core.build_tables(kind)
    names = core.table(dfl.MATCHES[0], core.TEST_SEED, False, kind)["names"]
    team = core.train(dfl.MATCHES, kind, tag="team_{}_all".format(kind))
    team.save_model(os.path.join(out, "team_model.txt"))
    with open(os.path.join(out, "team_model_features.txt"), "w") as handle:
        handle.write("\n".join(str(n) for n in names) + "\n")
    print("team model: {} trees, {} features -> {}".format(team.num_trees(), len(names), out))
    if "--dead" in sys.argv:
        dead = core.train(dfl.MATCHES, kind, target="dead", tag="dead_{}_all".format(kind))
        dead.save_model(os.path.join(out, "dead_model.txt"))
        print("dead-ball model: {} trees".format(dead.num_trees()))
