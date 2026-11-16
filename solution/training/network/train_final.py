"""Train the team TCN on all seven DFL matches (2 degraded copies each, both orientations) and
export it as plain arrays for solution/team_tcn.npz.

    python solution/training/network/train_final.py cache/team_model/team_tcn.npz 0
"""
import sys
import numpy as np
import common as C
import tcn_train as tcn
import tcn_numpy as tn

out = sys.argv[1]
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
model, mean, std = tcn.fit(C.MATCHES, (100, 101), rng_seed=seed)
names = list(np.load(C.gbdt._feature_path(C.MATCHES[0], 100, False, C.KID))["names"])
tn.export(model, mean, std, names, out)
print("->", out, "features", len(names))
