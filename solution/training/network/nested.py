"""Nested choice of the decoder switch cost: for each held-out match, the cost that maximises mean
spellF1 on the other six; then the held-out score at that cost."""
import os
import numpy as np
import common as C
GRID = (0.5, 1.0, 2.0, 4.0, 8.0)
load = lambda n, m: np.load(os.path.join(C.OUT, "p_{}_{}.npy".format(n, m[-6:]))).astype(np.float64)
logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
cands = {"tcn": lambda m: load("tcn2", m),
         "avg": lambda m: 1 / (1 + np.exp(-(logit(load("base", m)) + logit(load("tcn2", m))) / 2))}
base = {m: C.score(C.decode_team(load("base", m), m), m)["spell"] for m in C.MATCHES}
for name, get in cands.items():
    table = {}
    for sw in GRID:
        C.SWITCH = sw
        table[sw] = {m: C.score(C.decode_team(get(m), m), m)["spell"] for m in C.MATCHES}
    C.SWITCH = 1.0
    held, picks = {}, {}
    for m in C.MATCHES:
        others = [o for o in C.MATCHES if o != m]
        best = max(GRID, key=lambda sw: np.mean([table[sw][o] for o in others]))
        picks[m], held[m] = best, table[best][m]
    v = np.array([held[m] for m in C.MATCHES]); b = np.array([base[m] for m in C.MATCHES])
    print("{:<4} grid means: {}".format(name, "  ".join("{}:{:.4f}".format(sw, np.mean(list(table[sw].values()))) for sw in GRID)))
    print("{:<4} NESTED held-out mean {:.4f} (vs current {:.4f}: {:+.4f}, better on {}/7)  picks {}".format(
        name, v.mean(), b.mean(), v.mean() - b.mean(), int((v > b).sum()), [picks[m] for m in C.MATCHES]), flush=True)
