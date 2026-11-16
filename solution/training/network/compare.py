"""Compare saved out-of-fold probabilities: per-match spellF1, wins, averages, decoder switch cost.
    python compare.py base tcn2 [--avg] [--switch]"""
import sys
import os
import numpy as np
import common as C

def load(name, m):
    return np.load(os.path.join(C.OUT, "p_{}_{}.npy".format(name, m[-6:]))).astype(np.float64)

def run(get_p, switch=1.0):
    C.SWITCH = switch
    out = {m: C.score(C.decode_team(get_p(m), m), m)["spell"] for m in C.MATCHES}
    C.SWITCH = 1.0
    return out

def line(name, s, ref=None):
    v = np.array([s[m] for m in C.MATCHES])
    extra = ""
    if ref is not None:
        r = np.array([ref[m] for m in C.MATCHES])
        extra = "  vs base {:+.4f}, better on {}/7".format(v.mean() - r.mean(), int((v > r).sum()))
    print("{:<22} mean {:.4f}  ".format(name, v.mean()) + " ".join("{:.3f}".format(x) for x in v) + extra, flush=True)

base, cand = sys.argv[1], sys.argv[2]
ref = run(lambda m: load(base, m)); line(base, ref)
line(cand, run(lambda m: load(cand, m)), ref)
if "--avg" in sys.argv:
    logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
    avg = lambda m: 1 / (1 + np.exp(-(logit(load(base, m)) + logit(load(cand, m))) / 2))
    line("avg(" + base + "," + cand + ")", run(avg), ref)
if "--switch" in sys.argv:
    for sw in (0.5, 2.0, 4.0):
        line("{} switch {}".format(cand, sw), run(lambda m: load(cand, m), sw), ref)
