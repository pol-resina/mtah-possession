"""LightGBM team model, leave-one-match-out, memory-lean. Usage: python lgb_lean.py <name> <seeds...>"""
import json, os, sys, time
import numpy as np
import lightgbm as lgb
import common as C

def fit(train, seeds):
    sizes = []
    for m in train:
        _, y, _, _ = C.meta(m)
        sizes.append(int(np.sum(y != "D")))
    n = sum(sizes) * len(seeds) * 2
    X0, _ = C.table(train[0], seeds[0], False)
    Xall = np.empty((n, X0.shape[1]), dtype=np.float32); yall = np.empty(n, dtype=np.float32)
    del X0
    i = 0
    for m in train:
        for s in seeds:
            for swap in (False, True):
                X, y = C.table(m, s, swap)
                keep = y != "D"
                k = int(keep.sum())
                Xall[i:i + k] = X[keep]; yall[i:i + k] = (y[keep] == "H")
                i += k
                del X
    assert i == n
    ds = lgb.Dataset(Xall, label=yall, params=C.gbdt.PARAMS); ds.construct()
    del Xall
    return lgb.train(C.gbdt.PARAMS, ds, C.gbdt.N_TREES)

if __name__ == "__main__":
    name, seeds = sys.argv[1], tuple(int(s) for s in sys.argv[2:])
    res = {}
    for m in C.MATCHES:
        t0 = time.time()
        model = fit([o for o in C.MATCHES if o != m], seeds)
        Xe, _ = C.table(m, C.gbdt.EVAL_SEED, False)
        p = model.predict(Xe)
        np.save(os.path.join(C.OUT, "p_{}_{}.npy".format(name, m[-6:])), p.astype(np.float32))
        res[m] = C.score(C.decode_team(p, m), m)
        print("{} {} spellF1 {:.4f} macroF1 {:.4f} team_acc {:.4f} spells {}/{} ({:.0f}s)".format(
            name, m[-6:], res[m]["spell"], res[m]["macro"], res[m]["team_acc"], res[m]["n_pred"], res[m]["n_gt"], time.time() - t0), flush=True)
    json.dump(res, open(os.path.join(C.OUT, "results_{}.json".format(name)), "w"), indent=1)
    print("{} mean spellF1 {:.4f}".format(name, np.mean([v["spell"] for v in res.values()])), flush=True)
