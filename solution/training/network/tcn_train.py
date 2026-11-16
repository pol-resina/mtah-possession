"""Temporal conv net for P(home | alive) on Adrià's 530 features at 5 Hz, leave-one-match-out,
team only (true dead ball, our decoder). Usage: python tcn.py <name> [seeds...]"""
import json, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import common as C

torch.set_num_threads(8)
STEPS, BATCH, CROP, LR, WD = 2000, 32, 256, 2e-3, 1e-4
WIDTH, DILATIONS, DROP = 64, (1, 2, 4, 8, 16), 0.1


class Block(nn.Module):
    def __init__(self, c, d):
        super().__init__()
        self.a = nn.Conv1d(c, c, 3, dilation=d, padding=d)
        self.b = nn.Conv1d(c, c, 3, dilation=d, padding=d)
        self.drop = nn.Dropout(DROP)

    def forward(self, x):
        return x + self.b(self.drop(torch.relu(self.a(torch.relu(x)))))


class TCN(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.inp = nn.Conv1d(f, WIDTH, 1)
        self.blocks = nn.Sequential(*[Block(WIDTH, d) for d in DILATIONS])
        self.out = nn.Conv1d(WIDTH, 1, 1)

    def forward(self, x):            # x (B, F, T) -> logits (B, T)
        return self.out(torch.relu(self.blocks(self.inp(x))))[:, 0]


def halves_of(m, X, y):
    """Split one copy into its two halves: [(X_half, target, mask)]."""
    rows, _, halves, _ = C.meta(m)
    out = []
    for _, a, b in halves:
        sel = (rows >= a) & (rows <= b)
        yy = y[sel]
        out.append((X[sel], (yy == "H").astype(np.float32), (yy != "D").astype(np.float32)))
    return out


def standardise(seqs):
    n = sum(len(s[0]) for s in seqs)
    mean = np.zeros(seqs[0][0].shape[1]); sq = np.zeros_like(mean); cnt = np.zeros_like(mean)
    for X, _, _ in seqs:
        ok = np.isfinite(X)
        mean += np.where(ok, X, 0).sum(0); cnt += ok.sum(0)
    mean /= np.maximum(cnt, 1)
    for X, _, _ in seqs:
        d = np.where(np.isfinite(X), X - mean, 0)
        sq += (d * d).sum(0)
    std = np.sqrt(sq / np.maximum(cnt, 1)); std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def prep(X, mean, std):
    Z = (X - mean) / std
    Z[~np.isfinite(Z)] = 0.0
    return np.clip(Z, -10, 10).astype(np.float32)


def fit(train, seeds, rng_seed=0):
    seqs = []
    for m in train:
        for s in seeds:
            for swap in (False, True):
                X, y = C.table(m, s, swap)
                seqs += halves_of(m, X, y)
                del X
    mean, std = standardise(seqs)
    seqs = [(torch.from_numpy(prep(X, mean, std).T.copy()), torch.from_numpy(t), torch.from_numpy(k)) for X, t, k in seqs]
    torch.manual_seed(rng_seed); rng = np.random.default_rng(rng_seed)
    model = TCN(seqs[0][0].shape[0])
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=STEPS)
    lossf = nn.BCEWithLogitsLoss(reduction="none")
    model.train()
    for step in range(STEPS):
        xb, tb, kb = [], [], []
        for _ in range(BATCH):
            X, t, k = seqs[rng.integers(len(seqs))]
            a = int(rng.integers(0, X.shape[1] - CROP))
            xb.append(X[:, a:a + CROP]); tb.append(t[a:a + CROP]); kb.append(k[a:a + CROP])
        xb, tb, kb = torch.stack(xb), torch.stack(tb), torch.stack(kb)
        loss = (lossf(model(xb), tb) * kb).sum() / kb.sum().clamp(min=1)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    model.eval()
    return model, mean, std


def predict(model, mean, std, m, X):
    rows, _, halves, _ = C.meta(m)
    p = np.zeros(len(rows), dtype=np.float32)
    with torch.no_grad():
        for _, a, b in halves:
            sel = (rows >= a) & (rows <= b)
            z = torch.from_numpy(prep(X[sel], mean, std).T.copy())[None]
            p[sel] = torch.sigmoid(model(z))[0].numpy()
    return p


if __name__ == "__main__":
    name = sys.argv[1]
    seeds = tuple(int(s) for s in sys.argv[2:]) or (100, 101)
    res = {}
    for m in C.MATCHES:
        t0 = time.time()
        model, mean, std = fit([o for o in C.MATCHES if o != m], seeds)
        Xe, _ = C.table(m, C.gbdt.EVAL_SEED, False)
        p = predict(model, mean, std, m, Xe)
        np.save(os.path.join(C.OUT, "p_{}_{}.npy".format(name, m[-6:])), p)
        res[m] = C.score(C.decode_team(p, m), m)
        print("{} {} spellF1 {:.4f} macroF1 {:.4f} team_acc {:.4f} spells {}/{} ({:.0f}s)".format(
            name, m[-6:], res[m]["spell"], res[m]["macro"], res[m]["team_acc"], res[m]["n_pred"], res[m]["n_gt"], time.time() - t0), flush=True)
    json.dump(res, open(os.path.join(C.OUT, "results_{}.json".format(name)), "w"), indent=1)
    print("{} mean spellF1 {:.4f}".format(name, np.mean([v["spell"] for v in res.values()])), flush=True)
