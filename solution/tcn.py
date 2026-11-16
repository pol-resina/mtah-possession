"""Team step, temporal network: P(home | ball alive) at 5 Hz from the same 530 features as the
trees, read over about 12 s either side of each row. NumPy only (float64), so the result does not
depend on PyTorch; training/team_tcn.py fits it on the open DFL matches."""

import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TCN_MODEL = os.path.join(HERE, "team_tcn.npz")


def _conv(x, w, b, d):
    """x (C_in, T), w (C_out, C_in, K), dilation d, zero padding -> (C_out, T)."""
    k = w.shape[2]
    pad = d * (k - 1) // 2
    t = x.shape[1]
    xp = np.zeros((x.shape[0], t + 2 * pad))
    xp[:, pad:pad + t] = x
    y = np.repeat(b[:, None], t, axis=1).astype(np.float64)
    for j in range(k):
        y += w[:, :, j] @ xp[:, j * d:j * d + t]
    return y


def logits(params, z):
    """z (F, T): standardised features of one half -> (T,) logits of P(home | alive)."""
    relu = lambda v: np.maximum(v, 0.0)
    h = _conv(z, params["inp_w"], params["inp_b"], 1)
    for i, d in enumerate(params["dilations"]):
        a = _conv(relu(h), params["a{}_w".format(i)], params["a{}_b".format(i)], int(d))
        h = h + _conv(relu(a), params["b{}_w".format(i)], params["b{}_b".format(i)], int(d))
    return _conv(relu(h), params["out_w"], params["out_b"], 1)[0]


def probability(X_rows, rows, halves, names, path=TCN_MODEL):
    """P(home | alive) at the 5 Hz rows; the network runs over each half as one sequence."""
    with np.load(path, allow_pickle=False) as d:
        params = {k: d[k] for k in d.files}
    if list(params["names"]) != list(names):
        raise ValueError("feature columns differ from the ones the network was trained on")
    p = np.zeros(len(rows))
    for _, a, b in halves:
        sel = (rows >= a) & (rows <= b)
        z = (X_rows[sel].astype(np.float64) - params["mean"]) / params["std"]
        z[~np.isfinite(z)] = 0.0
        z = np.clip(z, -10, 10)
        p[sel] = 1.0 / (1.0 + np.exp(-logits(params, z.T)))
    return p
