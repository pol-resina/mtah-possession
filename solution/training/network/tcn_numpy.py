"""NumPy inference for the team TCN (float64): the same function as the PyTorch model, no torch."""
import numpy as np


def _conv(x, w, b, d):
    """x (C_in, T), w (C_out, C_in, K), dilation d, zero padding d*(K-1)/2 -> (C_out, T)."""
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
    """z (F, T) standardised features of one half -> (T,) logits of P(home | alive)."""
    relu = lambda v: np.maximum(v, 0.0)
    h = _conv(z, params["inp_w"], params["inp_b"], 1)
    for i, d in enumerate(params["dilations"]):
        a = _conv(relu(h), params["a{}_w".format(i)], params["a{}_b".format(i)], int(d))
        h = h + _conv(relu(a), params["b{}_w".format(i)], params["b{}_b".format(i)], int(d))
    return _conv(relu(h), params["out_w"], params["out_b"], 1)[0]


def prepare(X, mean, std):
    z = (X - mean) / std
    z[~np.isfinite(z)] = 0.0
    return np.clip(z, -10, 10)


def export(model, mean, std, names, path):
    """Save a trained tcn.TCN as plain arrays (float64)."""
    sd = {k: v.detach().double().numpy() for k, v in model.state_dict().items()}
    out = {"inp_w": sd["inp.weight"], "inp_b": sd["inp.bias"], "out_w": sd["out.weight"], "out_b": sd["out.bias"],
           "dilations": np.array([blk.a.dilation[0] for blk in model.blocks]), "mean": mean.astype(np.float64),
           "std": std.astype(np.float64), "names": np.array(names)}
    for i in range(len(model.blocks)):
        out["a{}_w".format(i)] = sd["blocks.{}.a.weight".format(i)]; out["a{}_b".format(i)] = sd["blocks.{}.a.bias".format(i)]
        out["b{}_w".format(i)] = sd["blocks.{}.b.weight".format(i)]; out["b{}_b".format(i)] = sd["blocks.{}.b.bias".format(i)]
    np.savez(path, **out)
