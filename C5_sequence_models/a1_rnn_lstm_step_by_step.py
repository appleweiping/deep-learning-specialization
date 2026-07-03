"""C5 W1 - Building a Recurrent Neural Network: Step by Step (pure numpy).

Implements the forward passes (and the RNN backward pass) for:
  * a vanilla RNN cell and a full RNN forward over T timesteps
  * an LSTM cell (forget/update/output gates, candidate cell) and full LSTM
    forward over T timesteps
plus softmax. Shapes follow the Coursera convention:
    x: (n_x, m, T_x)   a: (n_a, m, T_x)   y: (n_y, m, T_x)

Gradient correctness of rnn_cell_backward is verified numerically in
tests/test_c5_rnn.py.
"""
from __future__ import annotations

import numpy as np


def softmax(x):
    e = np.exp(x - np.max(x, axis=0, keepdims=True))
    return e / np.sum(e, axis=0, keepdims=True)


# --------------------------------------------------------------------------- #
# Vanilla RNN
# --------------------------------------------------------------------------- #
def rnn_cell_forward(xt, a_prev, parameters):
    Wax, Waa, Wya = parameters["Wax"], parameters["Waa"], parameters["Wya"]
    ba, by = parameters["ba"], parameters["by"]
    a_next = np.tanh(Waa @ a_prev + Wax @ xt + ba)
    yt_pred = softmax(Wya @ a_next + by)
    cache = (a_next, a_prev, xt, parameters)
    return a_next, yt_pred, cache


def rnn_forward(x, a0, parameters):
    caches = []
    n_x, m, T_x = x.shape
    n_y, n_a = parameters["Wya"].shape
    a = np.zeros((n_a, m, T_x))
    y_pred = np.zeros((n_y, m, T_x))
    a_next = a0
    for t in range(T_x):
        a_next, yt, cache = rnn_cell_forward(x[:, :, t], a_next, parameters)
        a[:, :, t] = a_next
        y_pred[:, :, t] = yt
        caches.append(cache)
    return a, y_pred, (caches, x)


def rnn_cell_backward(da_next, cache):
    (a_next, a_prev, xt, parameters) = cache
    Wax, Waa = parameters["Wax"], parameters["Waa"]
    dtanh = (1 - a_next ** 2) * da_next
    dxt = Wax.T @ dtanh
    dWax = dtanh @ xt.T
    da_prev = Waa.T @ dtanh
    dWaa = dtanh @ a_prev.T
    dba = np.sum(dtanh, axis=1, keepdims=True)
    return {"dxt": dxt, "da_prev": da_prev, "dWax": dWax, "dWaa": dWaa, "dba": dba}


# --------------------------------------------------------------------------- #
# LSTM
# --------------------------------------------------------------------------- #
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def lstm_cell_forward(xt, a_prev, c_prev, parameters):
    Wf, bf = parameters["Wf"], parameters["bf"]
    Wi, bi = parameters["Wi"], parameters["bi"]
    Wc, bc = parameters["Wc"], parameters["bc"]
    Wo, bo = parameters["Wo"], parameters["bo"]
    Wy, by = parameters["Wy"], parameters["by"]

    n_x, m = xt.shape
    concat = np.concatenate((a_prev, xt), axis=0)

    ft = sigmoid(Wf @ concat + bf)          # forget gate
    it = sigmoid(Wi @ concat + bi)          # update/input gate
    cct = np.tanh(Wc @ concat + bc)         # candidate cell
    c_next = ft * c_prev + it * cct
    ot = sigmoid(Wo @ concat + bo)          # output gate
    a_next = ot * np.tanh(c_next)
    yt = softmax(Wy @ a_next + by)

    cache = (a_next, c_next, a_prev, c_prev, ft, it, cct, ot, xt, parameters)
    return a_next, c_next, yt, cache


def lstm_forward(x, a0, parameters):
    caches = []
    n_x, m, T_x = x.shape
    n_y, n_a = parameters["Wy"].shape
    a = np.zeros((n_a, m, T_x))
    c = np.zeros((n_a, m, T_x))
    y = np.zeros((n_y, m, T_x))
    a_next = a0
    c_next = np.zeros((n_a, m))
    for t in range(T_x):
        a_next, c_next, yt, cache = lstm_cell_forward(
            x[:, :, t], a_next, c_next, parameters
        )
        a[:, :, t] = a_next
        c[:, :, t] = c_next
        y[:, :, t] = yt
        caches.append(cache)
    return a, y, c, (caches, x)


def _demo():
    np.random.seed(1)
    # RNN forward
    n_x, n_a, n_y, m, T = 3, 5, 2, 10, 4
    x = np.random.randn(n_x, m, T)
    a0 = np.random.randn(n_a, m)
    params = {
        "Waa": np.random.randn(n_a, n_a), "Wax": np.random.randn(n_a, n_x),
        "Wya": np.random.randn(n_y, n_a), "ba": np.random.randn(n_a, 1),
        "by": np.random.randn(n_y, 1),
    }
    a, y, _ = rnn_forward(x, a0, params)
    print(f"RNN  forward: a.shape={a.shape}  y.shape={y.shape}  a[4,1,-1]={a[4,1,-1]:.4f}")

    # LSTM forward
    lp = {}
    for g in ["f", "i", "c", "o"]:
        lp[f"W{g}"] = np.random.randn(n_a, n_a + n_x)
        lp[f"b{g}"] = np.random.randn(n_a, 1)
    lp["Wy"] = np.random.randn(n_y, n_a)
    lp["by"] = np.random.randn(n_y, 1)
    a, y, c, _ = lstm_forward(x, a0, lp)
    print(f"LSTM forward: a.shape={a.shape}  c.shape={c.shape}  y.shape={y.shape}  "
          f"a[4,1,-1]={a[4,1,-1]:.4f}")


if __name__ == "__main__":
    _demo()
