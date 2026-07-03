"""Verify the from-scratch numpy RNN/LSTM (C5 W1)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "C5_sequence_models"))
from a1_rnn_lstm_step_by_step import (  # noqa: E402
    lstm_cell_forward, lstm_forward, rnn_cell_backward, rnn_cell_forward,
    rnn_forward, softmax,
)


def _rnn_params(n_x, n_a, n_y, seed=1):
    np.random.seed(seed)
    return {
        "Waa": np.random.randn(n_a, n_a), "Wax": np.random.randn(n_a, n_x),
        "Wya": np.random.randn(n_y, n_a), "ba": np.random.randn(n_a, 1),
        "by": np.random.randn(n_y, 1),
    }


def test_softmax_sums_to_one():
    x = np.random.randn(5, 8)
    s = softmax(x)
    assert np.allclose(s.sum(axis=0), 1.0)


def test_rnn_forward_shapes():
    n_x, n_a, n_y, m, T = 3, 5, 2, 10, 7
    x = np.random.randn(n_x, m, T)
    a0 = np.random.randn(n_a, m)
    params = _rnn_params(n_x, n_a, n_y)
    a, y, (caches, _) = rnn_forward(x, a0, params)
    assert a.shape == (n_a, m, T)
    assert y.shape == (n_y, m, T)
    assert len(caches) == T
    # y is a probability distribution over classes at each step
    assert np.allclose(y.sum(axis=0), 1.0)


def test_rnn_cell_backward_numerical():
    """Check dWax and dxt of a single RNN cell against numerical gradients of
    the scalar sum(a_next)."""
    np.random.seed(3)
    n_x, n_a, n_y, m = 3, 4, 2, 5
    xt = np.random.randn(n_x, m)
    a_prev = np.random.randn(n_a, m)
    params = _rnn_params(n_x, n_a, n_y, seed=3)

    a_next, _, cache = rnn_cell_forward(xt, a_prev, params)
    da_next = np.ones_like(a_next)  # d sum(a_next)
    grads = rnn_cell_backward(da_next, cache)

    eps = 1e-6
    # dWax
    Wax = params["Wax"]
    for _ in range(5):
        i, j = np.random.randint(Wax.shape[0]), np.random.randint(Wax.shape[1])
        p2 = {k: v.copy() for k, v in params.items()}
        p2["Wax"][i, j] += eps
        ap, _, _ = rnn_cell_forward(xt, a_prev, p2)
        p3 = {k: v.copy() for k, v in params.items()}
        p3["Wax"][i, j] -= eps
        am, _, _ = rnn_cell_forward(xt, a_prev, p3)
        num = (ap.sum() - am.sum()) / (2 * eps)
        assert abs(num - grads["dWax"][i, j]) < 1e-4

    # dxt
    for _ in range(5):
        i, j = np.random.randint(n_x), np.random.randint(m)
        xp = xt.copy(); xp[i, j] += eps
        xm = xt.copy(); xm[i, j] -= eps
        ap, _, _ = rnn_cell_forward(xp, a_prev, params)
        am, _, _ = rnn_cell_forward(xm, a_prev, params)
        num = (ap.sum() - am.sum()) / (2 * eps)
        assert abs(num - grads["dxt"][i, j]) < 1e-4


def test_lstm_cell_gate_ranges():
    np.random.seed(2)
    n_x, n_a, n_y, m = 3, 5, 2, 4
    xt = np.random.randn(n_x, m)
    a_prev = np.random.randn(n_a, m)
    c_prev = np.random.randn(n_a, m)
    lp = {}
    for g in ["f", "i", "c", "o"]:
        lp[f"W{g}"] = np.random.randn(n_a, n_a + n_x)
        lp[f"b{g}"] = np.random.randn(n_a, 1)
    lp["Wy"] = np.random.randn(n_y, n_a)
    lp["by"] = np.random.randn(n_y, 1)
    a_next, c_next, yt, cache = lstm_cell_forward(xt, a_prev, c_prev, lp)
    # gates are sigmoids in (0,1)
    ft = cache[4]
    assert np.all((ft > 0) & (ft < 1))
    assert a_next.shape == (n_a, m)
    assert np.allclose(yt.sum(axis=0), 1.0)


def test_lstm_forward_shapes():
    np.random.seed(2)
    n_x, n_a, n_y, m, T = 3, 5, 2, 4, 6
    x = np.random.randn(n_x, m, T)
    a0 = np.random.randn(n_a, m)
    lp = {}
    for g in ["f", "i", "c", "o"]:
        lp[f"W{g}"] = np.random.randn(n_a, n_a + n_x)
        lp[f"b{g}"] = np.random.randn(n_a, 1)
    lp["Wy"] = np.random.randn(n_y, n_a)
    lp["by"] = np.random.randn(n_y, 1)
    a, y, c, _ = lstm_forward(x, a0, lp)
    assert a.shape == (n_a, m, T)
    assert c.shape == (n_a, m, T)
    assert y.shape == (n_y, m, T)
