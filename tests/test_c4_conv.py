"""Verify the from-scratch numpy conv/pool forward+backward (C4 W1)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "C4_convolutional_neural_networks"))
from a1_conv_step_by_step import (  # noqa: E402
    conv_backward, conv_forward, pool_backward, pool_forward, zero_pad,
)


def test_zero_pad_shape():
    X = np.random.randn(2, 4, 4, 3)
    Xp = zero_pad(X, 2)
    assert Xp.shape == (2, 8, 8, 3)
    assert np.all(Xp[:, 0, :, :] == 0)


def test_conv_forward_output_shape():
    np.random.seed(0)
    A_prev = np.random.randn(2, 5, 7, 4)
    W = np.random.randn(3, 3, 4, 8)
    b = np.random.randn(1, 1, 1, 8)
    Z, _ = conv_forward(A_prev, W, b, {"pad": 1, "stride": 2})
    # nH = (5-3+2)/2+1 = 3, nW = (7-3+2)/2+1 = 4
    assert Z.shape == (2, 3, 4, 8)


def test_conv_backward_matches_numerical():
    """Two-sided numerical gradient of sum(conv_forward) w.r.t. W and A_prev."""
    np.random.seed(1)
    A_prev = np.random.randn(1, 4, 4, 2)
    W = np.random.randn(2, 2, 2, 3)
    b = np.random.randn(1, 1, 1, 3)
    hp = {"pad": 1, "stride": 1}
    Z, cache = conv_forward(A_prev, W, b, hp)
    dZ = np.ones_like(Z)  # d(sum Z)/dZ = 1
    dA_prev, dW, db = conv_backward(dZ, cache)

    eps = 1e-5

    # check a few random entries of dW
    for _ in range(5):
        i, j, k, l = (np.random.randint(s) for s in W.shape)
        Wp = W.copy(); Wp[i, j, k, l] += eps
        Wm = W.copy(); Wm[i, j, k, l] -= eps
        Zp, _ = conv_forward(A_prev, Wp, b, hp)
        Zm, _ = conv_forward(A_prev, Wm, b, hp)
        num = (Zp.sum() - Zm.sum()) / (2 * eps)
        assert abs(num - dW[i, j, k, l]) < 1e-4

    # check a few random entries of dA_prev
    for _ in range(5):
        i, j, k, l = (np.random.randint(s) for s in A_prev.shape)
        Ap = A_prev.copy(); Ap[i, j, k, l] += eps
        Am = A_prev.copy(); Am[i, j, k, l] -= eps
        Zp, _ = conv_forward(Ap, W, b, hp)
        Zm, _ = conv_forward(Am, W, b, hp)
        num = (Zp.sum() - Zm.sum()) / (2 * eps)
        assert abs(num - dA_prev[i, j, k, l]) < 1e-4


def test_pool_forward_max_and_avg():
    np.random.seed(2)
    A_prev = np.random.randn(1, 4, 4, 2)
    Amax, _ = pool_forward(A_prev, {"stride": 2, "f": 2}, mode="max")
    Aavg, _ = pool_forward(A_prev, {"stride": 2, "f": 2}, mode="average")
    assert Amax.shape == (1, 2, 2, 2)
    # top-left max equals max over the 2x2x(channel) window
    assert np.isclose(Amax[0, 0, 0, 0], A_prev[0, :2, :2, 0].max())
    assert np.isclose(Aavg[0, 0, 0, 0], A_prev[0, :2, :2, 0].mean())


def test_pool_backward_max_routes_gradient():
    np.random.seed(3)
    A_prev = np.random.randn(1, 4, 4, 1)
    A, cache = pool_forward(A_prev, {"stride": 2, "f": 2}, mode="max")
    dA = np.ones_like(A)
    dA_prev = pool_backward(dA, cache, mode="max")
    # exactly one 1 per 2x2 window (the argmax), rest 0
    assert dA_prev.sum() == 4  # 4 windows, one unit each
    assert set(np.unique(dA_prev)).issubset({0.0, 1.0})
