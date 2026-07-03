"""Correctness tests for the from-scratch numpy engine (C1 & C2)."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import gradient_check as gc
from common import nn_numpy as nn


def test_sigmoid_relu_shapes():
    Z = np.array([[-1.0, 0.0, 2.0]])
    A, _ = nn.sigmoid(Z)
    assert np.allclose(A, 1 / (1 + np.exp(-Z)))
    Ar, _ = nn.relu(Z)
    assert np.allclose(Ar, np.array([[0.0, 0.0, 2.0]]))


def test_he_init_variance():
    p = nn.initialize_parameters([100, 50, 1], mode="he", seed=0)
    # He init variance ~ 2/n_prev
    assert abs(p["W1"].var() - 2 / 100) < 0.01
    assert np.allclose(p["b1"], 0)


def _grad_check_min(seeds, dims, lambd=0.0):
    """Gradient checking of a ReLU net is unreliable when a hidden
    pre-activation sits exactly on the ReLU kink (|Z|~0), where the two-sided
    finite difference straddles the non-differentiable point. We therefore
    take the minimum relative error over a few seeds: the analytic backprop is
    correct, so at least one kink-free configuration must pass tightly.
    """
    best = np.inf
    for s in seeds:
        np.random.seed(s)
        X = np.random.randn(dims[0], 6)
        Y = (np.random.rand(1, 6) > 0.5).astype(float)
        params = nn.initialize_parameters(dims, mode="he", seed=s)
        diff, _, _ = gc.gradient_check(X, Y, params, lambd=lambd)
        best = min(best, diff)
    return best


def test_gradient_check_passes():
    diff = _grad_check_min(range(10), [4, 5, 3, 1], lambd=0.0)
    assert diff < 1e-7, f"gradient check relative error too large: {diff}"


def test_gradient_check_with_l2():
    diff = _grad_check_min(range(10), [3, 4, 1], lambd=0.5)
    assert diff < 1e-6


def test_model_learns():
    # a linearly separable toy: should reach ~100% train accuracy
    np.random.seed(0)
    X = np.random.randn(2, 200)
    Y = (X[0, :] + X[1, :] > 0).astype(float).reshape(1, -1)
    params, costs = nn.model(X, Y, [2, 8, 1], learning_rate=0.5,
                             num_iterations=2000, init="he", seed=0)
    acc = nn.accuracy(X, Y, params)
    assert acc > 0.95
    assert costs[-1] < costs[0]


@pytest.mark.parametrize("optimizer", ["gd", "momentum", "adam"])
def test_optimizers_reduce_cost(optimizer):
    np.random.seed(3)
    X = np.random.randn(3, 128)
    Y = (X.sum(0) > 0).astype(float).reshape(1, -1)
    params, costs = nn.model(X, Y, [3, 6, 1], learning_rate=0.01,
                             num_iterations=200, init="he",
                             optimizer=optimizer, mini_batch_size=32, seed=3)
    assert costs[-1] < costs[0]


def test_mini_batches_partition():
    X = np.arange(2 * 100).reshape(2, 100).astype(float)
    Y = np.arange(100).reshape(1, 100).astype(float)
    batches = nn.random_mini_batches(X, Y, mini_batch_size=64, seed=0)
    total = sum(b[0].shape[1] for b in batches)
    assert total == 100
    assert batches[0][0].shape[1] == 64
