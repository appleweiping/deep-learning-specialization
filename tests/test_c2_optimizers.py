"""Tests for C2 optimizer / regularization behaviours."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import nn_numpy as nn


def test_adam_bias_correction_first_step():
    """On the first Adam step with zero-init moments, the update magnitude
    should be ~learning_rate * sign(grad) thanks to bias correction."""
    params = {"W1": np.zeros((1, 1)), "b1": np.zeros((1, 1))}
    grads = {"dW1": np.array([[3.0]]), "db1": np.array([[0.0]])}
    v, s = nn.initialize_adam(params)
    lr = 0.1
    nn.update_parameters_adam(params, grads, v, s, t=1, learning_rate=lr)
    # bias-corrected first step: v_hat/sqrt(s_hat) = grad/|grad| = 1
    assert abs(params["W1"][0, 0] - (-lr)) < 1e-6


def test_momentum_accumulates():
    params = {"W1": np.zeros((1, 1)), "b1": np.zeros((1, 1))}
    v = nn.initialize_velocity(params)
    grads = {"dW1": np.array([[1.0]]), "db1": np.array([[0.0]])}
    # constant gradient -> velocity grows toward grad
    for _ in range(50):
        nn.update_parameters_momentum(params, grads, v, beta=0.9, learning_rate=0.0)
    assert v["dW1"][0, 0] > 0.99  # (1-beta^50) ~ 1


def test_l2_shrinks_weights():
    np.random.seed(0)
    X = np.random.randn(2, 200)
    Y = (X.sum(0) > 0).astype(float).reshape(1, -1)
    p_none, _ = nn.model(X, Y, [2, 20, 1], learning_rate=0.3,
                         num_iterations=1500, init="he", lambd=0.0, seed=0)
    p_l2, _ = nn.model(X, Y, [2, 20, 1], learning_rate=0.3,
                       num_iterations=1500, init="he", lambd=2.0, seed=0)
    norm_none = np.linalg.norm(p_none["W1"])
    norm_l2 = np.linalg.norm(p_l2["W1"])
    assert norm_l2 < norm_none


def test_dropout_changes_forward():
    np.random.seed(0)
    X = np.random.randn(4, 10)
    params = nn.initialize_parameters([4, 8, 1], mode="he", seed=0)
    A_full, _, _ = nn.forward_propagation(X, params, keep_prob=1.0)
    A_drop, _, masks = nn.forward_propagation(X, params, keep_prob=0.5, seed=1)
    assert masks[0] is not None
    assert not np.allclose(A_full, A_drop)
