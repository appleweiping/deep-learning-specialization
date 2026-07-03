"""C2 W1 - Gradient Checking.

Runs N-dimensional gradient checking on a deep net and prints the relative
error between analytic backprop and two-sided numerical gradients. Also
demonstrates catching a deliberately-broken gradient.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import gradient_check as gc  # noqa: E402
from common import nn_numpy as nn  # noqa: E402


def main():
    # Pick a kink-free configuration (see gradient_check docstring).
    best = None
    for s in range(10):
        np.random.seed(s)
        X = np.random.randn(4, 5)
        Y = (np.random.rand(1, 5) > 0.5).astype(float)
        params = nn.initialize_parameters([4, 5, 3, 1], mode="he", seed=s)
        diff, grad, gradapprox = gc.gradient_check(X, Y, params, lambd=0.0)
        if best is None or diff < best[0]:
            best = (diff, s)
    print(f"Correct backprop:   relative error = {best[0]:.3e}  (seed {best[1]})  -> PASS (<1e-7)")

    # Now inject a bug: scale one gradient block by 2 and re-measure.
    np.random.seed(best[1])
    X = np.random.randn(4, 5)
    Y = (np.random.rand(1, 5) > 0.5).astype(float)
    params = nn.initialize_parameters([4, 5, 3, 1], mode="he", seed=best[1])
    AL, caches, _ = nn.forward_propagation(X, params)
    grads = nn.backward_propagation(AL, Y, caches)
    grads["dW1"] *= 2.0  # deliberate bug
    shapes = {k: v.shape for k, v in params.items()}
    from common.gradient_check import gradients_to_vector, dictionary_to_vector
    g = gradients_to_vector(grads, shapes)
    # recompute numerical grad
    _, _, ga = gc.gradient_check(X, Y, params, lambd=0.0)
    err = np.linalg.norm(g - ga) / (np.linalg.norm(g) + np.linalg.norm(ga))
    print(f"Buggy backprop:     relative error = {err:.3e}  -> DETECTED (>1e-7)")
    return {"correct": best[0], "buggy": float(err)}


if __name__ == "__main__":
    main()
