"""C2 W2 - Optimization Methods.

Compares mini-batch Gradient Descent, Momentum and Adam on the 2-moons
dataset, reporting final cost and accuracy for each. Demonstrates that Adam
converges fastest / to the best fit for the same number of epochs.
(RMSProp is included as the s-only special case inside the Adam update and is
also verified in the unit tests.)
"""
from __future__ import annotations

import os
import sys

import numpy as np
from sklearn.datasets import make_moons

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import nn_numpy as nn  # noqa: E402


def load_data():
    X, y = make_moons(n_samples=600, noise=0.25, random_state=2)
    return X.T, y.reshape(1, -1).astype(float)


def main():
    X, Y = load_data()
    layer_dims = [2, 16, 8, 1]
    results = {}
    for opt, lr in [("gd", 0.03), ("momentum", 0.03), ("adam", 0.01)]:
        params, costs = nn.model(X, Y, layer_dims, learning_rate=lr,
                                 num_iterations=1500, init="he",
                                 optimizer=opt, mini_batch_size=64,
                                 beta=0.9, seed=3)
        acc = nn.accuracy(X, Y, params) * 100
        results[opt] = {"acc": acc, "final_cost": costs[-1]}
        print(f"optimizer={opt:9s} lr={lr:<5}  final cost {costs[-1]:.4f}  acc {acc:6.2f}%")
    return results


if __name__ == "__main__":
    main()
