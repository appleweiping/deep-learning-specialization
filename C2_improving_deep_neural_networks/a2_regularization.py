"""C2 W1 - Regularization (L2 and dropout).

Trains the same deep net on a noisy 2D dataset (2 moons) with:
  * no regularization  (overfits: high train / lower test)
  * L2 regularization  (shrinks weights, improves test)
  * inverted dropout   (randomly drops units, improves test)
and reports train/test accuracy for each.
"""
from __future__ import annotations

import os
import sys

import numpy as np
from sklearn.datasets import make_moons

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import nn_numpy as nn  # noqa: E402


def load_noisy():
    # Small, noisy training set + a big over-parameterized net -> the
    # unregularized model overfits, so L2 / dropout visibly help on test.
    X, y = make_moons(n_samples=500, noise=0.30, random_state=0)
    Xtr, ytr = X[:120], y[:120]
    Xte, yte = X[120:], y[120:]
    return (Xtr.T, ytr.reshape(1, -1).astype(float),
            Xte.T, yte.reshape(1, -1).astype(float))


def main():
    Xtr, Ytr, Xte, Yte = load_noisy()
    layer_dims = [2, 60, 40, 1]
    configs = {
        "none": dict(lambd=0.0, keep_prob=1.0),
        "L2 (lambda=0.3)": dict(lambd=0.3, keep_prob=1.0),
        "dropout (keep=0.7)": dict(lambd=0.0, keep_prob=0.7),
    }
    results = {}
    for name, cfg in configs.items():
        params, costs = nn.model(Xtr, Ytr, layer_dims, learning_rate=0.3,
                                 num_iterations=12000, init="he", seed=3, **cfg)
        tr = nn.accuracy(Xtr, Ytr, params) * 100
        te = nn.accuracy(Xte, Yte, params) * 100
        results[name] = {"train": tr, "test": te}
        print(f"{name:20s}  train {tr:6.2f}%  test {te:6.2f}%")
    return results


if __name__ == "__main__":
    main()
