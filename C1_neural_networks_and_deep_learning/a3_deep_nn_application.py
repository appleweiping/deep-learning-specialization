"""C1 W4 - Building your Deep Neural Network: Step by Step + Application.

Uses the from-scratch L-layer numpy engine (common/nn_numpy.py) to train:
  (a) a 2-layer network  [LINEAR->RELU]->[LINEAR->SIGMOID]
  (b) a 4-layer deep network
on the synthetic 64x64x3 binary image dataset (cat / non-cat style),
demonstrating that depth improves the fit.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import nn_numpy as nn  # noqa: E402
from common.datasets import load_mnist  # noqa: E402


def load_flat():
    """MNIST 3-vs-5 binary task (real dataset), relabelled to {0,1}."""
    Xtr, ytr, Xte, yte = load_mnist(n_train=2000, n_test=1000, digits=(3, 5),
                                    flatten=True)
    ytr = (ytr == 5).astype(float).reshape(1, -1)
    yte = (yte == 5).astype(float).reshape(1, -1)
    return Xtr.T, ytr, Xte.T, yte


def main():
    X_train, Y_train, X_test, Y_test = load_flat()
    n_x = X_train.shape[0]
    print(f"train X {X_train.shape}, test X {X_test.shape}")

    # 2-layer
    print("\n=== 2-layer network [n_x, 7, 1] ===")
    p2, costs2 = nn.model(X_train, Y_train, [n_x, 7, 1],
                          learning_rate=0.0075, num_iterations=1500,
                          init="he", print_every=300, seed=1)
    tr2 = nn.accuracy(X_train, Y_train, p2) * 100
    te2 = nn.accuracy(X_test, Y_test, p2) * 100
    print(f"2-layer train acc {tr2:.2f}%  test acc {te2:.2f}%")

    # 4-layer deep net
    print("\n=== 4-layer deep network [n_x, 20, 7, 5, 1] ===")
    p4, costs4 = nn.model(X_train, Y_train, [n_x, 20, 7, 5, 1],
                          learning_rate=0.0075, num_iterations=1500,
                          init="he", print_every=300, seed=1)
    tr4 = nn.accuracy(X_train, Y_train, p4) * 100
    te4 = nn.accuracy(X_test, Y_test, p4) * 100
    print(f"4-layer train acc {tr4:.2f}%  test acc {te4:.2f}%")

    return {"two_layer": {"train": tr2, "test": te2},
            "four_layer": {"train": tr4, "test": te4}}


if __name__ == "__main__":
    main()
