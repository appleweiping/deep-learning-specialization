"""C2 W1 - Initialization.

Compares three weight initializations (zeros / large-random / He) on the
2-class 'flower' dataset, showing that:
  * zeros  -> symmetry never breaks, network stuck at ~50%
  * random(*10) -> exploding start, slow/poor convergence
  * He     -> fast convergence and best accuracy for ReLU nets.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import nn_numpy as nn  # noqa: E402
from common.datasets import load_planar_dataset  # noqa: E402


def main():
    X, Y = load_planar_dataset(seed=3)
    X, Y = X.astype(float), Y.astype(float)
    layer_dims = [X.shape[0], 10, 5, 1]

    results = {}
    for init in ["zeros", "random", "he"]:
        params, costs = nn.model(X, Y, layer_dims, learning_rate=0.5,
                                 num_iterations=8000, init=init, seed=3)
        acc = nn.accuracy(X, Y, params) * 100
        results[init] = {"acc": acc, "final_cost": costs[-1]}
        print(f"init={init:7s}  train acc {acc:6.2f}%  final cost {costs[-1]:.4f}")
    return results


if __name__ == "__main__":
    main()
