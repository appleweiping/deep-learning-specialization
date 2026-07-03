"""C1 W3 - Planar data classification with one hidden layer.

A 2-class classifier with a single tanh hidden layer, implemented from
scratch (forward/backward, cross-entropy). Trained on the 'flower' petal
dataset. Compares against a linear (logistic-regression) baseline and
sweeps hidden-layer size.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.datasets import load_planar_dataset  # noqa: E402


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def layer_sizes(X, Y, n_h):
    return X.shape[0], n_h, Y.shape[0]


def initialize_parameters(n_x, n_h, n_y, seed=2):
    np.random.seed(seed)
    return {
        "W1": np.random.randn(n_h, n_x) * 0.01,
        "b1": np.zeros((n_h, 1)),
        "W2": np.random.randn(n_y, n_h) * 0.01,
        "b2": np.zeros((n_y, 1)),
    }


def forward_propagation(X, params):
    Z1 = params["W1"] @ X + params["b1"]
    A1 = np.tanh(Z1)
    Z2 = params["W2"] @ A1 + params["b2"]
    A2 = sigmoid(Z2)
    cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
    return A2, cache


def compute_cost(A2, Y):
    m = Y.shape[1]
    eps = 1e-12
    logprobs = Y * np.log(A2 + eps) + (1 - Y) * np.log(1 - A2 + eps)
    return float(np.squeeze(-np.sum(logprobs) / m))


def backward_propagation(params, cache, X, Y):
    m = X.shape[1]
    A1, A2 = cache["A1"], cache["A2"]
    dZ2 = A2 - Y
    dW2 = (dZ2 @ A1.T) / m
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m
    dZ1 = (params["W2"].T @ dZ2) * (1 - np.power(A1, 2))
    dW1 = (dZ1 @ X.T) / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m
    return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}


def update_parameters(params, grads, learning_rate=1.2):
    for k in ("W1", "b1", "W2", "b2"):
        params[k] -= learning_rate * grads["d" + k]
    return params


def nn_model(X, Y, n_h, num_iterations=10000, learning_rate=1.2,
             print_every=0, seed=2):
    n_x, _, n_y = layer_sizes(X, Y, n_h)
    params = initialize_parameters(n_x, n_h, n_y, seed)
    costs = []
    for i in range(num_iterations):
        A2, cache = forward_propagation(X, params)
        cost = compute_cost(A2, Y)
        grads = backward_propagation(params, cache, X, Y)
        params = update_parameters(params, grads, learning_rate)
        if print_every and i % print_every == 0:
            print(f"iter {i:5d}  cost {cost:.6f}")
        if i % 1000 == 0:
            costs.append(cost)
    return params, costs


def predict(params, X):
    A2, _ = forward_propagation(X, params)
    return (A2 > 0.5).astype(int)


def accuracy(params, X, Y):
    p = predict(params, X)
    correct = np.dot(Y, p.T) + np.dot(1 - Y, 1 - p.T)
    return float(np.squeeze(correct) / float(Y.size) * 100)


def main():
    X, Y = load_planar_dataset()
    X, Y = X.astype(float), Y.astype(float)

    # single hidden layer, n_h = 4
    params, costs = nn_model(X, Y, n_h=4, num_iterations=10000,
                             learning_rate=1.2, print_every=2000)
    acc = accuracy(params, X, Y)
    print(f"1-hidden-layer (n_h=4) accuracy: {acc:.1f}%")

    print("\nHidden layer size sweep:")
    results = {}
    for n_h in [1, 2, 4, 8, 20]:
        p, _ = nn_model(X, Y, n_h=n_h, num_iterations=6000, learning_rate=1.2)
        a = accuracy(p, X, Y)
        results[n_h] = a
        print(f"  n_h={n_h:3d}  accuracy {a:.1f}%")

    return {"acc_nh4": acc, "sweep": results}


if __name__ == "__main__":
    main()
