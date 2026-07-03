"""C1 W2 - Logistic Regression with a Neural Network mindset.

Logistic regression built as a single-neuron network from scratch:
forward propagation (sigmoid), cross-entropy cost, backward propagation,
and gradient-descent optimization. Trained on a synthetic 64x64x3 binary
image dataset that mirrors the Coursera cat/non-cat task.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.datasets import load_mnist  # noqa: E402


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def initialize_with_zeros(dim):
    return np.zeros((dim, 1)), 0.0


def propagate(w, b, X, Y):
    """Forward + backward. Returns grads dict and scalar cost."""
    m = X.shape[1]
    A = sigmoid(w.T @ X + b)
    eps = 1e-12
    cost = -np.sum(Y * np.log(A + eps) + (1 - Y) * np.log(1 - A + eps)) / m
    dw = (X @ (A - Y).T) / m
    db = np.sum(A - Y) / m
    return {"dw": dw, "db": db}, float(np.squeeze(cost))


def optimize(w, b, X, Y, num_iterations, learning_rate, print_every=0):
    costs = []
    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y)
        w = w - learning_rate * grads["dw"]
        b = b - learning_rate * grads["db"]
        if i % 100 == 0:
            costs.append(cost)
            if print_every and i % print_every == 0:
                print(f"iter {i:4d}  cost {cost:.6f}")
    return w, b, costs


def predict(w, b, X):
    A = sigmoid(w.T @ X + b)
    return (A > 0.5).astype(int)


def logistic_regression_model(X_train, Y_train, X_test, Y_test,
                              num_iterations=2000, learning_rate=0.005,
                              print_every=0):
    w, b = initialize_with_zeros(X_train.shape[0])
    w, b, costs = optimize(w, b, X_train, Y_train, num_iterations,
                           learning_rate, print_every)
    train_acc = 100 - np.mean(np.abs(predict(w, b, X_train) - Y_train)) * 100
    test_acc = 100 - np.mean(np.abs(predict(w, b, X_test) - Y_test)) * 100
    return {"w": w, "b": b, "costs": costs,
            "train_acc": float(train_acc), "test_acc": float(test_acc)}


def main():
    np.random.seed(1)
    # Real binary task: MNIST digit 3 vs digit 5 (a genuinely hard pair).
    Xtr, ytr, Xte, yte = load_mnist(n_train=2000, n_test=1000, digits=(3, 5),
                                    flatten=True)
    # relabel to {0,1}: 3->0, 5->1
    ytr = (ytr == 5).astype(int)
    yte = (yte == 5).astype(int)
    X_train = Xtr.T          # already scaled to [0,1] by the loader
    X_test = Xte.T
    Y_train = ytr.reshape(1, -1)
    Y_test = yte.reshape(1, -1)
    print(f"train X {X_train.shape}, test X {X_test.shape}")

    out = logistic_regression_model(X_train, Y_train, X_test, Y_test,
                                    num_iterations=2000, learning_rate=0.05,
                                    print_every=200)
    print(f"Train accuracy: {out['train_acc']:.2f}%")
    print(f"Test  accuracy: {out['test_acc']:.2f}%")
    return out


if __name__ == "__main__":
    main()
