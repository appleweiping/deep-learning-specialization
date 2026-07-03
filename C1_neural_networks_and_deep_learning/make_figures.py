"""Generate the decision-boundary figure for the planar classifier (C1 W3)."""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.datasets import load_planar_dataset  # noqa: E402

from a2_planar_classification import nn_model, predict  # noqa: E402


def plot_decision_boundary(pred_func, X, Y, ax, title):
    x_min, x_max = X[0, :].min() - 0.5, X[0, :].max() + 0.5
    y_min, y_max = X[1, :].min() - 0.5, X[1, :].max() + 0.5
    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()].T)
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.6)
    ax.scatter(X[0, :], X[1, :], c=Y.ravel(), s=12, cmap=plt.cm.Spectral, edgecolors="k", linewidths=0.3)
    ax.set_title(title)


def main():
    X, Y = load_planar_dataset()
    X, Y = X.astype(float), Y.astype(float)
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "results", "C1")
    os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, n_h in zip(axes, [1, 4, 20]):
        params, _ = nn_model(X, Y, n_h=n_h, num_iterations=8000, learning_rate=1.2)
        plot_decision_boundary(lambda x: predict(params, x), X, Y, ax,
                               f"n_h = {n_h}")
    fig.suptitle("Planar 'flower' classifier — decision boundary vs hidden units")
    fig.tight_layout()
    path = os.path.join(out_dir, "planar_decision_boundary.png")
    fig.savefig(path, dpi=110)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
