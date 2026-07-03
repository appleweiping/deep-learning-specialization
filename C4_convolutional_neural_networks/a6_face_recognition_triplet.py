"""C4 W4 - Face Recognition with the triplet loss.

Trains a small convolutional embedding network with the triplet loss
    L = max(||f(A) - f(P)||^2 - ||f(A) - f(N)||^2 + alpha, 0)
so that images of the same identity are mapped close together and different
identities far apart (FaceNet, Schroff et al., 2015). Verification is then
thresholding the L2 distance between two embeddings.

Uses MNIST digits as stand-in "identities" (each digit class is one identity)
-- a real dataset that makes the triplet objective measurable on CPU. Reports
verification accuracy on unseen pairs.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
torch.set_num_threads(3)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.datasets import load_mnist  # noqa: E402


class EmbeddingNet(nn.Module):
    """Small CNN mapping a 28x28 image to a unit-norm 32-d embedding."""

    def __init__(self, dim=32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Linear(32 * 7 * 7, dim)

    def forward(self, x):
        h = self.conv(x).flatten(1)
        e = self.fc(h)
        return F.normalize(e, dim=1)  # L2-normalized embedding


def triplet_loss(anchor, positive, negative, alpha=0.2):
    pos = (anchor - positive).pow(2).sum(1)
    neg = (anchor - negative).pow(2).sum(1)
    return F.relu(pos - neg + alpha).mean()


def sample_triplets(X, y, n, rng):
    """Sample (anchor, positive, negative) index triplets."""
    classes = np.unique(y)
    by_class = {c: np.where(y == c)[0] for c in classes}
    A, P, N = [], [], []
    for _ in range(n):
        c = rng.choice(classes)
        a, p = rng.choice(by_class[c], size=2, replace=True)
        cn = rng.choice(classes[classes != c])
        nidx = rng.choice(by_class[cn])
        A.append(a); P.append(p); N.append(nidx)
    return np.array(A), np.array(P), np.array(N)


def verification_accuracy(model, X, y, rng, n_pairs=2000, threshold=None):
    """Build same/different pairs, embed, and threshold the L2 distance."""
    model.eval()
    classes = np.unique(y)
    by_class = {c: np.where(y == c)[0] for c in classes}
    pairs, labels = [], []
    for _ in range(n_pairs // 2):
        c = rng.choice(classes)
        i, j = rng.choice(by_class[c], size=2, replace=True)
        pairs.append((i, j)); labels.append(1)
        c2 = rng.choice(classes[classes != c])
        i2 = rng.choice(by_class[c]); j2 = rng.choice(by_class[c2])
        pairs.append((i2, j2)); labels.append(0)
    pairs = np.array(pairs); labels = np.array(labels)
    with torch.no_grad():
        emb = model(torch.tensor(X)).numpy()
    d = np.linalg.norm(emb[pairs[:, 0]] - emb[pairs[:, 1]], axis=1)
    if threshold is None:
        # pick the threshold that maximizes accuracy on this set
        ths = np.linspace(d.min(), d.max(), 100)
        accs = [((d < t).astype(int) == labels).mean() for t in ths]
        best = int(np.argmax(accs))
        return accs[best], ths[best]
    acc = ((d < threshold).astype(int) == labels).mean()
    return acc, threshold


def main(n_train=6000, n_test=2000, iters=400):
    Xtr, ytr, Xte, yte = load_mnist(n_train=n_train, n_test=n_test, flatten=False)
    Xtr = Xtr[:, None, :, :]  # add channel dim
    Xte = Xte[:, None, :, :]
    print(f"train {Xtr.shape}, test {Xte.shape}, {len(np.unique(ytr))} identities")

    model = EmbeddingNet()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    rng = np.random.RandomState(0)
    Xtr_t = torch.tensor(Xtr)

    batch = 128
    for it in range(iters):
        model.train()
        A, P, N = sample_triplets(Xtr, ytr, batch, rng)
        ea = model(Xtr_t[A]); ep = model(Xtr_t[P]); en = model(Xtr_t[N])
        loss = triplet_loss(ea, ep, en, alpha=0.2)
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 80 == 0 or it == iters - 1:
            acc, th = verification_accuracy(model, Xte, yte, rng)
            print(f"iter {it:4d}  triplet-loss {loss.item():.4f}  "
                  f"verification acc {acc*100:5.2f}% (thr {th:.3f})")

    acc, th = verification_accuracy(model, Xte, yte, rng)
    print(f"Final verification accuracy: {acc*100:.2f}%  (threshold {th:.3f})")
    return {"verification_acc": acc * 100, "threshold": float(th)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=400)
    args = ap.parse_args()
    main(iters=args.iters)
