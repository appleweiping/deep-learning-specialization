"""C2 W3 - Deep-learning framework intro.

The original assignment is a TensorFlow tutorial that builds a small
fully-connected classifier with automatic differentiation and an Adam
optimizer. The shared csdiy environment ships PyTorch (CPU) rather than
TensorFlow, so this is the faithful PyTorch equivalent: it exercises the same
ideas the lab teaches -- tensors, a parameterized model, an autodiff-based
training loop, mini-batches and the Adam optimizer -- on a real dataset
(MNIST, 6-class subset) instead of the SIGNS hand-gesture set.

Everything the from-scratch numpy engine does by hand (forward/backward,
Adam, mini-batches) is here delegated to the framework, which is the point of
the assignment: showing the same result with far less code.
"""
from __future__ import annotations

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


class MLP(nn.Module):
    """25 -> 12 -> n_classes fully-connected net, mirroring the lab's shape
    scaled to the input dimensionality."""

    def __init__(self, in_dim, n_classes):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, n_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def main():
    classes = (0, 1, 2, 3, 4, 5)
    Xtr, ytr, Xte, yte = load_mnist(n_train=6000, n_test=2000,
                                    digits=classes, flatten=True)
    remap = {c: i for i, c in enumerate(classes)}
    ytr = np.vectorize(remap.get)(ytr)
    yte = np.vectorize(remap.get)(yte)

    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.long)
    Xte_t = torch.tensor(Xte, dtype=torch.float32)
    yte_t = torch.tensor(yte, dtype=torch.long)

    model = MLP(Xtr.shape[1], len(classes))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    n = Xtr_t.shape[0]
    batch = 64
    epochs = 8
    for ep in range(epochs):
        perm = torch.randperm(n)
        total = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            xb, yb = Xtr_t[idx], ytr_t[idx]
            optimizer.zero_grad()
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            optimizer.step()
            total += loss.item() * xb.shape[0]
        with torch.no_grad():
            tr_acc = (model(Xtr_t).argmax(1) == ytr_t).float().mean().item()
            te_acc = (model(Xte_t).argmax(1) == yte_t).float().mean().item()
        print(f"epoch {ep+1}/{epochs}  loss {total/n:.4f}  "
              f"train acc {tr_acc*100:5.2f}%  test acc {te_acc*100:5.2f}%")

    with torch.no_grad():
        final_te = (model(Xte_t).argmax(1) == yte_t).float().mean().item() * 100
    print(f"Final test accuracy: {final_te:.2f}%")
    return {"test_acc": final_te}


if __name__ == "__main__":
    main()
