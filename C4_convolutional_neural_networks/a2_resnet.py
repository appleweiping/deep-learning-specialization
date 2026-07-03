"""C4 W2 - Residual Networks.

Implements the two building blocks of a ResNet (He et al., 2015) in PyTorch:
  * identity_block         : skip connection when input/output shapes match
  * convolutional_block    : skip connection with a 1x1 conv to match shapes
and assembles a small ResNet trained on a CIFAR-10 subset (CPU-scale but real).

Mirrors the ResNet50-style block structure from the Coursera assignment
(CONV -> BN -> ReLU stages with an additive shortcut) at a depth that trains
in a few minutes on CPU.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
torch.set_num_threads(3)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.datasets import load_cifar10  # noqa: E402


class IdentityBlock(nn.Module):
    """Skip connection where in/out channels match: out = ReLU(F(x) + x)."""

    def __init__(self, channels, bottleneck):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, bottleneck, 1)
        self.bn1 = nn.BatchNorm2d(bottleneck)
        self.conv2 = nn.Conv2d(bottleneck, bottleneck, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(bottleneck)
        self.conv3 = nn.Conv2d(bottleneck, channels, 1)
        self.bn3 = nn.BatchNorm2d(channels)

    def forward(self, x):
        shortcut = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        return F.relu(x + shortcut)


class ConvolutionalBlock(nn.Module):
    """Skip connection with a 1x1 conv on the shortcut to match channel/stride."""

    def __init__(self, in_ch, bottleneck, out_ch, stride=2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, bottleneck, 1, stride=stride)
        self.bn1 = nn.BatchNorm2d(bottleneck)
        self.conv2 = nn.Conv2d(bottleneck, bottleneck, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(bottleneck)
        self.conv3 = nn.Conv2d(bottleneck, out_ch, 1)
        self.bn3 = nn.BatchNorm2d(out_ch)
        # shortcut path
        self.sc_conv = nn.Conv2d(in_ch, out_ch, 1, stride=stride)
        self.sc_bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        shortcut = self.sc_bn(self.sc_conv(x))
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        return F.relu(x + shortcut)


class ResNet(nn.Module):
    def __init__(self, n_classes=4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
        )
        self.stage1 = nn.Sequential(
            ConvolutionalBlock(32, 16, 64, stride=1),
            IdentityBlock(64, 16),
        )
        self.stage2 = nn.Sequential(
            ConvolutionalBlock(64, 32, 128, stride=2),
            IdentityBlock(128, 32),
        )
        self.stage3 = nn.Sequential(
            ConvolutionalBlock(128, 64, 256, stride=2),
            IdentityBlock(256, 64),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, n_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


def evaluate(model, X, y, batch=256):
    model.eval()
    correct = 0
    with torch.no_grad():
        for i in range(0, X.shape[0], batch):
            logits = model(X[i:i + batch])
            correct += (logits.argmax(1) == y[i:i + batch]).sum().item()
    return correct / X.shape[0]


def main(n_train=6000, n_test=2000, epochs=6):
    classes = (0, 1, 8, 9)  # airplane, automobile, ship, truck
    Xtr, ytr, Xte, yte = load_cifar10(n_train=n_train, n_test=n_test, classes=classes)
    Xtr_t = torch.tensor(Xtr); ytr_t = torch.tensor(ytr)
    Xte_t = torch.tensor(Xte); yte_t = torch.tensor(yte)
    print(f"CIFAR-10 subset {classes}: train {Xtr.shape}, test {Xte.shape}")

    model = ResNet(n_classes=len(classes))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"ResNet params: {n_params:,}")
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    batch = 128
    n = Xtr_t.shape[0]
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        t0 = time.time()
        total = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            xb, yb = Xtr_t[idx], ytr_t[idx]
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * xb.shape[0]
        te = evaluate(model, Xte_t, yte_t)
        print(f"epoch {ep+1}/{epochs}  loss {total/n:.4f}  "
              f"test acc {te*100:5.2f}%  ({time.time()-t0:.1f}s)")

    final = evaluate(model, Xte_t, yte_t)
    print(f"Final test accuracy: {final*100:.2f}%")
    return {"test_acc": final * 100, "params": n_params}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--n-train", type=int, default=6000)
    ap.add_argument("--n-test", type=int, default=2000)
    args = ap.parse_args()
    main(args.n_train, args.n_test, args.epochs)
