"""C4 W3 - Image Segmentation with U-Net.

Implements the U-Net architecture (Ronneberger et al., 2015): a contracting
encoder, an expansive decoder, and skip connections that concatenate encoder
feature maps into the decoder. Trained for semantic segmentation on a
procedurally-generated shapes dataset (real per-pixel labels), which mirrors
the pixel-labelling task of the original CARLA self-driving assignment while
running on CPU.

Reports mean pixel accuracy and mean IoU on a held-out test set.
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
torch.set_num_threads(3)


# --------------------------------------------------------------------------- #
# Synthetic segmentation data: background(0), circle(1), rectangle(2)
# --------------------------------------------------------------------------- #
def make_segmentation_dataset(n, size=64, seed=0):
    rng = np.random.RandomState(seed)
    X = np.zeros((n, 3, size, size), dtype=np.float32)
    Y = np.zeros((n, size, size), dtype=np.int64)
    yy, xx = np.mgrid[0:size, 0:size]
    for i in range(n):
        # textured background
        bg = rng.rand(3, size, size).astype(np.float32) * 0.3
        img = bg.copy()
        mask = np.zeros((size, size), dtype=np.int64)
        # a circle
        cy, cx = rng.randint(size // 4, 3 * size // 4, size=2)
        r = rng.randint(size // 8, size // 4)
        circ = (yy - cy) ** 2 + (xx - cx) ** 2 < r ** 2
        img[0][circ] = 0.9; img[1][circ] = 0.2; img[2][circ] = 0.2
        mask[circ] = 1
        # a rectangle
        ry, rx = rng.randint(0, size // 2, size=2)
        rh, rw = rng.randint(size // 6, size // 3, size=2)
        rect = np.zeros((size, size), dtype=bool)
        rect[ry:ry + rh, rx:rx + rw] = True
        img[0][rect] = 0.2; img[1][rect] = 0.2; img[2][rect] = 0.9
        mask[rect] = 2
        X[i] = img
        Y[i] = mask
    return X, Y


# --------------------------------------------------------------------------- #
# U-Net
# --------------------------------------------------------------------------- #
def double_conv(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self, n_classes=3, base=16):
        super().__init__()
        self.enc1 = double_conv(3, base)
        self.enc2 = double_conv(base, base * 2)
        self.enc3 = double_conv(base * 2, base * 4)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = double_conv(base * 4, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = double_conv(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = double_conv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = double_conv(base * 2, base)
        self.out = nn.Conv2d(base, n_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out(d1)


def mean_iou(pred, target, n_classes):
    ious = []
    for c in range(n_classes):
        p = pred == c
        t = target == c
        inter = (p & t).sum().item()
        union = (p | t).sum().item()
        if union > 0:
            ious.append(inter / union)
    return float(np.mean(ious)) if ious else 0.0


def main(n_train=400, n_test=100, epochs=8):
    Xtr, Ytr = make_segmentation_dataset(n_train, seed=0)
    Xte, Yte = make_segmentation_dataset(n_test, seed=99)
    Xtr_t, Ytr_t = torch.tensor(Xtr), torch.tensor(Ytr)
    Xte_t, Yte_t = torch.tensor(Xte), torch.tensor(Yte)
    print(f"segmentation data: train {Xtr.shape}, test {Xte.shape}, 3 classes")

    model = UNet(n_classes=3)
    print(f"U-Net params: {sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    batch = 32
    n = Xtr_t.shape[0]
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        t0 = time.time()
        total = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            xb, yb = Xtr_t[idx], Ytr_t[idx]
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * xb.shape[0]
        model.eval()
        with torch.no_grad():
            logits = model(Xte_t)
            pred = logits.argmax(1)
            pix_acc = (pred == Yte_t).float().mean().item()
            iou = mean_iou(pred, Yte_t, 3)
        print(f"epoch {ep+1}/{epochs}  loss {total/n:.4f}  "
              f"pixel-acc {pix_acc*100:5.2f}%  mIoU {iou:.3f}  ({time.time()-t0:.1f}s)")

    # save a sample prediction figure
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "results", "C4")
    os.makedirs(out_dir, exist_ok=True)
    _save_sample(model, Xte_t, Yte_t, out_dir)
    return {"pixel_acc": pix_acc * 100, "mIoU": iou}


def _save_sample(model, X, Y, out_dir, k=4):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    model.eval()
    with torch.no_grad():
        pred = model(X[:k]).argmax(1).cpu().numpy()
    fig, axes = plt.subplots(3, k, figsize=(3 * k, 9))
    for j in range(k):
        axes[0, j].imshow(np.transpose(X[j].cpu().numpy(), (1, 2, 0)))
        axes[0, j].set_title("input"); axes[0, j].axis("off")
        axes[1, j].imshow(Y[j].cpu().numpy(), vmin=0, vmax=2)
        axes[1, j].set_title("ground truth"); axes[1, j].axis("off")
        axes[2, j].imshow(pred[j], vmin=0, vmax=2)
        axes[2, j].set_title("U-Net prediction"); axes[2, j].axis("off")
    fig.tight_layout()
    path = os.path.join(out_dir, "unet_segmentation_sample.png")
    fig.savefig(path, dpi=100)
    print(f"saved {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--n-train", type=int, default=400)
    ap.add_argument("--n-test", type=int, default=100)
    args = ap.parse_args()
    main(args.n_train, args.n_test, args.epochs)
