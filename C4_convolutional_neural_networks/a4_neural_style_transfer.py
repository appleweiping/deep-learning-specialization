"""C4 W4 - Art Generation with Neural Style Transfer (Gatys et al., 2015).

Optimizes a generated image so that its *content* matches a content image
(deep VGG feature activations) and its *style* matches a style image (Gram
matrices of shallow/mid VGG features). Implemented in PyTorch with a
pretrained VGG19 backbone.

If no content/style images are supplied, deterministic synthetic images are
generated (a shape "content" and a colorful stripe "style") so the script is
fully self-contained; the produced image is written to results/C4/.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
torch.set_num_threads(3)

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def make_synthetic_images(size=160):
    """Deterministic content (concentric shapes) and style (color stripes)."""
    yy, xx = np.mgrid[0:size, 0:size]
    # content: a bright disc + square on gray
    content = np.ones((3, size, size), dtype=np.float32) * 0.5
    disc = (yy - size * 0.4) ** 2 + (xx - size * 0.4) ** 2 < (size * 0.22) ** 2
    content[0][disc] = 0.9; content[1][disc] = 0.9; content[2][disc] = 0.2
    sq = (np.abs(yy - size * 0.7) < size * 0.18) & (np.abs(xx - size * 0.65) < size * 0.18)
    content[0][sq] = 0.2; content[1][sq] = 0.4; content[2][sq] = 0.9

    # style: diagonal rainbow stripes
    style = np.zeros((3, size, size), dtype=np.float32)
    phase = ((xx + yy) / size * 3.0)
    style[0] = 0.5 + 0.5 * np.sin(phase * 2.0)
    style[1] = 0.5 + 0.5 * np.sin(phase * 2.0 + 2.0)
    style[2] = 0.5 + 0.5 * np.sin(phase * 2.0 + 4.0)
    return torch.tensor(content)[None], torch.tensor(style)[None]


def load_image(path, size=160):
    from PIL import Image
    img = Image.open(path).convert("RGB").resize((size, size))
    arr = np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
    return torch.tensor(arr)[None]


def normalize(x):
    return (x - IMAGENET_MEAN) / IMAGENET_STD


def gram_matrix(feat):
    b, c, h, w = feat.shape
    f = feat.view(c, h * w)
    return (f @ f.t()) / (c * h * w)


class VGGFeatures(nn.Module):
    """Extract activations at selected VGG19 layers."""

    CONTENT_LAYER = 21  # conv4_2
    STYLE_LAYERS = [0, 5, 10, 19, 28]  # conv1_1 ... conv5_1

    def __init__(self):
        super().__init__()
        from torchvision.models import VGG19_Weights, vgg19
        vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features.eval()
        for p in vgg.parameters():
            p.requires_grad_(False)
        self.vgg = vgg

    def forward(self, x):
        content_feat = None
        style_feats = []
        h = x
        for i, layer in enumerate(self.vgg):
            h = layer(h)
            if i in self.STYLE_LAYERS:
                style_feats.append(h)
            if i == self.CONTENT_LAYER:
                content_feat = h
            if i >= max(self.CONTENT_LAYER, max(self.STYLE_LAYERS)):
                break
        return content_feat, style_feats


def run(content, style, steps=200, style_weight=1e6, content_weight=1.0, lr=0.05):
    model = VGGFeatures()
    with torch.no_grad():
        c_target, _ = model(normalize(content))
        _, s_targets = model(normalize(style))
        s_grams = [gram_matrix(s).detach() for s in s_targets]

    gen = content.clone().requires_grad_(True)
    opt = torch.optim.Adam([gen], lr=lr)

    history = []
    for step in range(steps):
        opt.zero_grad()
        c_feat, s_feats = model(normalize(gen.clamp(0, 1)))
        c_loss = F.mse_loss(c_feat, c_target)
        s_loss = sum(F.mse_loss(gram_matrix(sf), sg)
                     for sf, sg in zip(s_feats, s_grams))
        loss = content_weight * c_loss + style_weight * s_loss
        loss.backward()
        opt.step()
        with torch.no_grad():
            gen.clamp_(0, 1)
        if step % 40 == 0 or step == steps - 1:
            history.append((step, float(c_loss), float(s_loss), float(loss)))
            print(f"step {step:4d}  content {c_loss:.4f}  style {s_loss:.6f}  total {loss:.4f}")
    return gen.detach(), history


def save_result(content, style, gen, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def to_img(t):
        return np.clip(t[0].cpu().numpy().transpose(1, 2, 0), 0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, img, title in zip(axes, [content, style, gen],
                              ["content", "style", "generated"]):
        ax.imshow(to_img(img)); ax.set_title(title); ax.axis("off")
    fig.tight_layout()
    path = os.path.join(out_dir, "neural_style_transfer.png")
    fig.savefig(path, dpi=110)
    print(f"saved {path}")


def main(steps=200, content_path=None, style_path=None):
    if content_path and style_path:
        content = load_image(content_path)
        style = load_image(style_path)
    else:
        content, style = make_synthetic_images()
    gen, history = run(content, style, steps=steps)
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "results", "C4")
    os.makedirs(out_dir, exist_ok=True)
    save_result(content, style, gen, out_dir)
    return {"history": history}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--content", type=str, default=None)
    ap.add_argument("--style", type=str, default=None)
    args = ap.parse_args()
    main(args.steps, args.content, args.style)
