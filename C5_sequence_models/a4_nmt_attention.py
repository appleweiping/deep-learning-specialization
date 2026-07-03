"""C5 W3 - Neural Machine Translation with Attention.

Reproduces the date-translation assignment: translate human-readable dates
("3 May 1979", "monday march 7 1983", ...) into the ISO format
("1979-05-03"). Uses a Bi-LSTM encoder with an additive (Bahdanau-style)
attention mechanism and an LSTM decoder, in PyTorch.

The dataset is generated deterministically (real date strings in many formats)
so the task is fully reproducible on CPU. Reports character-level accuracy on
a held-out test set.
"""
from __future__ import annotations

import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
torch.set_num_threads(3)

MONTHS = ["january", "february", "march", "april", "may", "june", "july",
          "august", "september", "october", "november", "december"]


def make_dataset(n, seed=0):
    rng = random.Random(seed)
    samples = []
    for _ in range(n):
        y = rng.randint(1950, 2020)
        m = rng.randint(1, 12)
        d = rng.randint(1, 28)
        target = f"{y:04d}-{m:02d}-{d:02d}"
        fmt = rng.randint(0, 4)
        mon = MONTHS[m - 1]
        if fmt == 0:
            human = f"{d} {mon} {y}"
        elif fmt == 1:
            human = f"{mon} {d} {y}"
        elif fmt == 2:
            human = f"{d}/{m}/{y}"
        elif fmt == 3:
            human = f"{mon[:3]} {d}, {y}"
        else:
            human = f"{d}.{m}.{y}"
        samples.append((human.lower(), target))
    return samples


def build_vocab(samples):
    src_chars = set()
    tgt_chars = set()
    for h, t in samples:
        src_chars.update(h)
        tgt_chars.update(t)
    src_chars = ["<pad>"] + sorted(src_chars)
    tgt_chars = ["<pad>", "<sos>"] + sorted(tgt_chars)
    return ({c: i for i, c in enumerate(src_chars)},
            {c: i for i, c in enumerate(tgt_chars)},
            {i: c for i, c in enumerate(tgt_chars)})


def encode(samples, src_vocab, tgt_vocab, Tx):
    X, Y = [], []
    for h, t in samples:
        xs = [src_vocab[c] for c in h[:Tx]]
        xs += [0] * (Tx - len(xs))
        ys = [tgt_vocab[c] for c in t]
        X.append(xs); Y.append(ys)
    return torch.tensor(X), torch.tensor(Y)


class Attention(nn.Module):
    def __init__(self, enc_dim, dec_dim, attn_dim=32):
        super().__init__()
        self.W = nn.Linear(enc_dim + dec_dim, attn_dim)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, enc_out, dec_h):
        # enc_out: (B, Tx, enc_dim), dec_h: (B, dec_dim)
        Tx = enc_out.shape[1]
        h_rep = dec_h.unsqueeze(1).repeat(1, Tx, 1)
        energy = torch.tanh(self.W(torch.cat([enc_out, h_rep], dim=2)))
        scores = self.v(energy).squeeze(2)          # (B, Tx)
        alpha = F.softmax(scores, dim=1)
        context = torch.bmm(alpha.unsqueeze(1), enc_out).squeeze(1)
        return context


class NMT(nn.Module):
    def __init__(self, src_size, tgt_size, Ty, emb=32, enc_h=32, dec_h=48):
        super().__init__()
        self.Ty = Ty
        self.tgt_size = tgt_size
        self.embed = nn.Embedding(src_size, emb, padding_idx=0)
        self.encoder = nn.LSTM(emb, enc_h, batch_first=True, bidirectional=True)
        self.attn = Attention(enc_h * 2, dec_h)
        self.decoder = nn.LSTMCell(enc_h * 2 + tgt_size, dec_h)
        self.out = nn.Linear(dec_h, tgt_size)
        self.dec_h = dec_h

    def forward(self, x, teacher=None):
        B = x.shape[0]
        enc_out, _ = self.encoder(self.embed(x))
        h = torch.zeros(B, self.dec_h)
        c = torch.zeros(B, self.dec_h)
        prev = torch.zeros(B, self.tgt_size)
        prev[:, 1] = 1.0  # <sos>
        logits_all = []
        for t in range(self.Ty):
            context = self.attn(enc_out, h)
            h, c = self.decoder(torch.cat([context, prev], dim=1), (h, c))
            logits = self.out(h)
            logits_all.append(logits)
            if teacher is not None:
                prev = F.one_hot(teacher[:, t], self.tgt_size).float()
            else:
                prev = F.one_hot(logits.argmax(1), self.tgt_size).float()
        return torch.stack(logits_all, dim=1)  # (B, Ty, tgt_size)


def main(n_train=8000, n_test=1000, epochs=12):
    train = make_dataset(n_train, seed=1)
    test = make_dataset(n_test, seed=999)
    src_vocab, tgt_vocab, tgt_inv = build_vocab(train + test)
    Tx = max(len(h) for h, _ in train + test)
    Ty = 10  # "YYYY-MM-DD"
    Xtr, Ytr = encode(train, src_vocab, tgt_vocab, Tx)
    Xte, Yte = encode(test, src_vocab, tgt_vocab, Tx)
    print(f"src vocab {len(src_vocab)}, tgt vocab {len(tgt_vocab)}, Tx {Tx}, Ty {Ty}")

    model = NMT(len(src_vocab), len(tgt_vocab), Ty)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    batch = 128
    n = Xtr.shape[0]

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        total = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            xb, yb = Xtr[idx], Ytr[idx]
            opt.zero_grad()
            logits = model(xb, teacher=yb)
            loss = F.cross_entropy(logits.reshape(-1, len(tgt_vocab)), yb.reshape(-1))
            loss.backward()
            opt.step()
            total += loss.item() * xb.shape[0]
        # eval (no teacher forcing)
        model.eval()
        with torch.no_grad():
            pred = model(Xte).argmax(2)
            char_acc = (pred == Yte).float().mean().item()
            exact = (pred == Yte).all(dim=1).float().mean().item()
        print(f"epoch {ep+1}/{epochs}  loss {total/n:.4f}  "
              f"char-acc {char_acc*100:5.2f}%  exact-date {exact*100:5.2f}%")

    # show some translations
    print("\nSample translations (human -> predicted ISO):")
    with torch.no_grad():
        pred = model(Xte[:8]).argmax(2)
    for k in range(8):
        s = "".join(tgt_inv[i.item()] for i in pred[k])
        print(f"  {test[k][0]:24s} -> {s}   (gold {test[k][1]})")
    return {"char_acc": char_acc * 100, "exact_date": exact * 100}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    args = ap.parse_args()
    main(epochs=args.epochs)
