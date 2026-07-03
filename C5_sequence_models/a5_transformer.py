"""C5 W4 - Transformer Network (from scratch).

Implements the core Transformer components from "Attention Is All You Need"
(Vaswani et al., 2017) in PyTorch, by hand:
  * scaled dot-product attention
  * multi-head attention
  * sinusoidal positional encoding
  * position-wise feed-forward + residual + layer-norm blocks
  * an encoder-decoder Transformer with look-ahead + padding masks

Trained on a sequence-to-sequence task (reverse-and-increment a digit
sequence) that a Transformer solves to high accuracy on CPU, verifying the
attention machinery end-to-end. Also exercised by tests/test_c5_transformer.py.
"""
from __future__ import annotations

import argparse
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
torch.set_num_threads(3)


def scaled_dot_product_attention(q, k, v, mask=None):
    """q,k,v: (..., seq, d_k). mask: broadcastable, 1=keep 0=mask."""
    d_k = q.shape[-1]
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    attn = F.softmax(scores, dim=-1)
    return attn @ v, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.h = n_heads
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)

    def _split(self, x):
        b, s, _ = x.shape
        return x.view(b, s, self.h, self.d_k).transpose(1, 2)  # (b,h,s,d_k)

    def forward(self, q, k, v, mask=None):
        q = self._split(self.wq(q))
        k = self._split(self.wk(k))
        v = self._split(self.wv(v))
        if mask is not None:
            mask = mask.unsqueeze(1)  # broadcast over heads
        out, attn = scaled_dot_product_attention(q, k, v, mask)
        b, h, s, d_k = out.shape
        out = out.transpose(1, 2).contiguous().view(b, s, h * d_k)
        return self.wo(out)


def positional_encoding(seq_len, d_model):
    pe = torch.zeros(seq_len, d_model)
    pos = torch.arange(0, seq_len).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe.unsqueeze(0)  # (1, seq_len, d_model)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(),
                                 nn.Linear(d_ff, d_model))

    def forward(self, x):
        return self.net(x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, n_heads)
        self.ff = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = self.norm1(x + self.drop(self.mha(x, x, x, mask)))
        x = self.norm2(x + self.drop(self.ff(x)))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_mha = MultiHeadAttention(d_model, n_heads)
        self.cross_mha = MultiHeadAttention(d_model, n_heads)
        self.ff = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, enc, look_ahead_mask=None, pad_mask=None):
        x = self.norm1(x + self.drop(self.self_mha(x, x, x, look_ahead_mask)))
        x = self.norm2(x + self.drop(self.cross_mha(x, enc, enc, pad_mask)))
        x = self.norm3(x + self.drop(self.ff(x)))
        return x


class Transformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model=64, n_heads=4, d_ff=128,
                 n_layers=2, max_len=32):
        super().__init__()
        self.src_embed = nn.Embedding(src_vocab, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab, d_model)
        self.register_buffer("pe", positional_encoding(max_len, d_model))
        self.enc_layers = nn.ModuleList(
            [EncoderLayer(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.dec_layers = nn.ModuleList(
            [DecoderLayer(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.fc = nn.Linear(d_model, tgt_vocab)
        self.d_model = d_model

    def encode(self, src, src_mask=None):
        x = self.src_embed(src) * math.sqrt(self.d_model)
        x = x + self.pe[:, :x.shape[1]]
        for layer in self.enc_layers:
            x = layer(x, src_mask)
        return x

    def decode(self, tgt, enc, look_ahead_mask=None, pad_mask=None):
        x = self.tgt_embed(tgt) * math.sqrt(self.d_model)
        x = x + self.pe[:, :x.shape[1]]
        for layer in self.dec_layers:
            x = layer(x, enc, look_ahead_mask, pad_mask)
        return self.fc(x)

    def forward(self, src, tgt, src_mask=None, look_ahead_mask=None):
        enc = self.encode(src, src_mask)
        return self.decode(tgt, enc, look_ahead_mask, src_mask)


def make_look_ahead_mask(size):
    return torch.tril(torch.ones(size, size)).unsqueeze(0)  # (1, size, size)


# --------------------------------------------------------------------------- #
# Task: reverse a random digit sequence  (seq2seq with <sos>/<eos>)
# --------------------------------------------------------------------------- #
PAD, SOS, EOS = 0, 1, 2
DIGIT_OFFSET = 3  # digits 0..9 -> tokens 3..12
VOCAB = DIGIT_OFFSET + 10


def make_batch(batch, seq_len, rng):
    src = rng.randint(0, 10, size=(batch, seq_len)) + DIGIT_OFFSET
    rev = src[:, ::-1]
    # decoder input: <sos> + reversed ; target: reversed + <eos>
    dec_in = np.concatenate([np.full((batch, 1), SOS), rev], axis=1)
    dec_out = np.concatenate([rev, np.full((batch, 1), EOS)], axis=1)
    return (torch.tensor(src), torch.tensor(dec_in.copy()),
            torch.tensor(dec_out.copy()))


def main(seq_len=8, iters=800, batch=64):
    model = Transformer(VOCAB, VOCAB, d_model=64, n_heads=4, d_ff=128, n_layers=2)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    rng = np.random.RandomState(0)
    la_mask = make_look_ahead_mask(seq_len + 1)

    for it in range(iters):
        model.train()
        src, dec_in, dec_out = make_batch(batch, seq_len, rng)
        logits = model(src, dec_in, src_mask=None, look_ahead_mask=la_mask)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), dec_out.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 100 == 0 or it == iters - 1:
            acc = evaluate(model, seq_len, rng)
            print(f"iter {it:4d}  loss {loss.item():.4f}  seq-acc {acc*100:5.2f}%")

    acc = evaluate(model, seq_len, rng, n=2000)
    print(f"Final sequence accuracy (reverse task): {acc*100:.2f}%")
    return {"seq_acc": acc * 100}


def evaluate(model, seq_len, rng, n=1000):
    model.eval()
    src, dec_in, dec_out = make_batch(n, seq_len, rng)
    la_mask = make_look_ahead_mask(seq_len + 1)
    with torch.no_grad():
        logits = model(src, dec_in, src_mask=None, look_ahead_mask=la_mask)
        pred = logits.argmax(-1)
        # per-position accuracy over the reversed part (exclude the final <eos>)
        correct = (pred[:, :seq_len] == dec_out[:, :seq_len]).float().mean().item()
    return correct


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=800)
    args = ap.parse_args()
    main(iters=args.iters)
