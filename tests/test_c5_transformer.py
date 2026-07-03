"""Tests for the from-scratch Transformer components (C5 W4)."""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "C5_sequence_models"))
from a5_transformer import (  # noqa: E402
    MultiHeadAttention, Transformer, make_look_ahead_mask,
    positional_encoding, scaled_dot_product_attention,
)


def test_attention_is_convex_combination():
    torch.manual_seed(0)
    q = torch.randn(2, 5, 8)
    k = torch.randn(2, 5, 8)
    v = torch.randn(2, 5, 8)
    out, attn = scaled_dot_product_attention(q, k, v)
    # attention weights sum to 1 over the key axis
    assert torch.allclose(attn.sum(-1), torch.ones(2, 5), atol=1e-5)
    assert out.shape == (2, 5, 8)


def test_look_ahead_mask_blocks_future():
    mask = make_look_ahead_mask(4)[0]
    # lower-triangular: position i cannot attend to j>i
    assert mask[0, 1] == 0 and mask[0, 0] == 1
    assert mask[3, 0] == 1 and mask[2, 3] == 0


def test_masked_attention_ignores_future():
    torch.manual_seed(1)
    q = torch.randn(1, 4, 8)
    k = torch.randn(1, 4, 8)
    v = torch.randn(1, 4, 8)
    mask = make_look_ahead_mask(4)
    _, attn = scaled_dot_product_attention(q, k, v, mask)
    # upper triangle of attention must be exactly zero
    upper = torch.triu(attn[0], diagonal=1)
    assert torch.allclose(upper, torch.zeros_like(upper), atol=1e-6)


def test_positional_encoding_shape_and_range():
    pe = positional_encoding(16, 32)
    assert pe.shape == (1, 16, 32)
    assert pe.max() <= 1.0 and pe.min() >= -1.0


def test_multihead_output_shape():
    mha = MultiHeadAttention(d_model=32, n_heads=4)
    x = torch.randn(3, 7, 32)
    out = mha(x, x, x)
    assert out.shape == (3, 7, 32)


def test_transformer_forward_shape():
    model = Transformer(src_vocab=13, tgt_vocab=13, d_model=32, n_heads=4,
                        d_ff=64, n_layers=2, max_len=20)
    src = torch.randint(0, 13, (2, 8))
    tgt = torch.randint(0, 13, (2, 9))
    la = make_look_ahead_mask(9)
    logits = model(src, tgt, look_ahead_mask=la)
    assert logits.shape == (2, 9, 13)
