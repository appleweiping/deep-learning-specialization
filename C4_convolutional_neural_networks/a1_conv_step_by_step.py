"""C4 W1 - Convolutional Model: step by step (pure numpy).

Implements, by hand, the forward and backward passes for:
  * zero padding
  * a single convolution step
  * a full convolution forward pass (with stride & pad)
  * max / average pooling forward
  * convolution backward (dA_prev, dW, db)
  * max / average pooling backward

Correctness of the backward passes is verified against a numerical gradient
in tests/test_c4_conv.py.
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
# Forward
# --------------------------------------------------------------------------- #
def zero_pad(X, pad):
    """Pad the height/width of a batch of images X (m, nH, nW, nC)."""
    return np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)),
                  mode="constant", constant_values=0)


def conv_single_step(a_slice_prev, W, b):
    """Convolve one (f,f,nC_prev) slice with one filter W and bias b -> scalar."""
    s = a_slice_prev * W
    Z = np.sum(s)
    return float(Z + np.squeeze(b))


def conv_forward(A_prev, W, b, hparameters):
    """Forward pass of a conv layer.

    A_prev: (m, nH_prev, nW_prev, nC_prev)
    W:      (f, f, nC_prev, nC)
    b:      (1, 1, 1, nC)
    hparameters: {"stride": s, "pad": p}
    Returns Z (m, nH, nW, nC) and cache.
    """
    (m, nH_prev, nW_prev, nC_prev) = A_prev.shape
    (f, f, nC_prev_w, nC) = W.shape
    assert nC_prev == nC_prev_w
    stride = hparameters["stride"]
    pad = hparameters["pad"]

    nH = (nH_prev - f + 2 * pad) // stride + 1
    nW = (nW_prev - f + 2 * pad) // stride + 1

    Z = np.zeros((m, nH, nW, nC))
    A_prev_pad = zero_pad(A_prev, pad)

    for i in range(m):
        a_prev_pad = A_prev_pad[i]
        for h in range(nH):
            vs = h * stride
            ve = vs + f
            for w in range(nW):
                hs = w * stride
                he = hs + f
                for c in range(nC):
                    a_slice = a_prev_pad[vs:ve, hs:he, :]
                    Z[i, h, w, c] = conv_single_step(a_slice, W[..., c], b[..., c])

    cache = (A_prev, W, b, hparameters)
    return Z, cache


def pool_forward(A_prev, hparameters, mode="max"):
    """Forward pass of a pooling layer (max or average)."""
    (m, nH_prev, nW_prev, nC_prev) = A_prev.shape
    f = hparameters["f"]
    stride = hparameters["stride"]

    nH = (nH_prev - f) // stride + 1
    nW = (nW_prev - f) // stride + 1
    nC = nC_prev

    A = np.zeros((m, nH, nW, nC))
    for i in range(m):
        for h in range(nH):
            vs = h * stride
            ve = vs + f
            for w in range(nW):
                hs = w * stride
                he = hs + f
                for c in range(nC):
                    a_slice = A_prev[i, vs:ve, hs:he, c]
                    if mode == "max":
                        A[i, h, w, c] = np.max(a_slice)
                    else:
                        A[i, h, w, c] = np.mean(a_slice)
    cache = (A_prev, hparameters)
    return A, cache


# --------------------------------------------------------------------------- #
# Backward
# --------------------------------------------------------------------------- #
def conv_backward(dZ, cache):
    """Backward pass of a conv layer. Returns dA_prev, dW, db."""
    (A_prev, W, b, hparameters) = cache
    (m, nH_prev, nW_prev, nC_prev) = A_prev.shape
    (f, f, nC_prev, nC) = W.shape
    stride = hparameters["stride"]
    pad = hparameters["pad"]
    (m, nH, nW, nC) = dZ.shape

    dA_prev = np.zeros_like(A_prev)
    dW = np.zeros_like(W)
    db = np.zeros_like(b)

    A_prev_pad = zero_pad(A_prev, pad)
    dA_prev_pad = zero_pad(dA_prev, pad)

    for i in range(m):
        a_prev_pad = A_prev_pad[i]
        da_prev_pad = dA_prev_pad[i]
        for h in range(nH):
            vs, ve = h * stride, h * stride + f
            for w in range(nW):
                hs, he = w * stride, w * stride + f
                for c in range(nC):
                    a_slice = a_prev_pad[vs:ve, hs:he, :]
                    da_prev_pad[vs:ve, hs:he, :] += W[..., c] * dZ[i, h, w, c]
                    dW[..., c] += a_slice * dZ[i, h, w, c]
                    db[..., c] += dZ[i, h, w, c]
        if pad > 0:
            dA_prev[i] = da_prev_pad[pad:-pad, pad:-pad, :]
        else:
            dA_prev[i] = da_prev_pad
    return dA_prev, dW, db


def create_mask_from_window(x):
    return (x == np.max(x))


def distribute_value(dz, shape):
    (nH, nW) = shape
    return np.ones(shape) * (dz / (nH * nW))


def pool_backward(dA, cache, mode="max"):
    """Backward pass of a pooling layer. Returns dA_prev."""
    (A_prev, hparameters) = cache
    stride = hparameters["stride"]
    f = hparameters["f"]
    m, nH, nW, nC = dA.shape
    dA_prev = np.zeros_like(A_prev)

    for i in range(m):
        a_prev = A_prev[i]
        for h in range(nH):
            vs, ve = h * stride, h * stride + f
            for w in range(nW):
                hs, he = w * stride, w * stride + f
                for c in range(nC):
                    if mode == "max":
                        a_slice = a_prev[vs:ve, hs:he, c]
                        mask = create_mask_from_window(a_slice)
                        dA_prev[i, vs:ve, hs:he, c] += mask * dA[i, h, w, c]
                    else:
                        da = dA[i, h, w, c]
                        dA_prev[i, vs:ve, hs:he, c] += distribute_value(da, (f, f))
    return dA_prev


def _demo():
    np.random.seed(1)
    A_prev = np.random.randn(2, 5, 7, 4)
    W = np.random.randn(3, 3, 4, 8)
    b = np.random.randn(1, 1, 1, 8)
    Z, cache = conv_forward(A_prev, W, b, {"pad": 1, "stride": 2})
    print("conv_forward  Z.shape =", Z.shape, " Z.mean =", round(float(Z.mean()), 6))
    dA, dW, db = conv_backward(np.random.randn(*Z.shape), cache)
    print("conv_backward dA", dA.shape, "dW", dW.shape, "db", db.shape)
    A, pcache = pool_forward(A_prev, {"stride": 1, "f": 3}, mode="max")
    print("pool_forward(max) A.shape =", A.shape, " A.mean =", round(float(A.mean()), 6))
    dprev = pool_backward(np.random.randn(*A.shape), pcache, mode="max")
    print("pool_backward dA_prev.shape =", dprev.shape)


if __name__ == "__main__":
    _demo()
