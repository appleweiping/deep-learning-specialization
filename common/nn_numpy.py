"""From-scratch deep neural network in pure numpy.

This module is the shared engine behind the Course 1 & Course 2 assignments:
  * L-layer forward / backward propagation
  * ReLU / sigmoid activations
  * multiple weight initializations (zeros, random, He)
  * L2 regularization and inverted dropout
  * cross-entropy cost
  * a training loop supporting batch / mini-batch gradient descent
    with GD, Momentum, RMSProp and Adam optimizers.

Everything is implemented by hand (no autograd) so the maths is explicit and
gradient-checkable.
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
# Activations
# --------------------------------------------------------------------------- #
def sigmoid(Z):
    A = 1.0 / (1.0 + np.exp(-Z))
    return A, Z


def relu(Z):
    A = np.maximum(0, Z)
    return A, Z


def sigmoid_backward(dA, Z):
    s = 1.0 / (1.0 + np.exp(-Z))
    return dA * s * (1 - s)


def relu_backward(dA, Z):
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ


# --------------------------------------------------------------------------- #
# Initialization
# --------------------------------------------------------------------------- #
def initialize_parameters(layer_dims, mode: str = "he", seed: int = 3):
    """Initialize W1..WL, b1..bL for a network with dims ``layer_dims``.

    mode:
        "zeros"  -> all weights 0 (demonstrates symmetry-breaking failure)
        "random" -> N(0,1)*10 (demonstrates exploding/vanishing without scaling)
        "he"     -> N(0,1)*sqrt(2/n_{l-1}) (recommended for ReLU nets)
    """
    np.random.seed(seed)
    params = {}
    L = len(layer_dims)
    for l in range(1, L):
        n_prev, n = layer_dims[l - 1], layer_dims[l]
        if mode == "zeros":
            params[f"W{l}"] = np.zeros((n, n_prev))
        elif mode == "random":
            params[f"W{l}"] = np.random.randn(n, n_prev) * 10
        elif mode == "he":
            params[f"W{l}"] = np.random.randn(n, n_prev) * np.sqrt(2.0 / n_prev)
        elif mode == "xavier":
            params[f"W{l}"] = np.random.randn(n, n_prev) * np.sqrt(1.0 / n_prev)
        else:
            raise ValueError(f"unknown init mode {mode}")
        params[f"b{l}"] = np.zeros((n, 1))
    return params


# --------------------------------------------------------------------------- #
# Forward propagation
# --------------------------------------------------------------------------- #
def linear_forward(A, W, b):
    Z = W @ A + b
    return Z, (A, W, b)


def linear_activation_forward(A_prev, W, b, activation):
    Z, linear_cache = linear_forward(A_prev, W, b)
    if activation == "sigmoid":
        A, activation_cache = sigmoid(Z)
    elif activation == "relu":
        A, activation_cache = relu(Z)
    else:
        raise ValueError(activation)
    return A, (linear_cache, activation_cache)


def forward_propagation(X, params, keep_prob: float = 1.0, seed: int | None = None):
    """Full forward pass through an L-layer [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID net.

    If ``keep_prob`` < 1, inverted dropout is applied on the hidden activations.
    """
    if seed is not None:
        np.random.seed(seed)
    caches = []
    dropout_masks = []
    A = X
    L = len(params) // 2
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(
            A_prev, params[f"W{l}"], params[f"b{l}"], "relu"
        )
        if keep_prob < 1.0:
            D = (np.random.rand(*A.shape) < keep_prob).astype(A.dtype)
            A = A * D / keep_prob
            dropout_masks.append(D)
        else:
            dropout_masks.append(None)
        caches.append(cache)
    AL, cache = linear_activation_forward(
        A, params[f"W{L}"], params[f"b{L}"], "sigmoid"
    )
    caches.append(cache)
    return AL, caches, dropout_masks


# --------------------------------------------------------------------------- #
# Cost
# --------------------------------------------------------------------------- #
def compute_cost(AL, Y, params=None, lambd: float = 0.0):
    m = Y.shape[1]
    eps = 1e-12
    cross_entropy = -np.sum(Y * np.log(AL + eps) + (1 - Y) * np.log(1 - AL + eps)) / m
    cost = np.squeeze(cross_entropy)
    if lambd > 0 and params is not None:
        L = len(params) // 2
        l2 = sum(np.sum(np.square(params[f"W{l}"])) for l in range(1, L + 1))
        cost = cost + (lambd / (2 * m)) * l2
    return cost


# --------------------------------------------------------------------------- #
# Backward propagation
# --------------------------------------------------------------------------- #
def linear_backward(dZ, cache, lambd: float = 0.0):
    A_prev, W, b = cache
    m = A_prev.shape[1]
    dW = (dZ @ A_prev.T) / m + (lambd / m) * W
    db = np.sum(dZ, axis=1, keepdims=True) / m
    dA_prev = W.T @ dZ
    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation, lambd: float = 0.0):
    linear_cache, activation_cache = cache
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
    else:
        raise ValueError(activation)
    return linear_backward(dZ, linear_cache, lambd)


def backward_propagation(AL, Y, caches, lambd: float = 0.0,
                         keep_prob: float = 1.0, dropout_masks=None):
    grads = {}
    L = len(caches)
    Y = Y.reshape(AL.shape)
    eps = 1e-12
    dAL = -(np.divide(Y, AL + eps) - np.divide(1 - Y, 1 - AL + eps))

    current_cache = caches[L - 1]
    dA_prev, dW, db = linear_activation_backward(dAL, current_cache, "sigmoid", lambd)
    grads[f"dA{L-1}"], grads[f"dW{L}"], grads[f"db{L}"] = dA_prev, dW, db

    for l in reversed(range(L - 1)):
        dA = grads[f"dA{l+1}"]
        if keep_prob < 1.0 and dropout_masks is not None and dropout_masks[l] is not None:
            dA = dA * dropout_masks[l] / keep_prob
        current_cache = caches[l]
        dA_prev, dW, db = linear_activation_backward(dA, current_cache, "relu", lambd)
        grads[f"dA{l}"], grads[f"dW{l+1}"], grads[f"db{l+1}"] = dA_prev, dW, db
    return grads


# --------------------------------------------------------------------------- #
# Optimizers
# --------------------------------------------------------------------------- #
def initialize_adam(params):
    L = len(params) // 2
    v, s = {}, {}
    for l in range(1, L + 1):
        v[f"dW{l}"] = np.zeros_like(params[f"W{l}"])
        v[f"db{l}"] = np.zeros_like(params[f"b{l}"])
        s[f"dW{l}"] = np.zeros_like(params[f"W{l}"])
        s[f"db{l}"] = np.zeros_like(params[f"b{l}"])
    return v, s


def initialize_velocity(params):
    L = len(params) // 2
    v = {}
    for l in range(1, L + 1):
        v[f"dW{l}"] = np.zeros_like(params[f"W{l}"])
        v[f"db{l}"] = np.zeros_like(params[f"b{l}"])
    return v


def update_parameters_gd(params, grads, learning_rate):
    L = len(params) // 2
    for l in range(1, L + 1):
        params[f"W{l}"] -= learning_rate * grads[f"dW{l}"]
        params[f"b{l}"] -= learning_rate * grads[f"db{l}"]
    return params


def update_parameters_momentum(params, grads, v, beta, learning_rate):
    L = len(params) // 2
    for l in range(1, L + 1):
        v[f"dW{l}"] = beta * v[f"dW{l}"] + (1 - beta) * grads[f"dW{l}"]
        v[f"db{l}"] = beta * v[f"db{l}"] + (1 - beta) * grads[f"db{l}"]
        params[f"W{l}"] -= learning_rate * v[f"dW{l}"]
        params[f"b{l}"] -= learning_rate * v[f"db{l}"]
    return params, v


def update_parameters_adam(params, grads, v, s, t, learning_rate,
                           beta1=0.9, beta2=0.999, epsilon=1e-8):
    L = len(params) // 2
    v_corr, s_corr = {}, {}
    for l in range(1, L + 1):
        for p in ("dW", "db"):
            key = f"{p}{l}"
            v[key] = beta1 * v[key] + (1 - beta1) * grads[key]
            s[key] = beta2 * s[key] + (1 - beta2) * (grads[key] ** 2)
            v_corr[key] = v[key] / (1 - beta1 ** t)
            s_corr[key] = s[key] / (1 - beta2 ** t)
        params[f"W{l}"] -= learning_rate * v_corr[f"dW{l}"] / (np.sqrt(s_corr[f"dW{l}"]) + epsilon)
        params[f"b{l}"] -= learning_rate * v_corr[f"db{l}"] / (np.sqrt(s_corr[f"db{l}"]) + epsilon)
    return params, v, s


# --------------------------------------------------------------------------- #
# Mini-batches
# --------------------------------------------------------------------------- #
def random_mini_batches(X, Y, mini_batch_size=64, seed=0):
    np.random.seed(seed)
    m = X.shape[1]
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape(Y.shape[0], m)

    mini_batches = []
    num_complete = m // mini_batch_size
    for k in range(num_complete):
        mb_X = shuffled_X[:, k * mini_batch_size:(k + 1) * mini_batch_size]
        mb_Y = shuffled_Y[:, k * mini_batch_size:(k + 1) * mini_batch_size]
        mini_batches.append((mb_X, mb_Y))
    if m % mini_batch_size != 0:
        mb_X = shuffled_X[:, num_complete * mini_batch_size:]
        mb_Y = shuffled_Y[:, num_complete * mini_batch_size:]
        mini_batches.append((mb_X, mb_Y))
    return mini_batches


# --------------------------------------------------------------------------- #
# High-level model
# --------------------------------------------------------------------------- #
def predict(X, params):
    AL, _, _ = forward_propagation(X, params, keep_prob=1.0)
    return (AL > 0.5).astype(int)


def accuracy(X, Y, params):
    preds = predict(X, params)
    return float(np.mean(preds == Y))


def model(X, Y, layer_dims, learning_rate=0.0075, num_iterations=2500,
          init="he", lambd=0.0, keep_prob=1.0, optimizer="gd",
          mini_batch_size=None, beta=0.9, beta1=0.9, beta2=0.999,
          epsilon=1e-8, print_every=0, seed=3):
    """Train an L-layer network. Returns (params, costs)."""
    np.random.seed(seed)
    params = initialize_parameters(layer_dims, mode=init, seed=seed)
    costs = []

    if optimizer == "momentum":
        v = initialize_velocity(params)
    elif optimizer == "adam":
        v, s = initialize_adam(params)
    t = 0
    mb_seed = seed

    for i in range(num_iterations):
        if mini_batch_size is None:
            batches = [(X, Y)]
        else:
            mb_seed += 1
            batches = random_mini_batches(X, Y, mini_batch_size, mb_seed)

        epoch_cost = 0.0
        for mb_X, mb_Y in batches:
            AL, caches, masks = forward_propagation(
                mb_X, params, keep_prob=keep_prob,
                seed=(seed + i if keep_prob < 1 else None),
            )
            epoch_cost += compute_cost(AL, mb_Y, params, lambd)
            grads = backward_propagation(AL, mb_Y, caches, lambd, keep_prob, masks)

            if optimizer == "gd":
                params = update_parameters_gd(params, grads, learning_rate)
            elif optimizer == "momentum":
                params, v = update_parameters_momentum(params, grads, v, beta, learning_rate)
            elif optimizer == "adam":
                t += 1
                params, v, s = update_parameters_adam(
                    params, grads, v, s, t, learning_rate, beta1, beta2, epsilon
                )
            else:
                raise ValueError(optimizer)
        epoch_cost /= len(batches)

        if print_every and i % print_every == 0:
            print(f"iter {i:5d}  cost {epoch_cost:.6f}")
        if i % 100 == 0:
            costs.append(epoch_cost)

    return params, costs
