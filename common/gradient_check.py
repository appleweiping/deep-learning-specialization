"""N-dimensional gradient checking (Course 2, Week 1 assignment).

Compares analytic backprop gradients against numerical (finite-difference)
gradients using the two-sided difference and the relative-error norm
    ||grad - gradapprox|| / (||grad|| + ||gradapprox||).
"""
from __future__ import annotations

import numpy as np

from . import nn_numpy as nn


def dictionary_to_vector(params):
    """Flatten a parameter dict {W1,b1,...} into a single (n,1) column vector."""
    keys, count, theta = [], 0, None
    L = len(params) // 2
    ordered = []
    for l in range(1, L + 1):
        ordered.append(f"W{l}")
        ordered.append(f"b{l}")
    for key in ordered:
        vec = params[key].reshape(-1, 1)
        keys += [key] * vec.shape[0]
        theta = vec if theta is None else np.concatenate((theta, vec), axis=0)
        count += vec.shape[0]
    return theta, keys


def vector_to_dictionary(theta, shapes):
    """Inverse of dictionary_to_vector given the original {key: shape} map."""
    params = {}
    idx = 0
    for key, shape in shapes.items():
        size = int(np.prod(shape))
        params[key] = theta[idx:idx + size].reshape(shape)
        idx += size
    return params


def gradients_to_vector(grads, shapes):
    theta = None
    for key in shapes:  # same order as params
        g = grads[f"d{key}"].reshape(-1, 1)
        theta = g if theta is None else np.concatenate((theta, g), axis=0)
    return theta


def gradient_check(X, Y, params, epsilon=1e-7, lambd=0.0):
    """Return the relative error between analytic and numerical gradients."""
    shapes = {k: v.shape for k, v in params.items()}

    # analytic gradients
    AL, caches, _ = nn.forward_propagation(X, params, keep_prob=1.0)
    grads = nn.backward_propagation(AL, Y, caches, lambd=lambd)
    grad = gradients_to_vector(grads, shapes)

    theta, _ = dictionary_to_vector(params)
    n = theta.shape[0]
    gradapprox = np.zeros((n, 1))

    for i in range(n):
        theta_plus = np.copy(theta)
        theta_plus[i, 0] += epsilon
        AL_p, _, _ = nn.forward_propagation(
            X, vector_to_dictionary(theta_plus, shapes), keep_prob=1.0
        )
        J_plus = nn.compute_cost(AL_p, Y, vector_to_dictionary(theta_plus, shapes), lambd)

        theta_minus = np.copy(theta)
        theta_minus[i, 0] -= epsilon
        AL_m, _, _ = nn.forward_propagation(
            X, vector_to_dictionary(theta_minus, shapes), keep_prob=1.0
        )
        J_minus = nn.compute_cost(AL_m, Y, vector_to_dictionary(theta_minus, shapes), lambd)

        gradapprox[i, 0] = (J_plus - J_minus) / (2 * epsilon)

    numerator = np.linalg.norm(grad - gradapprox)
    denominator = np.linalg.norm(grad) + np.linalg.norm(gradapprox)
    difference = numerator / (denominator + 1e-12)
    return difference, grad, gradapprox
