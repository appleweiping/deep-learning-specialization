"""C5 W1 - Character-level language model: Dinosaur names.

A vanilla RNN language model trained character-by-character in pure numpy
(forward, backward-through-time, gradient clipping, Adagrad-style updates) to
generate novel dinosaur-like names.

The training corpus (data/dinos.txt) is a public list of dinosaur genus names.
If it is missing it is fetched from a public mirror at runtime.
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data", "dinos.txt")


def load_names():
    if not os.path.exists(DATA):
        # Build the corpus from the bundled curated list of real dinosaur genera.
        import _build_dinos
        _build_dinos.main()
    with open(DATA) as f:
        data = f.read().lower()
    names = [n.strip() for n in data.split("\n") if n.strip()]
    return names, data


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / np.sum(e)


def clip(gradients, maxValue):
    for k in gradients:
        np.clip(gradients[k], -maxValue, maxValue, out=gradients[k])
    return gradients


def rnn_step_forward(parameters, a_prev, x):
    Waa, Wax, Wya = parameters["Waa"], parameters["Wax"], parameters["Wya"]
    by, ba = parameters["by"], parameters["b"]
    a_next = np.tanh(Wax @ x + Waa @ a_prev + ba)
    p_t = softmax(Wya @ a_next + by)
    return a_next, p_t


def rnn_forward(X, Y, a0, parameters, vocab_size):
    x, a, y_hat = {}, {}, {}
    a[-1] = np.copy(a0)
    loss = 0
    for t in range(len(X)):
        x[t] = np.zeros((vocab_size, 1))
        if X[t] is not None:
            x[t][X[t]] = 1
        a[t], y_hat[t] = rnn_step_forward(parameters, a[t - 1], x[t])
        loss -= np.log(y_hat[t][Y[t], 0] + 1e-12)
    cache = (y_hat, a, x)
    return loss, cache


def rnn_backward(X, Y, parameters, cache):
    gradients = {}
    (y_hat, a, x) = cache
    Waa, Wax, Wya = parameters["Waa"], parameters["Wax"], parameters["Wya"]
    gradients["dWax"] = np.zeros_like(Wax)
    gradients["dWaa"] = np.zeros_like(Waa)
    gradients["dWya"] = np.zeros_like(Wya)
    gradients["db"] = np.zeros_like(parameters["b"])
    gradients["dby"] = np.zeros_like(parameters["by"])
    gradients["da_next"] = np.zeros_like(a[0])

    for t in reversed(range(len(X))):
        dy = np.copy(y_hat[t])
        dy[Y[t]] -= 1
        gradients["dWya"] += dy @ a[t].T
        gradients["dby"] += dy
        da = Wya.T @ dy + gradients["da_next"]
        dtanh = (1 - a[t] ** 2) * da
        gradients["db"] += dtanh
        gradients["dWax"] += dtanh @ x[t].T
        gradients["dWaa"] += dtanh @ a[t - 1].T
        gradients["da_next"] = Waa.T @ dtanh
    return gradients, a


def update_parameters(parameters, gradients, lr):
    parameters["Wax"] -= lr * gradients["dWax"]
    parameters["Waa"] -= lr * gradients["dWaa"]
    parameters["Wya"] -= lr * gradients["dWya"]
    parameters["b"] -= lr * gradients["db"]
    parameters["by"] -= lr * gradients["dby"]
    return parameters


def initialize_parameters(n_a, n_x, n_y, seed=1):
    np.random.seed(seed)
    return {
        "Wax": np.random.randn(n_a, n_x) * 0.01,
        "Waa": np.random.randn(n_a, n_a) * 0.01,
        "Wya": np.random.randn(n_y, n_a) * 0.01,
        "b": np.zeros((n_a, 1)),
        "by": np.zeros((n_y, 1)),
    }


def sample(parameters, char_to_ix, ix_to_char, seed):
    Waa, Wax, Wya = parameters["Waa"], parameters["Wax"], parameters["Wya"]
    by, b = parameters["by"], parameters["b"]
    vocab_size = by.shape[0]
    n_a = Waa.shape[1]

    x = np.zeros((vocab_size, 1))
    a_prev = np.zeros((n_a, 1))
    indices = []
    idx = -1
    counter = 0
    newline_character = char_to_ix["\n"]
    rng = np.random.RandomState(seed)

    while idx != newline_character and counter != 50:
        a = np.tanh(Wax @ x + Waa @ a_prev + b)
        z = Wya @ a + by
        y = softmax(z)
        idx = rng.choice(range(vocab_size), p=y.ravel())
        indices.append(idx)
        x = np.zeros((vocab_size, 1))
        x[idx] = 1
        a_prev = a
        counter += 1
    if counter == 50:
        indices.append(newline_character)
    return indices


def optimize(X, Y, a_prev, parameters, vocab_size, lr):
    loss, cache = rnn_forward(X, Y, a_prev, parameters, vocab_size)
    gradients, a = rnn_backward(X, Y, parameters, cache)
    gradients = clip(gradients, 5)
    parameters = update_parameters(parameters, gradients, lr)
    return loss, gradients, a[len(X) - 1]


def model(names, ix_to_char, char_to_ix, vocab_size, num_iterations=22000,
          n_a=50, lr=0.01, seed=0, verbose=True):
    n_x = n_y = vocab_size
    parameters = initialize_parameters(n_a, n_x, n_y, seed=1)
    loss = -np.log(1.0 / vocab_size) * 7  # smooth loss init
    np.random.seed(seed)
    examples = names[:]
    np.random.shuffle(examples)
    a_prev = np.zeros((n_a, 1))

    samples_log = []
    for j in range(num_iterations):
        idx = j % len(examples)
        single = [char_to_ix[c] for c in examples[idx]]
        X = [None] + single
        Y = single + [char_to_ix["\n"]]
        cur_loss, _, a_prev = optimize(X, Y, a_prev, parameters, vocab_size, lr)
        loss = loss * 0.999 + cur_loss * 0.001
        if verbose and j % 3000 == 0:
            print(f"iter {j:6d}  loss {loss:.4f}")
            gens = [
                "".join(ix_to_char[i] for i in sample(parameters, char_to_ix, ix_to_char, seed=j + k)).strip()
                for k in range(4)
            ]
            print("   samples:", ", ".join(gens))
            samples_log.append((j, gens))
    return parameters, loss, samples_log


def main(num_iterations=22000):
    names, data = load_names()
    chars = sorted(set(data))
    if "\n" not in chars:
        chars.append("\n")
    vocab_size = len(chars)
    char_to_ix = {c: i for i, c in enumerate(chars)}
    ix_to_char = {i: c for i, c in enumerate(chars)}
    print(f"{len(names)} names, vocab size {vocab_size}")

    parameters, final_loss, log = model(names, ix_to_char, char_to_ix, vocab_size,
                                        num_iterations=num_iterations)
    print(f"\nFinal smoothed loss: {final_loss:.4f}")
    print("Final generated names:")
    finals = []
    for k in range(12):
        s = "".join(ix_to_char[i] for i in sample(parameters, char_to_ix, ix_to_char, seed=10000 + k)).strip()
        finals.append(s.capitalize())
    print("  " + ", ".join(finals))
    return {"final_loss": final_loss, "samples": finals}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=22000)
    args = ap.parse_args()
    main(args.iters)
