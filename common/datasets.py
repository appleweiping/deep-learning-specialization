"""Dataset loaders / generators used across the from-scratch assignments.

All datasets here are either downloaded from public mirrors at runtime or
generated deterministically in numpy so that every assignment is runnable
on a CPU-only machine with no gated Coursera downloads.
"""
from __future__ import annotations

import gzip
import os
import struct
import urllib.request

import numpy as np

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# --------------------------------------------------------------------------- #
# Planar "flower" dataset (Course 1, Week 3)
# --------------------------------------------------------------------------- #
def load_planar_dataset(seed: int = 1):
    """The two-class 'flower' petal dataset from the planar-classification lab.

    Returns X of shape (2, m) and Y of shape (1, m).
    """
    np.random.seed(seed)
    m = 400
    N = m // 2          # points per class
    D = 2               # dimensionality
    X = np.zeros((m, D))
    Y = np.zeros((m, 1), dtype="uint8")
    a = 4               # maximum ray of the flower

    for j in range(2):
        ix = range(N * j, N * (j + 1))
        t = np.linspace(j * 3.12, (j + 1) * 3.12, N) + np.random.randn(N) * 0.2
        r = a * np.sin(4 * t) + np.random.randn(N) * 0.2
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        Y[ix] = j

    return X.T, Y.T


# --------------------------------------------------------------------------- #
# Synthetic "cat vs non-cat" style image dataset (Course 1, Week 2 & 4)
# --------------------------------------------------------------------------- #
def load_image_binary_dataset(seed: int = 1, n_train: int = 209, n_test: int = 50,
                              px: int = 64):
    """A separable synthetic RGB-image binary dataset that mirrors the shape and
    difficulty of the Coursera cat/non-cat dataset (64x64x3 images, ~209 train
    / 50 test) without redistributing the gated .h5 file.

    Class 1 ("cat"): images with a warm, high red-channel textured blob.
    Class 0 ("non-cat"): cooler, blue/green dominated noise images.

    Returns train_x_orig (n,px,px,3) uint8-like float, train_y (1,n),
            test_x_orig, test_y, classes.
    """
    rng = np.random.RandomState(seed)
    # Overlapping class means + label noise make this genuinely non-trivial:
    # logistic regression will overfit the training set, and a deep net
    # generalizes a little better (as in the original cat/non-cat lab).

    def _make(n, label_ratio=0.5, noise=0.08):
        xs, ys = [], []
        for i in range(n):
            is_cat = 1 if rng.rand() < label_ratio else 0
            img = rng.rand(px, px, 3) * 90  # strong base noise
            if is_cat:
                cy = px // 2 + int(rng.randn() * px * 0.12)
                cx = px // 2 + int(rng.randn() * px * 0.12)
                radius = px * (0.28 + rng.rand() * 0.12)
                yy, xx = np.ogrid[:px, :px]
                mask = (yy - cy) ** 2 + (xx - cx) ** 2 < radius ** 2
                img[..., 0] += mask * (90 + rng.rand(px, px) * 70)
                img[..., 1] += mask * (50 + rng.rand(px, px) * 50)
                img[..., 2] += rng.rand(px, px) * 60
            else:
                img[..., 2] += 70 + rng.rand(px, px) * 90
                img[..., 1] += 40 + rng.rand(px, px) * 70
                img[..., 0] += rng.rand(px, px) * 50
            img = np.clip(img, 0, 255)
            label = is_cat
            if rng.rand() < noise:      # flip a fraction of labels
                label = 1 - label
            xs.append(img)
            ys.append(label)
        return np.array(xs), np.array(ys).reshape(1, -1)

    train_x, train_y = _make(n_train)
    test_x, test_y = _make(n_test)
    classes = np.array([b"non-cat", b"cat"])
    return train_x, train_y, test_x, test_y, classes


# --------------------------------------------------------------------------- #
# MNIST (used by C2 TensorFlow-intro / optimizer assignments as a real dataset)
# --------------------------------------------------------------------------- #
_MNIST_URLS = {
    # OpenML / Google mirror of the original Yann LeCun files.
    "train_images": "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
    "train_labels": "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
    "test_images": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz",
    "test_labels": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz",
}


def _download(url: str, dest: str) -> None:
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return
    _ensure_dir(os.path.dirname(dest))
    print(f"downloading {url} -> {dest}", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "csdiy-dlspec/1.0"})
    tmp = dest + ".part"
    with urllib.request.urlopen(req, timeout=300) as r, open(tmp, "wb") as f:
        while True:
            chunk = r.read(1 << 20)  # stream 1MB at a time
            if not chunk:
                break
            f.write(chunk)
    os.replace(tmp, dest)


def _read_idx_images(path: str) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051, magic
        buf = f.read(rows * cols * num)
        data = np.frombuffer(buf, dtype=np.uint8).reshape(num, rows, cols)
    return data


def _read_idx_labels(path: str) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, num = struct.unpack(">II", f.read(8))
        assert magic == 2049, magic
        buf = f.read(num)
        labels = np.frombuffer(buf, dtype=np.uint8)
    return labels


_CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"


def load_cifar10(n_train: int | None = None, n_test: int | None = None,
                 classes: tuple[int, ...] | None = None):
    """Download (once) and load CIFAR-10 as NCHW float32 in [0,1].

    Returns X_train (n,3,32,32), y_train (n,), X_test, y_test.
    Optionally restrict to a subset of the 10 classes and relabel to 0..k-1.
    """
    import pickle
    import tarfile

    dest = os.path.join(_DATA_DIR, "cifar10", "cifar-10-python.tar.gz")
    _download(_CIFAR_URL, dest)
    root = os.path.join(_DATA_DIR, "cifar10")
    extracted = os.path.join(root, "cifar-10-batches-py")
    if not os.path.exists(extracted):
        with tarfile.open(dest, "r:gz") as t:
            t.extractall(root)

    def _load_batch(path):
        # CIFAR-10 batches are distributed as pickle files by the canonical
        # toronto.edu source (the only supported on-disk format). We only ever
        # unpickle the archive we downloaded from that trusted URL above.
        with open(path, "rb") as f:
            d = pickle.load(f, encoding="bytes")
        X = d[b"data"].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
        y = np.array(d[b"labels"], dtype=np.int64)
        return X, y

    Xtr_parts, ytr_parts = [], []
    for i in range(1, 6):
        X, y = _load_batch(os.path.join(extracted, f"data_batch_{i}"))
        Xtr_parts.append(X)
        ytr_parts.append(y)
    Xtr = np.concatenate(Xtr_parts)
    ytr = np.concatenate(ytr_parts)
    Xte, yte = _load_batch(os.path.join(extracted, "test_batch"))

    if classes is not None:
        remap = {c: i for i, c in enumerate(classes)}
        keep_tr = np.isin(ytr, classes)
        keep_te = np.isin(yte, classes)
        Xtr, ytr = Xtr[keep_tr], np.vectorize(remap.get)(ytr[keep_tr])
        Xte, yte = Xte[keep_te], np.vectorize(remap.get)(yte[keep_te])

    if n_train is not None:
        Xtr, ytr = Xtr[:n_train], ytr[:n_train]
    if n_test is not None:
        Xte, yte = Xte[:n_test], yte[:n_test]
    return Xtr, ytr, Xte, yte


def load_mnist(n_train: int | None = None, n_test: int | None = None,
               digits: tuple[int, ...] | None = None, flatten: bool = True):
    """Download (once) and load MNIST. Returns X_train, y_train, X_test, y_test.

    If ``digits`` is given, keep only those classes (useful for small binary tasks).
    """
    paths = {}
    for key, url in _MNIST_URLS.items():
        dest = os.path.join(_DATA_DIR, "mnist", os.path.basename(url))
        _download(url, dest)
        paths[key] = dest

    Xtr = _read_idx_images(paths["train_images"]).astype(np.float32) / 255.0
    ytr = _read_idx_labels(paths["train_labels"]).astype(np.int64)
    Xte = _read_idx_images(paths["test_images"]).astype(np.float32) / 255.0
    yte = _read_idx_labels(paths["test_labels"]).astype(np.int64)

    if digits is not None:
        keep_tr = np.isin(ytr, digits)
        keep_te = np.isin(yte, digits)
        Xtr, ytr, Xte, yte = Xtr[keep_tr], ytr[keep_tr], Xte[keep_te], yte[keep_te]

    if n_train is not None:
        Xtr, ytr = Xtr[:n_train], ytr[:n_train]
    if n_test is not None:
        Xte, yte = Xte[:n_test], yte[:n_test]

    if flatten:
        Xtr = Xtr.reshape(Xtr.shape[0], -1)
        Xte = Xte.reshape(Xte.shape[0], -1)

    return Xtr, ytr, Xte, yte
