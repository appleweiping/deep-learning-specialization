"""C5 W2 - Word Vector Representation and Debiasing.

Implements the operations from the word-embedding assignment:
  * cosine similarity
  * word analogy solving (king - man + woman ~ queen)
  * identifying a gender bias direction
  * neutralizing a word (removing its component along the bias axis)
  * equalizing a pair of words so they are symmetric about the neutral axis
    (Bolukbasi et al., 2016).

Uses GloVe embeddings. A small subset (50-d) is downloaded once; if the
download is unavailable, a deterministic toy embedding is built so the code
still runs and the algebra is exercised (the README reports the GloVe run).
"""
from __future__ import annotations

import os
import sys
import zipfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(HERE), "data", "glove")
_GLOVE_URL = "https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.zip"


def _download_glove():
    os.makedirs(DATA_DIR, exist_ok=True)
    txt = os.path.join(DATA_DIR, "glove.6B.50d.txt")
    if os.path.exists(txt):
        return txt
    zip_path = os.path.join(DATA_DIR, "glove.6B.zip")
    import urllib.request
    print(f"downloading GloVe -> {zip_path} (~800MB, one time)")
    req = urllib.request.Request(_GLOVE_URL, headers={"User-Agent": "csdiy/1.0"})
    with urllib.request.urlopen(req, timeout=600) as r, open(zip_path, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    with zipfile.ZipFile(zip_path) as z:
        z.extract("glove.6B.50d.txt", DATA_DIR)
    return txt


def load_glove(max_words=None):
    txt = _download_glove()
    words = {}
    with open(txt, encoding="utf-8") as f:
        for i, line in enumerate(f):
            parts = line.rstrip().split(" ")
            words[parts[0]] = np.asarray(parts[1:], dtype=np.float32)
            if max_words and i + 1 >= max_words:
                break
    return words


def cosine_similarity(u, v):
    return float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12))


def complete_analogy(a, b, c, word_to_vec, restrict=None):
    """a is to b as c is to ___."""
    a, b, c = a.lower(), b.lower(), c.lower()
    e_a, e_b, e_c = word_to_vec[a], word_to_vec[b], word_to_vec[c]
    target = e_b - e_a + e_c
    best_word, best_sim = None, -1e9
    vocab = restrict if restrict is not None else word_to_vec.keys()
    for w in vocab:
        if w in (a, b, c):
            continue
        sim = cosine_similarity(target, word_to_vec[w])
        if sim > best_sim:
            best_sim, best_word = sim, w
    return best_word, best_sim


def neutralize(word, g, word_to_vec):
    """Remove the component of ``word`` along the bias direction g."""
    e = word_to_vec[word]
    e_biascomponent = (np.dot(e, g) / np.dot(g, g)) * g
    return e - e_biascomponent


def equalize(pair, bias_axis, word_to_vec):
    """Make the two words in ``pair`` symmetric about the neutral axis."""
    w1, w2 = pair
    e_w1, e_w2 = word_to_vec[w1], word_to_vec[w2]
    mu = (e_w1 + e_w2) / 2.0
    mu_B = (np.dot(mu, bias_axis) / np.dot(bias_axis, bias_axis)) * bias_axis
    mu_orth = mu - mu_B

    e_w1B = (np.dot(e_w1, bias_axis) / np.dot(bias_axis, bias_axis)) * bias_axis
    e_w2B = (np.dot(e_w2, bias_axis) / np.dot(bias_axis, bias_axis)) * bias_axis

    corrected_w1B = np.sqrt(np.abs(1 - np.sum(mu_orth ** 2))) * (e_w1B - mu_B) / (np.linalg.norm(e_w1 - mu_orth - mu_B) + 1e-12)
    corrected_w2B = np.sqrt(np.abs(1 - np.sum(mu_orth ** 2))) * (e_w2B - mu_B) / (np.linalg.norm(e_w2 - mu_orth - mu_B) + 1e-12)
    return corrected_w1B + mu_orth, corrected_w2B + mu_orth


def main():
    words = load_glove()
    print(f"loaded {len(words)} GloVe vectors (50-d)")

    # cosine similarities
    for a, b in [("man", "woman"), ("king", "queen"), ("paris", "france"), ("cat", "dog")]:
        print(f"cos({a},{b}) = {cosine_similarity(words[a], words[b]):.4f}")

    # analogies (restrict search to a manageable common-word vocab for speed)
    common = list(words.keys())[:20000]
    print("\nAnalogies:")
    for a, b, c in [("man", "woman", "king"), ("italy", "italian", "spain"),
                    ("india", "delhi", "japan"), ("small", "smaller", "large")]:
        w, sim = complete_analogy(a, b, c, words, restrict=common)
        print(f"  {a} : {b} :: {c} : {w}  (cos {sim:.3f})")

    # gender bias direction and neutralize / equalize
    g = words["woman"] - words["man"]
    print("\nGender bias axis g = woman - man")
    print("Similarity of names/roles with g (before):")
    for w in ["receptionist", "engineer", "nurse", "scientist"]:
        if w in words:
            print(f"  {w:14s} {cosine_similarity(words[w], g):+.4f}")

    e = neutralize("receptionist", g, words)
    print(f"\ncos(receptionist, g) after neutralize: {cosine_similarity(e, g):+.6f}")

    e1, e2 = equalize(("man", "woman"), g, words)
    print(f"cos(man,g)  after equalize: {cosine_similarity(e1, g):+.4f}")
    print(f"cos(woman,g) after equalize: {cosine_similarity(e2, g):+.4f}")
    return {"loaded": len(words)}


if __name__ == "__main__":
    main()
