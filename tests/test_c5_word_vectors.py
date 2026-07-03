"""Verify the word-vector algebra (C5 W2) on deterministic synthetic vectors.

The full assignment runs on GloVe embeddings; here we check that the
operations (cosine similarity, analogy, neutralize, equalize) are correct
independent of which embedding table is used.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "C5_sequence_models"))
from a3_word_vectors_debiasing import (  # noqa: E402
    complete_analogy, cosine_similarity, equalize, neutralize,
)


def _toy_embeddings():
    """A tiny hand-built 3-d embedding where analogies hold exactly."""
    # dimension 0 = "royalty", dim 1 = "gender (male+/female-)", dim 2 = noise
    return {
        "king":  np.array([1.0, 1.0, 0.0]),
        "queen": np.array([1.0, -1.0, 0.0]),
        "man":   np.array([0.0, 1.0, 0.0]),
        "woman": np.array([0.0, -1.0, 0.0]),
        "prince": np.array([0.9, 1.0, 0.1]),
    }


def test_cosine_similarity_basic():
    u = np.array([1.0, 0.0])
    v = np.array([1.0, 0.0])
    assert abs(cosine_similarity(u, v) - 1.0) < 1e-6
    assert abs(cosine_similarity(u, np.array([0.0, 1.0]))) < 1e-6
    assert abs(cosine_similarity(u, np.array([-1.0, 0.0])) + 1.0) < 1e-6


def test_analogy_king_queen():
    emb = _toy_embeddings()
    # man : woman :: king : ?  -> queen
    w, _ = complete_analogy("man", "woman", "king", emb)
    assert w == "queen"


def test_neutralize_removes_bias_component():
    emb = _toy_embeddings()
    g = emb["woman"] - emb["man"]  # the gender axis
    # a word that has a gender component
    emb["engineer"] = np.array([0.3, 0.8, 0.2])
    before = abs(cosine_similarity(emb["engineer"], g))
    e = neutralize("engineer", g, emb)
    after = abs(cosine_similarity(e, g))
    assert before > 0.1
    assert after < 1e-6  # component along g removed


def test_equalize_symmetry():
    emb = _toy_embeddings()
    g = emb["woman"] - emb["man"]
    e1, e2 = equalize(("man", "woman"), g, emb)
    # after equalize, the two are symmetric about the neutral axis:
    # their similarities to g are equal in magnitude, opposite sign
    s1 = cosine_similarity(e1, g)
    s2 = cosine_similarity(e2, g)
    assert abs(s1 + s2) < 1e-6
