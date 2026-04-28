"""
tests/test_trust_score.py

Unit tests for compute_trust_score() in src/retriever.py.

The tests use a mock embedder that returns controlled vectors so results
are deterministic and do not require loading any model files.

Run with: python -m pytest tests/test_trust_score.py -v
"""

import numpy as np
import pytest

from src.retriever import compute_trust_score


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

class MockEmbedder:
    """Returns pre-set row vectors, indexed by the order encode() is called."""

    def __init__(self, vectors: np.ndarray):
        """vectors: 2-D array, shape (n_texts, dim)"""
        self._vectors = np.array(vectors, dtype="float32")

    def encode(self, texts):
        n = len(texts)
        return self._vectors[:n]


class MockFAISSIndex:
    """Minimal FAISS index mock that supports reconstruct(i)."""

    def __init__(self, vectors: np.ndarray):
        self._vectors = np.array(vectors, dtype="float32")

    def reconstruct(self, i: int) -> np.ndarray:
        return self._vectors[i].copy()


def _unit(v):
    """Return a unit vector."""
    v = np.array(v, dtype="float32")
    return v / np.linalg.norm(v)


# ---------------------------------------------------------------------------
# High-confidence scenario: all chunks point in the same direction
# ---------------------------------------------------------------------------

class TestHighConfidence:
    def test_identical_chunks_score_one(self):
        """Identical embeddings → cosine similarity = 1.0 → high confidence."""
        vec = _unit([1.0, 0.0, 0.0])
        vecs = np.stack([vec, vec, vec])
        chunks = ["chunk A", "chunk B", "chunk C"]
        embedder = MockEmbedder(vecs)

        score, low_conf = compute_trust_score([0, 1, 2], chunks, embedder)
        assert abs(score - 1.0) < 1e-5
        assert low_conf is False

    def test_similar_chunks_high_score(self):
        """Slightly varied but similar vectors → score close to 1, not low-conf."""
        base = _unit([1.0, 0.0, 0.0])
        near = _unit([0.99, 0.14, 0.0])
        vecs = np.stack([base, near, base])
        chunks = ["A", "B", "C"]
        embedder = MockEmbedder(vecs)

        score, low_conf = compute_trust_score([0, 1, 2], chunks, embedder,
                                              low_confidence_threshold=0.30)
        assert score > 0.30
        assert low_conf is False


# ---------------------------------------------------------------------------
# Low-confidence scenario: chunks point in orthogonal / opposite directions
# ---------------------------------------------------------------------------

class TestLowConfidence:
    def test_orthogonal_chunks_low_score(self):
        """Orthogonal embeddings → cosine similarity = 0 → low confidence."""
        v1 = _unit([1.0, 0.0, 0.0])
        v2 = _unit([0.0, 1.0, 0.0])
        v3 = _unit([0.0, 0.0, 1.0])
        vecs = np.stack([v1, v2, v3])
        chunks = ["A", "B", "C"]
        embedder = MockEmbedder(vecs)

        score, low_conf = compute_trust_score([0, 1, 2], chunks, embedder,
                                              low_confidence_threshold=0.30)
        assert score < 0.30
        assert low_conf is True

    def test_opposite_vectors_negative_similarity(self):
        """Opposite vectors → score < 0 → definitely low confidence."""
        v1 = _unit([1.0, 0.0])
        v2 = _unit([-1.0, 0.0])
        vecs = np.stack([v1, v2])
        chunks = ["A", "B"]
        embedder = MockEmbedder(vecs)

        score, low_conf = compute_trust_score([0, 1], chunks, embedder,
                                              low_confidence_threshold=0.30)
        assert score < 0.0
        assert low_conf is True


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_single_chunk_returns_full_confidence(self):
        """With only one chunk there are no pairs; trust defaults to 1.0."""
        vec = _unit([1.0, 0.0])
        embedder = MockEmbedder(np.stack([vec]))
        chunks = ["only chunk"]

        score, low_conf = compute_trust_score([0], chunks, embedder)
        assert score == 1.0
        assert low_conf is False

    def test_empty_indices_returns_full_confidence(self):
        """Empty topk_idxs should return 1.0, not crash."""
        embedder = MockEmbedder(np.zeros((0, 2), dtype="float32"))
        chunks = ["A", "B"]

        score, low_conf = compute_trust_score([], chunks, embedder)
        assert score == 1.0
        assert low_conf is False

    def test_custom_threshold_respected(self):
        """Threshold parameter controls the low-confidence cutoff."""
        v1 = _unit([1.0, 0.0])
        v2 = _unit([0.8, 0.6])  # cosine ≈ 0.80
        vecs = np.stack([v1, v2])
        chunks = ["A", "B"]
        embedder = MockEmbedder(vecs)

        # With high threshold → low confidence
        _, low_conf_high = compute_trust_score([0, 1], chunks, embedder,
                                               low_confidence_threshold=0.90)
        assert low_conf_high is True

        # With low threshold → not low confidence
        _, low_conf_low = compute_trust_score([0, 1], chunks, embedder,
                                              low_confidence_threshold=0.50)
        assert low_conf_low is False

    def test_out_of_bounds_indices_skipped(self):
        """Indices beyond chunk list length should not cause IndexError."""
        vec = _unit([1.0, 0.0])
        vecs = np.stack([vec, vec])
        chunks = ["A"]  # only 1 chunk but indices include 1 (out of bounds)
        embedder = MockEmbedder(vecs)

        # Should not raise; only the in-bounds chunk is used → single chunk → 1.0
        score, low_conf = compute_trust_score([0, 1], chunks, embedder)
        assert score == 1.0
        assert low_conf is False

    def test_two_chunks_pairwise(self):
        """Two chunks: exactly one pair computed."""
        v1 = _unit([1.0, 0.0])
        v2 = _unit([0.6, 0.8])  # cosine = 0.6
        vecs = np.stack([v1, v2])
        chunks = ["A", "B"]
        embedder = MockEmbedder(vecs)

        score, _ = compute_trust_score([0, 1], chunks, embedder)
        assert abs(score - 0.6) < 1e-4


# ---------------------------------------------------------------------------
# FAISS reconstruct path (preferred in production)
# ---------------------------------------------------------------------------

class TestFAISSReconstructPath:
    def test_reconstruct_high_confidence(self):
        """Using faiss_index instead of embedder, identical vectors → score 1."""
        vec = _unit([1.0, 0.0, 0.0])
        vecs = np.stack([vec, vec, vec])
        chunks = ["A", "B", "C"]
        faiss_idx = MockFAISSIndex(vecs)
        # embedder is intentionally wrong — should NOT be called
        embedder = MockEmbedder(np.zeros((3, 3), dtype="float32"))

        score, low_conf = compute_trust_score(
            [0, 1, 2], chunks, embedder, faiss_index=faiss_idx
        )
        assert abs(score - 1.0) < 1e-5
        assert low_conf is False

    def test_reconstruct_low_confidence(self):
        """Orthogonal pre-stored vectors → low confidence via reconstruct path."""
        v1 = _unit([1.0, 0.0, 0.0])
        v2 = _unit([0.0, 1.0, 0.0])
        v3 = _unit([0.0, 0.0, 1.0])
        vecs = np.stack([v1, v2, v3])
        chunks = ["A", "B", "C"]
        faiss_idx = MockFAISSIndex(vecs)
        embedder = MockEmbedder(np.zeros((3, 3), dtype="float32"))

        score, low_conf = compute_trust_score(
            [0, 1, 2], chunks, embedder, faiss_index=faiss_idx,
            low_confidence_threshold=0.30
        )
        assert score < 0.30
        assert low_conf is True

    def test_reconstruct_result_matches_encode_for_same_vectors(self):
        """Both paths should produce the same score when given the same vectors."""
        v1 = _unit([1.0, 0.5, 0.0])
        v2 = _unit([0.8, 0.6, 0.0])
        vecs = np.stack([v1, v2])
        chunks = ["A", "B"]

        score_embed, _ = compute_trust_score(
            [0, 1], chunks, MockEmbedder(vecs)
        )
        score_faiss, _ = compute_trust_score(
            [0, 1], chunks, MockEmbedder(vecs), faiss_index=MockFAISSIndex(vecs)
        )
        assert abs(score_embed - score_faiss) < 1e-4

    def test_reconstruct_fallback_on_exception(self):
        """If reconstruct raises, function falls back to embedder gracefully."""
        class BrokenIndex:
            def reconstruct(self, i):
                raise RuntimeError("broken")

        v1 = _unit([1.0, 0.0])
        v2 = _unit([0.0, 1.0])
        vecs = np.stack([v1, v2])
        chunks = ["A", "B"]

        # Should fall back to embedder without raising
        score, low_conf = compute_trust_score(
            [0, 1], chunks, MockEmbedder(vecs), faiss_index=BrokenIndex()
        )
        # Orthogonal → low confidence
        assert score < 0.30
        assert low_conf is True
