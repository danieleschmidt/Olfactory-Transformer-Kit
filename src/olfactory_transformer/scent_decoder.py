"""ScentDecoder: Multi-label classifier for 20 GoodScents-derived scent categories."""

import numpy as np
from typing import List


class ScentDecoder:
    """Linear decoder from transformer embeddings to scent category probabilities.

    Uses a linear layer (embed_dim -> n_categories) followed by sigmoid activation
    to produce multi-label classification probabilities.

    Args:
        embed_dim: Dimensionality of input embeddings (default 64).
        n_categories: Number of scent categories (default 20).
        seed: Random seed for weight initialization.
    """

    SCENT_CATEGORIES: List[str] = [
        "floral", "citrus", "woody", "musky", "fresh",
        "sweet", "spicy", "earthy", "marine", "green",
        "fruity", "herbal", "resinous", "animalic", "powdery",
        "smoky", "balsamic", "aldehydic", "lactonic", "waxy",
    ]

    def __init__(self, embed_dim: int = 64, n_categories: int = 20, seed: int = 42):
        self.embed_dim = embed_dim
        self.n_categories = n_categories

        rng = np.random.default_rng(seed)
        # Linear layer: embed_dim -> n_categories
        self.W = rng.normal(0, 0.02, (embed_dim, n_categories)).astype(np.float32)
        self.b = np.zeros(n_categories, dtype=np.float32)

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid."""
        return np.where(
            x >= 0,
            1.0 / (1.0 + np.exp(-x)),
            np.exp(x) / (1.0 + np.exp(x)),
        )

    def _pool_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Mean-pool sequence embeddings to get a single vector per example.

        Args:
            embeddings: Array of shape (batch, seq, embed_dim) or (seq, embed_dim)
                        or (embed_dim,).

        Returns:
            Array of shape (batch, embed_dim).
        """
        if embeddings.ndim == 1:
            return embeddings[np.newaxis, :]  # (1, embed_dim)
        elif embeddings.ndim == 2:
            # (seq, embed_dim) -> mean over seq -> (1, embed_dim)
            return embeddings.mean(axis=0, keepdims=True)
        else:
            # (batch, seq, embed_dim) -> mean over seq -> (batch, embed_dim)
            return embeddings.mean(axis=1)

    def decode(self, embeddings: np.ndarray) -> np.ndarray:
        """Decode embeddings into sigmoid probabilities for each scent category.

        Args:
            embeddings: Transformer output of shape (batch, seq, embed_dim),
                        (seq, embed_dim), or (embed_dim,).

        Returns:
            Array of shape (batch, n_categories) with sigmoid probabilities in [0, 1].
        """
        pooled = self._pool_embeddings(embeddings)  # (batch, embed_dim)
        logits = pooled @ self.W + self.b  # (batch, n_categories)
        return self._sigmoid(logits)

    def get_labels(self, embeddings: np.ndarray, threshold: float = 0.3) -> List[str]:
        """Get scent category labels exceeding the threshold probability.

        For batched input, uses the first example in the batch.

        Args:
            embeddings: Transformer output embeddings.
            threshold: Probability threshold (default 0.3).

        Returns:
            List of scent category label strings.
        """
        probs = self.decode(embeddings)  # (batch, n_categories)
        # Use first example if batched
        p = probs[0] if probs.ndim > 1 else probs
        return [
            self.SCENT_CATEGORIES[i]
            for i in range(self.n_categories)
            if p[i] >= threshold
        ]
