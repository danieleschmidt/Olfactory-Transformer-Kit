"""OlfactoryTransformer: Pure numpy 3-layer encoder-only transformer."""

import numpy as np
from typing import Optional


class OlfactoryTransformer:
    """3-layer encoder-only transformer for molecular olfaction using pure numpy.

    Args:
        vocab_size: Size of the token vocabulary (default 10).
        embed_dim: Embedding dimensionality (default 64).
        n_heads: Number of attention heads (default 4).
        n_layers: Number of transformer encoder layers (default 3).
        max_len: Maximum sequence length (default 64).
    """

    def __init__(
        self,
        vocab_size: int = 10,
        embed_dim: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        max_len: int = 64,
        seed: int = 42,
    ):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.max_len = max_len
        self.head_dim = embed_dim // n_heads

        assert embed_dim % n_heads == 0, "embed_dim must be divisible by n_heads"

        rng = np.random.default_rng(seed)

        # Token embedding matrix: vocab_size x embed_dim
        self.embedding = rng.normal(0, 0.02, (vocab_size, embed_dim)).astype(np.float32)

        # Positional encoding (sinusoidal, precomputed)
        self.pos_encoding = self._build_pos_encoding(max_len, embed_dim)

        # Per-layer weights
        self.layers = []
        for _ in range(n_layers):
            layer = {
                # Multi-head self-attention projections
                "W_Q": rng.normal(0, 0.02, (embed_dim, embed_dim)).astype(np.float32),
                "W_K": rng.normal(0, 0.02, (embed_dim, embed_dim)).astype(np.float32),
                "W_V": rng.normal(0, 0.02, (embed_dim, embed_dim)).astype(np.float32),
                "W_O": rng.normal(0, 0.02, (embed_dim, embed_dim)).astype(np.float32),
                # Feed-forward layers
                "W_FF1": rng.normal(0, 0.02, (embed_dim, embed_dim * 4)).astype(np.float32),
                "b_FF1": np.zeros(embed_dim * 4, dtype=np.float32),
                "W_FF2": rng.normal(0, 0.02, (embed_dim * 4, embed_dim)).astype(np.float32),
                "b_FF2": np.zeros(embed_dim, dtype=np.float32),
                # Layer norm parameters (attn and ff)
                "ln1_gamma": np.ones(embed_dim, dtype=np.float32),
                "ln1_beta": np.zeros(embed_dim, dtype=np.float32),
                "ln2_gamma": np.ones(embed_dim, dtype=np.float32),
                "ln2_beta": np.zeros(embed_dim, dtype=np.float32),
            }
            self.layers.append(layer)

    @staticmethod
    def _build_pos_encoding(max_len: int, embed_dim: int) -> np.ndarray:
        """Build sinusoidal positional encoding matrix.

        Returns:
            Array of shape (1, max_len, embed_dim).
        """
        pos = np.arange(max_len)[:, np.newaxis]  # (max_len, 1)
        i = np.arange(embed_dim)[np.newaxis, :]  # (1, embed_dim)
        angles = pos / np.power(10000, (2 * (i // 2)) / embed_dim)
        # Apply sin to even indices, cos to odd indices
        angles[:, 0::2] = np.sin(angles[:, 0::2])
        angles[:, 1::2] = np.cos(angles[:, 1::2])
        return angles[np.newaxis, :, :].astype(np.float32)  # (1, max_len, embed_dim)

    @staticmethod
    def _layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """Apply layer normalization.

        Args:
            x: Input array of shape (..., embed_dim).
            gamma: Scale parameter.
            beta: Bias parameter.
            eps: Small constant for numerical stability.

        Returns:
            Normalized array of same shape as x.
        """
        mean = x.mean(axis=-1, keepdims=True)
        std = x.std(axis=-1, keepdims=True)
        return (gamma * (x - mean) / (std + eps) + beta).astype(x.dtype)

    @staticmethod
    def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Numerically stable softmax."""
        x_max = x.max(axis=axis, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / (exp_x.sum(axis=axis, keepdims=True) + 1e-9)

    @staticmethod
    def _relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def attend(
        self,
        Q: np.ndarray,
        K: np.ndarray,
        V: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Scaled dot-product attention.

        Args:
            Q: Query matrix of shape (batch, seq, dim).
            K: Key matrix of shape (batch, seq, dim).
            V: Value matrix of shape (batch, seq, dim).
            mask: Optional boolean mask of shape (batch, seq, seq), True=mask out.

        Returns:
            Attention output of shape (batch, seq, dim).
        """
        d_k = Q.shape[-1]
        scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(d_k)  # (batch, seq, seq)
        if mask is not None:
            scores = np.where(mask, -1e9, scores)
        attn_weights = self._softmax(scores, axis=-1)
        return np.matmul(attn_weights, V)  # (batch, seq, dim)

    def _multihead_attention(
        self,
        x: np.ndarray,
        layer: dict,
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Multi-head self-attention.

        Args:
            x: Input of shape (batch, seq, embed_dim).
            layer: Dict of layer weights.
            mask: Optional attention mask.

        Returns:
            Output of shape (batch, seq, embed_dim).
        """
        batch, seq, _ = x.shape

        # Project to Q, K, V
        Q = x @ layer["W_Q"]  # (batch, seq, embed_dim)
        K = x @ layer["W_K"]
        V = x @ layer["W_V"]

        # Split into heads: (batch, seq, n_heads, head_dim) -> (batch, n_heads, seq, head_dim)
        def split_heads(t):
            t = t.reshape(batch, seq, self.n_heads, self.head_dim)
            return t.transpose(0, 2, 1, 3)  # (batch, n_heads, seq, head_dim)

        Q_h = split_heads(Q)
        K_h = split_heads(K)
        V_h = split_heads(V)

        # Flatten batch and heads for attend()
        bnh = batch * self.n_heads
        Q_flat = Q_h.reshape(bnh, seq, self.head_dim)
        K_flat = K_h.reshape(bnh, seq, self.head_dim)
        V_flat = V_h.reshape(bnh, seq, self.head_dim)

        # Expand mask if provided
        if mask is not None:
            mask_flat = np.tile(mask[:, np.newaxis, :, :], (1, self.n_heads, 1, 1))
            mask_flat = mask_flat.reshape(bnh, seq, seq)
        else:
            mask_flat = None

        attn_out = self.attend(Q_flat, K_flat, V_flat, mask_flat)  # (bnh, seq, head_dim)

        # Merge heads: (batch, n_heads, seq, head_dim) -> (batch, seq, embed_dim)
        attn_out = attn_out.reshape(batch, self.n_heads, seq, self.head_dim)
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(batch, seq, self.embed_dim)

        # Output projection
        return attn_out @ layer["W_O"]

    def forward(self, token_ids: np.ndarray) -> np.ndarray:
        """Run a forward pass through the transformer.

        Args:
            token_ids: Integer array of shape (batch, seq) with token indices.

        Returns:
            Contextual embeddings of shape (batch, seq, embed_dim).
        """
        token_ids = np.asarray(token_ids, dtype=np.int32)
        if token_ids.ndim == 1:
            token_ids = token_ids[np.newaxis, :]  # add batch dim

        batch, seq = token_ids.shape

        # Token embedding lookup
        x = self.embedding[token_ids]  # (batch, seq, embed_dim)

        # Add positional encoding (truncate to actual seq len)
        x = x + self.pos_encoding[:, :seq, :]

        # Pass through encoder layers
        for layer in self.layers:
            # --- Self-attention sub-layer with residual connection ---
            attn_out = self._multihead_attention(x, layer)
            x = self._layer_norm(x + attn_out, layer["ln1_gamma"], layer["ln1_beta"])

            # --- Feed-forward sub-layer with residual connection ---
            ff = self._relu(x @ layer["W_FF1"] + layer["b_FF1"])  # (batch, seq, 4*embed_dim)
            ff = ff @ layer["W_FF2"] + layer["b_FF2"]  # (batch, seq, embed_dim)
            x = self._layer_norm(x + ff, layer["ln2_gamma"], layer["ln2_beta"])

        return x.astype(np.float32)  # (batch, seq, embed_dim)
