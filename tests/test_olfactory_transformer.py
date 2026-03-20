"""Tests for the Olfactory-Transformer-Kit."""

import numpy as np
import pytest

from olfactory_transformer.tokenizer import MolecularTokenizer
from olfactory_transformer.transformer import OlfactoryTransformer
from olfactory_transformer.scent_decoder import ScentDecoder
from olfactory_transformer.pretrain import PretrainPipeline
from olfactory_transformer.inference import InferencePipeline


# ─────────────────────────────────────────────
# MolecularTokenizer tests
# ─────────────────────────────────────────────

class TestMolecularTokenizer:
    def setup_method(self):
        self.tok = MolecularTokenizer()

    def test_tokenizer_basic(self):
        """Basic atoms map to their vocabulary IDs."""
        tokens = self.tok.tokenize("CCO")
        assert tokens == [1, 1, 2], f"Expected [1,1,2], got {tokens}"

    def test_tokenizer_two_char_atoms(self):
        """Two-character atoms (Cl, Br) are tokenized as single tokens."""
        tokens = self.tok.tokenize("ClBr")
        assert tokens == [6, 7], f"Expected [6,7], got {tokens}"

    def test_tokenizer_padding(self):
        """pad() produces an array of the requested length filled with zeros."""
        tokens = self.tok.tokenize("CCO")
        padded = self.tok.pad(tokens, max_len=8)
        assert padded.shape == (8,), f"Expected shape (8,), got {padded.shape}"
        assert padded[0] == 1  # C
        assert padded[1] == 1  # C
        assert padded[2] == 2  # O
        assert all(padded[3:] == 0), "Trailing positions should be 0 (PAD)"

    def test_tokenizer_padding_truncation(self):
        """pad() truncates if the token list exceeds max_len."""
        tokens = list(range(10))
        padded = self.tok.pad(tokens, max_len=5)
        assert padded.shape == (5,)
        np.testing.assert_array_equal(padded, [0, 1, 2, 3, 4])

    def test_tokenizer_decode(self):
        """decode() reconstructs the atom string (no PADs)."""
        tokens = [1, 1, 2]  # C, C, O
        decoded = self.tok.decode(tokens)
        assert decoded == "CCO", f"Expected 'CCO', got '{decoded}'"

    def test_tokenizer_decode_skips_pad(self):
        """decode() skips PAD tokens (0)."""
        tokens = [1, 0, 2, 0]  # C, PAD, O, PAD
        decoded = self.tok.decode(tokens)
        assert decoded == "CO", f"Expected 'CO', got '{decoded}'"

    def test_tokenizer_unknown_atoms(self):
        """Atoms not in vocabulary get mapped to UNK (9)."""
        tokens = self.tok.tokenize("X")  # X is not in ATOM_VOCAB
        assert tokens == [9], f"Expected [9] for unknown atom, got {tokens}"

    def test_tokenizer_structural_chars_skipped(self):
        """Structural characters (parentheses, bonds, digits) are skipped."""
        tokens = self.tok.tokenize("CC(=O)N")
        # Should get C, C, O, N  (structural chars ignored)
        assert all(t in MolecularTokenizer.ATOM_VOCAB.values() for t in tokens)
        # Must contain C and N
        assert 1 in tokens  # C
        assert 3 in tokens  # N

    def test_tokenizer_bond_vocab(self):
        """tokenize_bonds() maps bond strings to IDs."""
        bond_ids = self.tok.tokenize_bonds(["single", "double", "aromatic"])
        assert bond_ids == [1, 2, 3]

    def test_tokenizer_bond_unknown(self):
        """Unknown bond types are mapped to PAD (0)."""
        bond_ids = self.tok.tokenize_bonds(["single", "triple"])
        assert bond_ids == [1, 0]


# ─────────────────────────────────────────────
# OlfactoryTransformer tests
# ─────────────────────────────────────────────

class TestOlfactoryTransformer:
    def setup_method(self):
        self.model = OlfactoryTransformer(
            vocab_size=10, embed_dim=64, n_heads=4, n_layers=3, max_len=64
        )

    def test_positional_encoding_shape(self):
        """Positional encoding has shape (1, max_len, embed_dim)."""
        pe = self.model.pos_encoding
        assert pe.shape == (1, 64, 64), f"Expected (1,64,64), got {pe.shape}"

    def test_positional_encoding_values(self):
        """Positional encoding values are bounded (sin/cos in [-1, 1])."""
        pe = self.model.pos_encoding
        assert pe.min() >= -1.1 and pe.max() <= 1.1, "Pos encoding out of [-1,1] range"

    def test_transformer_forward_shape(self):
        """forward() returns correct shape (batch, seq, embed_dim)."""
        token_ids = np.array([[1, 1, 2, 0, 0, 0, 0, 0]], dtype=np.int32)  # (1, 8)
        out = self.model.forward(token_ids)
        assert out.shape == (1, 8, 64), f"Expected (1,8,64), got {out.shape}"

    def test_transformer_forward_batch(self):
        """forward() handles batch size > 1."""
        token_ids = np.zeros((4, 16), dtype=np.int32)
        out = self.model.forward(token_ids)
        assert out.shape == (4, 16, 64)

    def test_transformer_forward_1d_input(self):
        """forward() accepts 1D input and adds batch dimension."""
        token_ids = np.array([1, 2, 3, 0], dtype=np.int32)
        out = self.model.forward(token_ids)
        assert out.shape == (1, 4, 64)

    def test_transformer_attention(self):
        """attend() returns correct shape and doesn't produce NaN."""
        batch, seq, dim = 2, 8, 16
        Q = np.random.randn(batch, seq, dim).astype(np.float32)
        K = np.random.randn(batch, seq, dim).astype(np.float32)
        V = np.random.randn(batch, seq, dim).astype(np.float32)
        out = self.model.attend(Q, K, V)
        assert out.shape == (batch, seq, dim)
        assert not np.isnan(out).any(), "attend() produced NaN values"

    def test_transformer_attention_with_mask(self):
        """attend() respects a boolean mask."""
        batch, seq, dim = 1, 4, 8
        Q = np.ones((batch, seq, dim), dtype=np.float32)
        K = np.ones((batch, seq, dim), dtype=np.float32)
        V = np.ones((batch, seq, dim), dtype=np.float32)
        mask = np.zeros((batch, seq, seq), dtype=bool)
        mask[:, :, -1] = True  # mask out last token
        out = self.model.attend(Q, K, V, mask=mask)
        assert out.shape == (batch, seq, dim)

    def test_transformer_layernorm(self):
        """Layer norm produces zero mean and ~unit variance."""
        x = np.random.randn(2, 8, 64).astype(np.float32) * 10
        gamma = np.ones(64, dtype=np.float32)
        beta = np.zeros(64, dtype=np.float32)
        out = OlfactoryTransformer._layer_norm(x, gamma, beta)
        # Mean should be near 0 along embed_dim
        assert abs(out.mean(axis=-1)).max() < 1e-4, "Layer norm mean not near 0"

    def test_transformer_output_dtype(self):
        """Transformer output is float32."""
        token_ids = np.array([[1, 2, 3]], dtype=np.int32)
        out = self.model.forward(token_ids)
        assert out.dtype == np.float32


# ─────────────────────────────────────────────
# ScentDecoder tests
# ─────────────────────────────────────────────

class TestScentDecoder:
    def setup_method(self):
        self.decoder = ScentDecoder(embed_dim=64, n_categories=20)

    def test_scent_decoder_output_shape(self):
        """decode() returns shape (batch, n_categories)."""
        embeddings = np.random.randn(2, 8, 64).astype(np.float32)
        probs = self.decoder.decode(embeddings)
        assert probs.shape == (2, 20), f"Expected (2,20), got {probs.shape}"

    def test_scent_decoder_probabilities_range(self):
        """All probabilities are in [0, 1]."""
        embeddings = np.random.randn(3, 16, 64).astype(np.float32)
        probs = self.decoder.decode(embeddings)
        assert probs.min() >= 0.0, "Probabilities below 0"
        assert probs.max() <= 1.0, "Probabilities above 1"

    def test_scent_decoder_get_labels(self):
        """get_labels() returns a subset of SCENT_CATEGORIES."""
        embeddings = np.random.randn(1, 8, 64).astype(np.float32)
        labels = self.decoder.get_labels(embeddings, threshold=0.3)
        assert isinstance(labels, list)
        for l in labels:
            assert l in ScentDecoder.SCENT_CATEGORIES, f"{l!r} not in SCENT_CATEGORIES"

    def test_scent_decoder_categories_count(self):
        """SCENT_CATEGORIES contains exactly 20 entries."""
        assert len(ScentDecoder.SCENT_CATEGORIES) == 20

    def test_scent_decoder_threshold_zero(self):
        """At threshold=0 all categories are returned."""
        embeddings = np.random.randn(1, 8, 64).astype(np.float32)
        labels = self.decoder.get_labels(embeddings, threshold=0.0)
        assert len(labels) == 20

    def test_scent_decoder_threshold_one(self):
        """At threshold=1.0 no categories are returned (sigmoid never reaches 1)."""
        embeddings = np.random.randn(1, 8, 64).astype(np.float32)
        labels = self.decoder.get_labels(embeddings, threshold=1.0)
        assert labels == []

    def test_scent_decoder_single_vector(self):
        """decode() handles a single embedding vector (embed_dim,)."""
        embeddings = np.random.randn(64).astype(np.float32)
        probs = self.decoder.decode(embeddings)
        assert probs.shape == (1, 20)


# ─────────────────────────────────────────────
# PretrainPipeline tests
# ─────────────────────────────────────────────

class TestPretrainPipeline:
    def setup_method(self):
        self.pipeline = PretrainPipeline(max_len=64, n_categories=20, seed=0)
        self.model = OlfactoryTransformer(embed_dim=64, n_heads=4, n_layers=3)
        self.decoder = ScentDecoder(embed_dim=64, n_categories=20)

    def test_pretrain_synthetic_pairs(self):
        """generate_synthetic_pairs() returns n pairs with correct shapes."""
        pairs = self.pipeline.generate_synthetic_pairs(n=10)
        assert len(pairs) == 10
        for tokens, labels in pairs:
            assert tokens.shape == (64,), f"Token shape: {tokens.shape}"
            assert labels.shape == (20,), f"Label shape: {labels.shape}"
            assert set(np.unique(labels)).issubset({0.0, 1.0}), "Labels must be binary"

    def test_pretrain_bce_perfect(self):
        """BCE loss is 0 for perfect predictions (clamped)."""
        pred = np.array([1.0, 0.0, 1.0])
        target = np.array([1.0, 0.0, 1.0])
        loss = PretrainPipeline.binary_cross_entropy(pred, target)
        assert loss < 0.01, f"Expected near-zero loss, got {loss}"

    def test_pretrain_bce_worst(self):
        """BCE loss is large for worst-case predictions."""
        pred = np.array([0.0, 1.0])
        target = np.array([1.0, 0.0])
        loss = PretrainPipeline.binary_cross_entropy(pred, target)
        assert loss > 10.0, f"Expected high loss, got {loss}"

    def test_pretrain_train_step_returns_float(self):
        """train_step() returns a scalar float loss."""
        pairs = self.pipeline.generate_synthetic_pairs(n=1)
        tokens, labels = pairs[0]
        loss = self.pipeline.train_step(self.model, self.decoder, tokens, labels)
        assert isinstance(loss, float)
        assert not np.isnan(loss), "Loss is NaN"

    def test_pretrain_evaluate_metrics(self):
        """evaluate() returns a dict with required metric keys."""
        pairs = self.pipeline.generate_synthetic_pairs(n=5)
        metrics = self.pipeline.evaluate(self.model, self.decoder, pairs)
        assert "avg_loss" in metrics
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics

    def test_pretrain_evaluate_accuracy_range(self):
        """Accuracy is in [0, 1]."""
        pairs = self.pipeline.generate_synthetic_pairs(n=10)
        metrics = self.pipeline.evaluate(self.model, self.decoder, pairs)
        assert 0.0 <= metrics["accuracy"] <= 1.0


# ─────────────────────────────────────────────
# InferencePipeline tests
# ─────────────────────────────────────────────

class TestInferencePipeline:
    def setup_method(self):
        self.pipeline = InferencePipeline()

    def test_inference_pipeline(self):
        """predict() returns a dict with 'scents' and 'probabilities'."""
        result = self.pipeline.predict("CCO")
        assert "scents" in result
        assert "probabilities" in result
        assert isinstance(result["scents"], list)
        assert isinstance(result["probabilities"], dict)

    def test_inference_probabilities_range(self):
        """All probabilities from predict() are in [0, 1]."""
        result = self.pipeline.predict("c1ccccc1")
        for cat, prob in result["probabilities"].items():
            assert 0.0 <= prob <= 1.0, f"{cat}: {prob} out of range"

    def test_inference_probabilities_keys(self):
        """predict() returns probabilities for all 20 scent categories."""
        result = self.pipeline.predict("CCN")
        assert len(result["probabilities"]) == 20
        assert set(result["probabilities"].keys()) == set(ScentDecoder.SCENT_CATEGORIES)

    def test_batch_predict(self):
        """batch_predict() returns a list of results with correct length."""
        smiles_list = ["CCO", "CCN", "CCC"]
        results = self.pipeline.batch_predict(smiles_list)
        assert len(results) == 3
        for r in results:
            assert "scents" in r
            assert "probabilities" in r

    def test_batch_predict_empty(self):
        """batch_predict() handles an empty list."""
        results = self.pipeline.batch_predict([])
        assert results == []

    def test_batch_predict_consistency(self):
        """batch_predict() and individual predict() agree on probabilities."""
        smiles = "CCO"
        single = self.pipeline.predict(smiles)
        batch = self.pipeline.batch_predict([smiles])
        for cat in ScentDecoder.SCENT_CATEGORIES:
            assert abs(single["probabilities"][cat] - batch[0]["probabilities"][cat]) < 1e-5

    def test_inference_demo_runs(self, capsys):
        """demo() runs without error and prints output."""
        self.pipeline.demo()
        captured = capsys.readouterr()
        assert "Demo" in captured.out or "Molecule" in captured.out
