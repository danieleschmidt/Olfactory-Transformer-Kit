"""PretrainPipeline: Synthetic data generation and evaluation for the olfactory transformer."""

import numpy as np
from typing import List, Tuple, Dict

from .tokenizer import MolecularTokenizer
from .transformer import OlfactoryTransformer
from .scent_decoder import ScentDecoder


# Simple SMILES fragments for synthetic molecule generation
_SMILES_FRAGMENTS = [
    "CCO", "CC(=O)O", "c1ccccc1", "CCN", "CCCC",
    "CC(C)CC", "OCC", "NCC", "SCC", "FCC",
    "ClCC", "BrCC", "PCCC", "CC#N", "C=CC",
    "CC(=O)N", "c1ccncc1", "CC(O)CC", "CCCCCC", "CCOCC",
]


class PretrainPipeline:
    """Pipeline for synthetic pre-training of the olfactory transformer.

    Generates random (molecule_tokens, scent_labels) pairs for training experiments.
    Note: this implementation provides only a forward pass (no backpropagation).
    """

    def __init__(self, max_len: int = 64, n_categories: int = 20, seed: int = 42):
        self.max_len = max_len
        self.n_categories = n_categories
        self.rng = np.random.default_rng(seed)
        self.tokenizer = MolecularTokenizer()

    def generate_synthetic_pairs(
        self, n: int = 100
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate synthetic (token_ids, scent_labels) pairs.

        Each pair contains:
        - token_ids: padded token array of shape (max_len,)
        - scent_labels: binary label array of shape (n_categories,)

        Args:
            n: Number of pairs to generate (default 100).

        Returns:
            List of (token_ids, scent_labels) tuples.
        """
        pairs = []
        n_frags = len(_SMILES_FRAGMENTS)
        for i in range(n):
            # Pick a random SMILES fragment
            smiles = _SMILES_FRAGMENTS[i % n_frags]
            tokens = self.tokenizer.tokenize(smiles)
            padded = self.tokenizer.pad(tokens, self.max_len)

            # Random multi-label scent labels (binary)
            n_active = self.rng.integers(1, 6)  # 1-5 active categories
            label = np.zeros(self.n_categories, dtype=np.float32)
            active_idx = self.rng.choice(self.n_categories, size=n_active, replace=False)
            label[active_idx] = 1.0

            pairs.append((padded, label))
        return pairs

    @staticmethod
    def binary_cross_entropy(pred: np.ndarray, target: np.ndarray) -> float:
        """Compute binary cross-entropy loss.

        Args:
            pred: Predicted probabilities in [0, 1], shape (n,) or (batch, n).
            target: Binary targets, same shape as pred.

        Returns:
            Scalar loss value (float).
        """
        eps = 1e-7
        pred = np.clip(pred, eps, 1 - eps)
        loss = -(target * np.log(pred) + (1 - target) * np.log(1 - pred))
        return float(loss.mean())

    def train_step(
        self,
        model: OlfactoryTransformer,
        decoder: ScentDecoder,
        tokens: np.ndarray,
        labels: np.ndarray,
    ) -> float:
        """Perform a single forward pass and compute the loss (no backprop).

        Args:
            model: OlfactoryTransformer instance.
            decoder: ScentDecoder instance.
            tokens: Token IDs of shape (batch, seq) or (seq,).
            labels: Binary scent labels of shape (batch, n_categories) or (n_categories,).

        Returns:
            Scalar loss value (float).
        """
        # Ensure batch dimension
        if tokens.ndim == 1:
            tokens = tokens[np.newaxis, :]
        if labels.ndim == 1:
            labels = labels[np.newaxis, :]

        embeddings = model.forward(tokens)         # (batch, seq, embed_dim)
        probs = decoder.decode(embeddings)          # (batch, n_categories)
        loss = self.binary_cross_entropy(probs, labels)
        return loss

    def evaluate(
        self,
        model: OlfactoryTransformer,
        decoder: ScentDecoder,
        pairs: List[Tuple[np.ndarray, np.ndarray]],
    ) -> Dict[str, float]:
        """Evaluate the model on a list of (token_ids, labels) pairs.

        Computes:
        - avg_loss: Mean binary cross-entropy across all pairs.
        - accuracy: Fraction of label elements correctly predicted (threshold=0.5).
        - precision: TP / (TP + FP) across all predictions.
        - recall: TP / (TP + FN) across all true positives.

        Args:
            model: OlfactoryTransformer instance.
            decoder: ScentDecoder instance.
            pairs: List of (token_ids, labels) tuples.

        Returns:
            Dict with metric names as keys and float values.
        """
        total_loss = 0.0
        all_preds = []
        all_targets = []

        for tokens, labels in pairs:
            t = tokens[np.newaxis, :] if tokens.ndim == 1 else tokens
            l = labels[np.newaxis, :] if labels.ndim == 1 else labels

            embeddings = model.forward(t)
            probs = decoder.decode(embeddings)
            total_loss += self.binary_cross_entropy(probs, l)

            preds = (probs >= 0.5).astype(np.float32)
            all_preds.append(preds)
            all_targets.append(l)

        avg_loss = total_loss / len(pairs) if pairs else 0.0

        all_preds = np.vstack(all_preds)
        all_targets = np.vstack(all_targets)

        # Element-wise accuracy
        accuracy = float((all_preds == all_targets).mean())

        # Precision and recall
        tp = float((all_preds * all_targets).sum())
        fp = float((all_preds * (1 - all_targets)).sum())
        fn = float(((1 - all_preds) * all_targets).sum())
        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)

        return {
            "avg_loss": avg_loss,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
        }
