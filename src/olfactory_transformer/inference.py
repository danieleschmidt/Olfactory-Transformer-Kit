"""InferencePipeline: End-to-end prediction from SMILES to scent descriptions."""

import numpy as np
from typing import List, Dict

from .tokenizer import MolecularTokenizer
from .transformer import OlfactoryTransformer
from .scent_decoder import ScentDecoder


# Example molecules for the demo
_DEMO_MOLECULES = [
    ("CCO", "ethanol"),
    ("CC(=O)O", "acetic acid (vinegar)"),
    ("c1ccccc1", "benzene"),
    ("CC(C)CCC=O", "isovaleraldehyde"),
    ("CC(=O)Oc1ccccc1C(=O)O", "aspirin"),
]


class InferencePipeline:
    """End-to-end pipeline for predicting scent descriptors from SMILES strings.

    Loads a MolecularTokenizer, OlfactoryTransformer, and ScentDecoder
    with default parameters (randomly initialized weights — not trained).
    """

    def __init__(
        self,
        vocab_size: int = 10,
        embed_dim: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        max_len: int = 64,
        n_categories: int = 20,
        threshold: float = 0.3,
    ):
        self.tokenizer = MolecularTokenizer()
        self.transformer = OlfactoryTransformer(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            max_len=max_len,
        )
        self.decoder = ScentDecoder(
            embed_dim=embed_dim,
            n_categories=n_categories,
        )
        self.max_len = max_len
        self.threshold = threshold

    def predict(self, smiles: str) -> Dict:
        """Predict scent descriptors for a single SMILES string.

        Args:
            smiles: SMILES-like molecular string.

        Returns:
            Dict with:
            - "scents": list of predicted scent category strings
            - "probabilities": dict mapping category name to probability
        """
        tokens = self.tokenizer.tokenize(smiles)
        padded = self.tokenizer.pad(tokens, self.max_len)
        token_ids = padded[np.newaxis, :]  # (1, seq)

        embeddings = self.transformer.forward(token_ids)  # (1, seq, embed_dim)
        probs = self.decoder.decode(embeddings)           # (1, n_categories)
        probs_1d = probs[0]                               # (n_categories,)

        labels = self.decoder.get_labels(embeddings, threshold=self.threshold)
        prob_dict = {
            cat: float(probs_1d[i])
            for i, cat in enumerate(ScentDecoder.SCENT_CATEGORIES)
        }

        return {
            "scents": labels,
            "probabilities": prob_dict,
        }

    def batch_predict(self, smiles_list: List[str]) -> List[Dict]:
        """Predict scent descriptors for a batch of SMILES strings.

        Args:
            smiles_list: List of SMILES-like molecular strings.

        Returns:
            List of prediction dicts (same format as predict()).
        """
        if not smiles_list:
            return []

        batch_tokens = []
        for smiles in smiles_list:
            tokens = self.tokenizer.tokenize(smiles)
            padded = self.tokenizer.pad(tokens, self.max_len)
            batch_tokens.append(padded)

        token_ids = np.stack(batch_tokens, axis=0)  # (batch, seq)
        embeddings = self.transformer.forward(token_ids)  # (batch, seq, embed_dim)
        all_probs = self.decoder.decode(embeddings)       # (batch, n_categories)

        results = []
        for i, smiles in enumerate(smiles_list):
            probs_1d = all_probs[i]
            labels = [
                ScentDecoder.SCENT_CATEGORIES[j]
                for j in range(len(ScentDecoder.SCENT_CATEGORIES))
                if probs_1d[j] >= self.threshold
            ]
            prob_dict = {
                cat: float(probs_1d[j])
                for j, cat in enumerate(ScentDecoder.SCENT_CATEGORIES)
            }
            results.append({"scents": labels, "probabilities": prob_dict})
        return results

    def demo(self) -> None:
        """Run inference on 5 example molecules and print results."""
        print("=== Olfactory Transformer Demo ===\n")
        for smiles, name in _DEMO_MOLECULES:
            result = self.predict(smiles)
            print(f"Molecule: {name} ({smiles})")
            print(f"  Predicted scents: {result['scents'] or ['(none above threshold)']}")
            top3 = sorted(
                result["probabilities"].items(), key=lambda x: -x[1]
            )[:3]
            print(f"  Top-3 probabilities: {[(k, f'{v:.3f}') for k, v in top3]}")
            print()
