"""MolecularTokenizer: atom/bond vocabulary tokenization for SMILES-like strings."""

import re
import numpy as np
from typing import List


class MolecularTokenizer:
    """Tokenizer for molecular SMILES-like strings using atom and bond vocabularies."""

    ATOM_VOCAB = {
        "PAD": 0,
        "C": 1,
        "O": 2,
        "N": 3,
        "S": 4,
        "F": 5,
        "Cl": 6,
        "Br": 7,
        "P": 8,
        "UNK": 9,
    }

    BOND_VOCAB = {
        "PAD": 0,
        "single": 1,
        "double": 2,
        "aromatic": 3,
    }

    # Reverse mapping for decoding
    ATOM_VOCAB_INV = {v: k for k, v in ATOM_VOCAB.items()}
    BOND_VOCAB_INV = {v: k for k, v in BOND_VOCAB.items()}

    # Two-character atoms must be matched before single-char atoms
    _ATOM_PATTERN = re.compile(r"(Cl|Br|[CNOSFPcnos]|[A-Z])")

    def tokenize(self, smiles_like: str) -> List[int]:
        """Tokenize a SMILES-like string into a list of atom token IDs.

        Args:
            smiles_like: A SMILES-like molecular string.

        Returns:
            List of integer token IDs corresponding to atoms.
        """
        tokens = []
        i = 0
        s = smiles_like
        while i < len(s):
            # Try two-char atoms first
            if i + 1 < len(s) and s[i:i+2] in self.ATOM_VOCAB:
                tokens.append(self.ATOM_VOCAB[s[i:i+2]])
                i += 2
            elif s[i].upper() in self.ATOM_VOCAB:
                # Handle lowercase aromatic atoms (c, n, o, s) as their uppercase equivalents
                atom = s[i].upper()
                tokens.append(self.ATOM_VOCAB.get(atom, self.ATOM_VOCAB["UNK"]))
                i += 1
            elif s[i] in "()[]=#@\\/.+-0123456789%\n\r\t ":
                # Skip structural characters and whitespace
                i += 1
            else:
                # Unknown atom
                tokens.append(self.ATOM_VOCAB["UNK"])
                i += 1
        return tokens

    def tokenize_bonds(self, bond_list: list) -> List[int]:
        """Tokenize a list of bond type strings into bond token IDs.

        Args:
            bond_list: List of bond type strings (e.g., ["single", "double", "aromatic"]).

        Returns:
            List of integer bond token IDs.
        """
        return [self.BOND_VOCAB.get(bond, self.BOND_VOCAB["PAD"]) for bond in bond_list]

    def pad(self, tokens: List[int], max_len: int = 64) -> np.ndarray:
        """Pad or truncate a token list to max_len.

        Args:
            tokens: List of integer token IDs.
            max_len: Target sequence length (default 64).

        Returns:
            numpy array of shape (max_len,) with PAD=0 for missing positions.
        """
        result = np.zeros(max_len, dtype=np.int32)
        n = min(len(tokens), max_len)
        result[:n] = tokens[:n]
        return result

    def decode(self, tokens: List[int]) -> str:
        """Decode a list of atom token IDs back to a SMILES-like string.

        Args:
            tokens: List of integer token IDs.

        Returns:
            String representation of the molecule (skips PAD tokens).
        """
        parts = []
        for t in tokens:
            atom = self.ATOM_VOCAB_INV.get(t, "?")
            if atom == "PAD":
                continue
            if atom == "UNK":
                parts.append("?")
            else:
                parts.append(atom)
        return "".join(parts)
