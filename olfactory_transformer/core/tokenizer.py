"""Molecule tokenizer for SMILES and molecular representations."""

from typing import List, Dict, Optional, Union, Tuple
import re
import json
from pathlib import Path
from collections import Counter
import logging

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, AllChem
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False
    logging.warning("RDKit not available. Some molecular features will be disabled.")

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    # Create dummy numpy for type annotations
    class np:
        ndarray = object

from .config import OlfactoryConfig


class MoleculeTokenizer:
    """Tokenizer for molecular SMILES strings and molecular features."""
    
    def __init__(
        self,
        vocab_size: int = 50000,
        max_length: int = 512,
        special_tokens: Optional[Dict[str, str]] = None,
    ):
        self.vocab_size = vocab_size
        self.max_length = max_length
        
        # Special tokens
        default_special_tokens = {
            "pad_token": "[PAD]",
            "unk_token": "[UNK]", 
            "cls_token": "[CLS]",
            "sep_token": "[SEP]",
            "mask_token": "[MASK]",
        }
        self.special_tokens = special_tokens or default_special_tokens
        
        # Initialize vocabulary
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self._build_initial_vocab()
        
        # SMILES regex patterns
        self.smiles_pattern = re.compile(
            r'(\[[^\]]+\]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])'
        )
    
    def _build_initial_vocab(self) -> None:
        """Build initial vocabulary with special tokens."""
        for i, token in enumerate(self.special_tokens.values()):
            self.token_to_id[token] = i
            self.id_to_token[i] = token
    
    def build_vocab_from_smiles(self, smiles_list: List[str]) -> None:
        """Build vocabulary with input validation."""
        if not isinstance(smiles_list, list):
            raise TypeError("smiles_list must be a list")
        
        if len(smiles_list) > 1000000:  # Reasonable limit
            raise ValueError("Too many SMILES strings (max 1M)")
        
        # Filter and validate SMILES
        valid_smiles = []
        for smiles in smiles_list:
            if isinstance(smiles, str) and smiles.strip() and len(smiles) <= 1000:
                valid_smiles.append(smiles)
        
        if not valid_smiles:
            logging.warning("No valid SMILES strings provided - using minimal vocabulary")
            # Keep only special tokens for minimal vocabulary
            return
        
        logging.info(f"Building vocabulary from {len(valid_smiles)}/{len(smiles_list)} valid SMILES")
        smiles_list = valid_smiles
        """Build vocabulary from list of SMILES strings."""
        # Tokenize all SMILES
        all_tokens = []
        for smiles in smiles_list:
            tokens = self.tokenize_smiles(smiles)
            all_tokens.extend(tokens)
        
        # Count token frequencies
        token_counts = Counter(all_tokens)
        
        # Add most frequent tokens to vocabulary
        current_id = len(self.special_tokens)
        for token, count in token_counts.most_common(self.vocab_size - len(self.special_tokens)):
            if token not in self.token_to_id:
                self.token_to_id[token] = current_id
                self.id_to_token[current_id] = token
                current_id += 1
    
    def tokenize_smiles(self, smiles: str) -> List[str]:
        """Tokenize SMILES string into list of tokens."""
        if not smiles:
            return []
        
        # Clean SMILES
        smiles = smiles.strip()
        
        # Use regex to split SMILES
        tokens = self.smiles_pattern.findall(smiles)
        
        return tokens
    
    def encode(
        self, 
        smiles: str,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        padding: bool = True,
        truncation: bool = True,
    ) -> Dict[str, List[int]]:
        """Encode SMILES with security validation."""
        # Input validation and sanitization
        if not isinstance(smiles, str):
            raise TypeError("SMILES must be a string")
        
        if len(smiles) > 10000:  # Reasonable limit
            raise ValueError("SMILES string too long (max 10000 characters)")
        
        # Security: Check for suspicious patterns
        suspicious_patterns = ['exec', 'eval', 'import', '__', 'subprocess', 'os.system']
        if any(pattern in smiles.lower() for pattern in suspicious_patterns):
            raise ValueError("SMILES contains suspicious patterns")
        
        max_length = max_length or self.max_length
        if max_length > 10000:  # Security limit
            raise ValueError("max_length too large (max 10000)")
        """Encode SMILES string to token IDs."""
        max_length = max_length or self.max_length
        
        # Tokenize
        tokens = self.tokenize_smiles(smiles)
        
        # Add special tokens
        if add_special_tokens:
            tokens = [self.special_tokens["cls_token"]] + tokens + [self.special_tokens["sep_token"]]
        
        # Truncate if necessary
        if truncation and len(tokens) > max_length:
            tokens = tokens[:max_length-1] + [self.special_tokens["sep_token"]]
        
        # Convert to IDs
        input_ids = []
        for token in tokens:
            if token in self.token_to_id:
                input_ids.append(self.token_to_id[token])
            else:
                input_ids.append(self.token_to_id[self.special_tokens["unk_token"]])
        
        # Create attention mask
        attention_mask = [1] * len(input_ids)
        
        # Pad if necessary
        if padding and len(input_ids) < max_length:
            pad_length = max_length - len(input_ids)
            pad_id = self.token_to_id[self.special_tokens["pad_token"]]
            input_ids.extend([pad_id] * pad_length)
            attention_mask.extend([0] * pad_length)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs back to SMILES string."""
        tokens = []
        for token_id in token_ids:
            if token_id in self.id_to_token:
                token = self.id_to_token[token_id]
                if skip_special_tokens and token in self.special_tokens.values():
                    continue
                tokens.append(token)
        
        return "".join(tokens)
    
    def extract_molecular_features(self, smiles: str) -> Dict[str, float]:
        """Extract molecular descriptors with validation."""
        # Input validation
        if not isinstance(smiles, str) or not smiles.strip():
            logging.warning("Invalid SMILES input for feature extraction")
            return {}
        
        if len(smiles) > 1000:  # Reasonable limit for feature extraction
            logging.warning("SMILES too long for feature extraction")
            return {}
        """Extract molecular descriptors from SMILES."""
        if not HAS_RDKIT:
            return {}
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return {}
            
            features = {
                "molecular_weight": Descriptors.MolWt(mol),
                "logp": Descriptors.MolLogP(mol),
                "num_atoms": mol.GetNumAtoms(),
                "num_bonds": mol.GetNumBonds(),
                "num_rings": Descriptors.RingCount(mol),
                "num_aromatic_rings": Descriptors.NumAromaticRings(mol),
                "tpsa": Descriptors.TPSA(mol),
                "num_hbd": Descriptors.NumHDonors(mol),
                "num_hba": Descriptors.NumHAcceptors(mol),
                "num_rotatable_bonds": Descriptors.NumRotatableBonds(mol),
            }
            
            return features
            
        except Exception as e:
            logging.warning(f"Failed to extract features from SMILES {smiles}: {e}")
            return {}
    
    def get_fingerprint(self, smiles: str, radius: int = 2, n_bits: int = 2048) -> Optional[np.ndarray]:
        """Generate molecular fingerprint."""
        if not HAS_RDKIT:
            return None
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            if HAS_NUMPY:
                return np.array(fp)
            else:
                return list(fp)
            
        except Exception as e:
            logging.warning(f"Failed to generate fingerprint for {smiles}: {e}")
            return None
    
    def save_pretrained(self, save_directory: Union[str, Path]) -> None:
        """Save tokenizer with path validation."""
        if not save_directory:
            raise ValueError("Save directory cannot be empty")
        
        save_directory = Path(save_directory)
        
        # Security: Validate save path
        try:
            save_directory = save_directory.resolve()
        except Exception as e:
            raise ValueError(f"Invalid save directory: {e}")
        
        # Prevent path traversal
        if '..' in str(save_directory) or not str(save_directory).startswith('/root/'):
            logging.warning(f"Potentially unsafe save path: {save_directory}")
        """Save tokenizer to directory."""
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        
        # Save vocabulary
        vocab_file = save_directory / "vocab.json"
        with open(vocab_file, 'w') as f:
            json.dump(self.token_to_id, f, indent=2)
        
        # Save config
        config = {
            "vocab_size": self.vocab_size,
            "max_length": self.max_length,
            "special_tokens": self.special_tokens,
        }
        config_file = save_directory / "tokenizer_config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
    
    @classmethod
    def from_pretrained(cls, model_path: Union[str, Path]) -> "MoleculeTokenizer":
        """Load tokenizer with security validation."""
        if not model_path:
            raise ValueError("Model path cannot be empty")
        
        model_path = Path(model_path)
        
        # Security validation
        try:
            model_path = model_path.resolve()
        except Exception as e:
            raise ValueError(f"Invalid model path: {e}")
        
        # Prevent path traversal
        if '..' in str(model_path):
            raise ValueError("Path traversal detected in model path")
        """Load tokenizer from directory."""
        model_path = Path(model_path)
        
        # Load config
        config_file = model_path / "tokenizer_config.json"
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        # Create tokenizer
        tokenizer = cls(**config)
        
        # Load vocabulary
        vocab_file = model_path / "vocab.json"
        with open(vocab_file, 'r') as f:
            tokenizer.token_to_id = json.load(f)
        
        # Build reverse mapping
        tokenizer.id_to_token = {v: k for k, v in tokenizer.token_to_id.items()}
        
        return tokenizer
    
    @property
    def vocab_size_actual(self) -> int:
        """Get actual vocabulary size."""
        return len(self.token_to_id)
    
    def __len__(self) -> int:
        return len(self.token_to_id)