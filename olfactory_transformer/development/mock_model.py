"""Mock model for development and testing without dependencies."""

import logging
import random
import time
from typing import Dict, List, Optional, Any
from ..core.config import ScentPrediction


class MockOlfactoryTransformer:
    """Mock transformer for development without PyTorch."""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.model_loaded = True
        logging.info("🔧 Mock Olfactory Transformer initialized for development")
    
    def predict_scent(self, smiles: str, **kwargs) -> ScentPrediction:
        """Mock scent prediction with realistic synthetic data."""
        time.sleep(0.1)  # Simulate processing time
        
        # Generate synthetic but realistic predictions
        primary_notes = self._generate_mock_notes(smiles)
        intensity = random.uniform(4.0, 9.0)
        confidence = random.uniform(0.75, 0.95)
        
        return ScentPrediction(
            primary_notes=primary_notes,
            intensity=intensity,
            confidence=confidence,
            similar_perfumes=self._generate_similar_perfumes(),
            chemical_family=self._infer_chemical_family(smiles),
            safety_warnings=self._check_safety(smiles)
        )
    
    def _generate_mock_notes(self, smiles: str) -> List[str]:
        """Generate realistic scent notes based on SMILES structure."""
        # Simple heuristics based on molecular patterns
        notes_pools = {
            'aromatic': ['floral', 'rose', 'jasmine', 'woody', 'spicy'],
            'aliphatic': ['fresh', 'citrus', 'green', 'marine'],
            'ester': ['fruity', 'sweet', 'tropical'],
            'alcohol': ['fresh', 'clean', 'powdery'],
            'default': ['complex', 'unique', 'sophisticated']
        }
        
        # Simple pattern matching
        if 'C=C' in smiles or 'c1' in smiles:
            pool = notes_pools['aromatic']
        elif 'COC' in smiles or 'C(=O)O' in smiles:
            pool = notes_pools['ester']
        elif 'CO' in smiles:
            pool = notes_pools['alcohol']
        elif any(c.isdigit() for c in smiles):
            pool = notes_pools['aliphatic']
        else:
            pool = notes_pools['default']
        
        return random.sample(pool, min(3, len(pool)))
    
    def _generate_similar_perfumes(self) -> List[str]:
        """Generate mock similar perfumes."""
        perfumes = [
            "Chanel No. 5", "Dior Sauvage", "Tom Ford Black Orchid",
            "Creed Aventus", "Maison Margiela Replica", "Le Labo Santal 33"
        ]
        return random.sample(perfumes, random.randint(1, 3))
    
    def _infer_chemical_family(self, smiles: str) -> str:
        """Mock chemical family inference."""
        families = {
            'C=C': 'alkene',
            'c1': 'aromatic',
            'COC': 'ether',
            'C(=O)O': 'carboxylic_acid',
            'CO': 'alcohol'
        }
        
        for pattern, family in families.items():
            if pattern in smiles:
                return family
        return 'organic_compound'
    
    def _check_safety(self, smiles: str) -> List[str]:
        """Mock safety analysis."""
        warnings = []
        
        # Simple safety heuristics
        if 'Cl' in smiles or 'Br' in smiles:
            warnings.append("Contains halogen - use with caution")
        if len(smiles) > 50:
            warnings.append("Complex molecule - verify safety data")
        if 'N' in smiles and 'O' in smiles:
            warnings.append("Contains N-O groups - check allergen potential")
        
        return warnings
    
    def analyze_molecule(self, smiles: str, **kwargs) -> Dict[str, Any]:
        """Mock detailed molecular analysis."""
        prediction = self.predict_scent(smiles)
        
        return {
            'smiles': smiles,
            'molecular_weight': random.uniform(100, 400),
            'logp': random.uniform(-1, 5),
            'scent_prediction': prediction,
            'synthetic_accessibility': random.uniform(1, 10),
            'drug_likeness': random.uniform(0, 1),
            'analysis_timestamp': time.time()
        }
    
    def predict_from_sensors(self, sensor_data: Dict[str, float]) -> ScentPrediction:
        """Mock sensor-based prediction."""
        time.sleep(0.05)  # Faster for sensor data
        
        # Synthesize prediction based on sensor values
        avg_signal = sum(sensor_data.values()) / len(sensor_data)
        
        if avg_signal > 500:
            notes = ['strong', 'pungent', 'chemical']
        elif avg_signal > 200:
            notes = ['moderate', 'floral', 'organic']
        else:
            notes = ['subtle', 'fresh', 'clean']
        
        return ScentPrediction(
            primary_notes=notes,
            intensity=min(10, avg_signal / 50),
            confidence=random.uniform(0.6, 0.9),
            similar_perfumes=["Sensor-based match"],
            detection_method="electronic_nose"
        )


class MockTokenizer:
    """Mock tokenizer for development."""
    
    def __init__(self):
        self.vocab_size = 1000
        logging.info("🔧 Mock tokenizer initialized")
    
    def tokenize(self, smiles: str) -> List[int]:
        """Mock tokenization."""
        # Simple character-based tokenization
        return [hash(c) % self.vocab_size for c in smiles[:50]]
    
    def decode(self, tokens: List[int]) -> str:
        """Mock decoding."""
        return f"mock_molecule_{hash(tuple(tokens)) % 10000}"
    
    def encode_batch(self, smiles_list: List[str]) -> List[List[int]]:
        """Mock batch encoding."""
        return [self.tokenize(smiles) for smiles in smiles_list]


# Convenience factory function
def create_mock_model(config=None):
    """Create a mock model for development."""
    return MockOlfactoryTransformer(config)