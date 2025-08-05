"""
Olfactory-Transformer-Kit: Foundation model for computational olfaction.

The first open-source foundation model for computational olfaction,
enabling smell-sense AI through molecular structure to scent description mapping.
"""

__version__ = "0.1.0"

from .core.model import OlfactoryTransformer
from .core.tokenizer import MoleculeTokenizer
from .sensors.enose import ENoseInterface
from .design.inverse import ScentDesigner
from .evaluation.metrics import PerceptualEvaluator
from .training.trainer import OlfactoryTrainer

__all__ = [
    "OlfactoryTransformer",
    "MoleculeTokenizer", 
    "ENoseInterface",
    "ScentDesigner",
    "PerceptualEvaluator",
    "OlfactoryTrainer",
]