"""Core olfactory transformer model components."""

from .tokenizer import MoleculeTokenizer
from .config import OlfactoryConfig

# Conditional import for torch-dependent model
try:
    from .model import OlfactoryTransformer
    _HAS_MODEL = True
except ImportError:
    OlfactoryTransformer = None
    _HAS_MODEL = False

__all__ = [
    "MoleculeTokenizer", 
    "OlfactoryConfig",
]

if _HAS_MODEL:
    __all__.append("OlfactoryTransformer")