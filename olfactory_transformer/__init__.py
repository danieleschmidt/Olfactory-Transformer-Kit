"""
Olfactory-Transformer-Kit: Foundation model for computational olfaction.

The first open-source foundation model for computational olfaction,
enabling smell-sense AI through molecular structure to scent description mapping.
"""

__version__ = "0.1.0"

# Conditional imports using dependency isolation
from .utils.dependency_isolation import dependency_manager

try:
    from .core.model import OlfactoryTransformer
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False
    # Create mock OlfactoryTransformer
    class OlfactoryTransformer:
        """Mock OlfactoryTransformer for graceful degradation."""
        def __init__(self, config=None):
            self.config = config
            
        def predict_scent(self, smiles: str):
            """Mock scent prediction."""
            from .core.config import ScentPrediction
            return ScentPrediction(
                primary_notes=['mock_floral', 'mock_fresh'],
                intensity=5.0,
                similar_perfumes=['Mock Fragrance'],
                confidence=0.5
            )
            
        @classmethod
        def from_pretrained(cls, model_name: str):
            """Mock pretrained model loading.""" 
            return cls()

from .core.tokenizer import MoleculeTokenizer
from .sensors.enose import ENoseInterface

# Optional advanced features
try:
    from .design.inverse import ScentDesigner
except ImportError:
    ScentDesigner = None

try:
    from .evaluation.metrics import PerceptualEvaluator
except ImportError:
    PerceptualEvaluator = None

try:
    from .training.trainer import OlfactoryTrainer
except ImportError:
    OlfactoryTrainer = None

# Export available classes
__all__ = [
    "MoleculeTokenizer", 
    "ENoseInterface",
]

if _HAS_TORCH:
    __all__.append("OlfactoryTransformer")

if ScentDesigner is not None:
    __all__.append("ScentDesigner")

if PerceptualEvaluator is not None:
    __all__.append("PerceptualEvaluator")

if OlfactoryTrainer is not None:
    __all__.append("OlfactoryTrainer")

# Enhanced features
try:
    from .core.enhanced_features import EnhancedOlfactorySystem
    __all__.append("EnhancedOlfactorySystem")
except ImportError:
    EnhancedOlfactorySystem = None

# Research modules
try:
    from .research.breakthrough_algorithms_2025 import BreakthroughResearchOrchestrator2025
    __all__.append("BreakthroughResearchOrchestrator2025")
except ImportError:
    BreakthroughResearchOrchestrator2025 = None

# Robust processing and dependency management (Generation 2)
try:
    from .utils.robust_processing import robust_pipeline, ValidationResult, ProcessingResult
    from .utils.dependency_manager import dependency_manager
    from .utils.i18n_manager import i18n_manager
    __all__.extend(["robust_pipeline", "dependency_manager", "i18n_manager", 
                   "ValidationResult", "ProcessingResult"])
except ImportError:
    robust_pipeline = None
    dependency_manager = None
    i18n_manager = None