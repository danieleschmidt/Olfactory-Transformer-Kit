"""Utility modules for optimization and scaling."""

# Import utility modules with conditional imports for torch dependencies
try:
    from .distributed import DistributedTraining, FederatedOlfactory
    _HAS_DISTRIBUTED = True
except ImportError:
    DistributedTraining = None
    FederatedOlfactory = None
    _HAS_DISTRIBUTED = False

try:
    from .monitoring import PerformanceMonitor, ResourceTracker
    _HAS_MONITORING = True
except ImportError:
    PerformanceMonitor = None
    ResourceTracker = None
    _HAS_MONITORING = False

__all__ = []

if _HAS_DISTRIBUTED:
    __all__.extend(["DistributedTraining", "FederatedOlfactory"])

if _HAS_MONITORING:
    __all__.extend(["PerformanceMonitor", "ResourceTracker"])

# Lazy loading functions to avoid circular imports
def get_model_cache():
    """Lazy import ModelCache."""
    from .caching import ModelCache
    return ModelCache

def get_prediction_cache():  
    """Lazy import PredictionCache."""
    from .caching import PredictionCache
    return PredictionCache

def get_model_optimizer():
    """Lazy import ModelOptimizer.""" 
    from .optimization import ModelOptimizer
    return ModelOptimizer

def get_inference_accelerator():
    """Lazy import InferenceAccelerator."""
    from .optimization import InferenceAccelerator
    return InferenceAccelerator