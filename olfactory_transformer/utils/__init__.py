"""Utility modules for optimization and scaling."""

# Import utility modules - avoiding circular imports by lazy loading
# from .caching import ModelCache, PredictionCache  # Loaded on-demand
# from .optimization import ModelOptimizer, InferenceAccelerator  # Loaded on-demand  
from .distributed import DistributedTraining, FederatedOlfactory
from .monitoring import PerformanceMonitor, ResourceTracker

__all__ = [
    "DistributedTraining", 
    "FederatedOlfactory",
    "PerformanceMonitor",
    "ResourceTracker",
]

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