"""Edge deployment optimization for IoT and mobile devices."""

__version__ = "1.0.0"

from .deployment import EdgeDeployment
from .optimization import ModelOptimizer, QuantizationConfig
from .runtime import EdgeRuntime, InferenceEngine
from .monitoring import EdgeMonitoring

__all__ = [
    "EdgeDeployment",
    "ModelOptimizer", 
    "QuantizationConfig",
    "EdgeRuntime",
    "InferenceEngine",
    "EdgeMonitoring",
]