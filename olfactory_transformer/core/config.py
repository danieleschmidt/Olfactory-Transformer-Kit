"""Configuration classes for the Olfactory Transformer."""

from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from pathlib import Path
import json


@dataclass
class OlfactoryConfig:
    """Configuration class for OlfactoryTransformer model."""
    
    # Model architecture
    vocab_size: int = 50000
    hidden_size: int = 1024
    num_attention_heads: int = 16
    num_hidden_layers: int = 24
    intermediate_size: int = 4096
    max_position_embeddings: int = 2048
    
    # Molecular encoder
    gnn_layers: int = 5
    molecular_features: int = 512
    conformer_attention: bool = True
    
    # Olfactory decoder
    num_scent_classes: int = 1000
    intensity_levels: int = 10
    similarity_dim: int = 256
    
    def __post_init__(self):
        """Post-initialization processing."""
        # Ensure scent classes doesn't exceed available descriptors
        if hasattr(self, '_scent_descriptors_count'):
            self.num_scent_classes = min(self.num_scent_classes, self._scent_descriptors_count)
    
    # Sensor fusion
    sensor_channels: int = 64
    temporal_window: int = 100
    
    # Training
    dropout: float = 0.1
    layer_norm_epsilon: float = 1e-12
    initializer_range: float = 0.02
    
    # Inference
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.9
    
    # Paths
    model_path: Optional[Path] = None
    tokenizer_path: Optional[Path] = None
    
    # Device
    device: str = "auto"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            k: str(v) if isinstance(v, Path) else v
            for k, v in self.__dict__.items()
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "OlfactoryConfig":
        """Create config from dictionary."""
        return cls(**config_dict)
    
    @classmethod
    def from_json(cls, json_path: Path) -> "OlfactoryConfig":
        """Load config from JSON file."""
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    def save_json(self, json_path: Path) -> None:
        """Save config to JSON file."""
        with open(json_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


@dataclass
class ScentPrediction:
    """Container for scent prediction results."""
    
    primary_notes: List[str] = field(default_factory=list)
    descriptors: List[str] = field(default_factory=list)
    intensity: float = 0.0
    confidence: float = 0.0
    similar_perfumes: List[str] = field(default_factory=list)
    chemical_family: str = "unknown"
    ifra_category: str = "unclassified"
    safety_warnings: List[str] = field(default_factory=list)
    detection_method: str = "molecular_analysis"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert prediction to dictionary."""
        return {
            "primary_notes": self.primary_notes,
            "descriptors": self.descriptors,
            "intensity": self.intensity,
            "confidence": self.confidence,
            "similar_perfumes": self.similar_perfumes,
            "chemical_family": self.chemical_family,
            "ifra_category": self.ifra_category,
        }


@dataclass
class SensorReading:
    """Container for sensor array readings with robust validation."""
    
    gas_sensors: Dict[str, float]
    timestamp: Optional[float] = None
    environmental: Dict[str, float] = field(default_factory=dict)
    spectral: Optional[List[float]] = None
    temperature: float = 20.0
    humidity: float = 50.0
    
    def __post_init__(self):
        """Post-initialization validation and defaults."""
        if self.timestamp is None:
            import time
            self.timestamp = time.time()
        
        # Validate gas sensors data
        if not self.gas_sensors:
            raise ValueError("Gas sensors data cannot be empty")
        
        # Ensure all sensor values are valid numbers
        for sensor_name, value in self.gas_sensors.items():
            if not isinstance(value, (int, float)) or not (-1000 <= value <= 10000):
                raise ValueError(f"Invalid sensor value for {sensor_name}: {value}")
        
        # Validate environmental data if provided
        if self.environmental:
            for env_name, value in self.environmental.items():
                if not isinstance(value, (int, float)) or not (-100 <= value <= 1000):
                    raise ValueError(f"Invalid environmental value for {env_name}: {value}")
        
        # Validate temperature and humidity ranges
        if not (-50 <= self.temperature <= 100):
            raise ValueError(f"Temperature out of range: {self.temperature}")
        
        if not (0 <= self.humidity <= 100):
            raise ValueError(f"Humidity out of range: {self.humidity}")
    
    @property
    def sensor_types(self) -> List[str]:
        """Get list of available sensor types."""
        types = list(self.gas_sensors.keys())
        if self.environmental:
            types.extend(self.environmental.keys())
        return types
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert reading to dictionary."""
        return {
            "timestamp": self.timestamp,
            "gas_sensors": self.gas_sensors,
            "environmental": self.environmental,
            "spectral": self.spectral,
            "temperature": self.temperature,
            "humidity": self.humidity,
            "sensor_types": self.sensor_types
        }