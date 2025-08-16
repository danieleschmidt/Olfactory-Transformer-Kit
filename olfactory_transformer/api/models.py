"""API data models and schemas."""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, validator


class ModelStatus(str, Enum):
    """Model status enumeration."""
    LOADING = "loading"
    READY = "ready"
    ERROR = "error"
    UPDATING = "updating"


class PredictionMode(str, Enum):
    """Prediction mode enumeration."""
    FAST = "fast"
    ACCURATE = "accurate"
    BALANCED = "balanced"


class ScentCategory(str, Enum):
    """Scent category enumeration."""
    FLORAL = "floral"
    CITRUS = "citrus"
    WOODY = "woody"
    FRESH = "fresh"
    SPICY = "spicy"
    FRUITY = "fruity"
    HERBAL = "herbal"
    MARINE = "marine"
    MUSKY = "musky"
    AMBER = "amber"


class MoleculeInput(BaseModel):
    """Input model for molecule data."""
    smiles: str = Field(..., description="SMILES string representation")
    name: Optional[str] = Field(None, description="Molecule name")
    cas_number: Optional[str] = Field(None, description="CAS registry number")
    
    @validator('smiles')
    def validate_smiles(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError("SMILES string cannot be empty")
        if len(v) > 500:
            raise ValueError("SMILES string too long")
        return v.strip()


class SensorData(BaseModel):
    """Sensor data input model."""
    readings: Dict[str, float] = Field(..., description="Sensor readings")
    timestamp: Optional[datetime] = Field(None, description="Reading timestamp")
    sensor_types: List[str] = Field(..., description="Types of sensors")
    calibration_applied: bool = Field(False, description="Whether calibration was applied")
    
    @validator('readings')
    def validate_readings(cls, v):
        if not v:
            raise ValueError("Sensor readings cannot be empty")
        
        for sensor, value in v.items():
            if not isinstance(value, (int, float)):
                raise ValueError(f"Invalid reading for sensor {sensor}")
            if not (-1000 <= value <= 1000):
                raise ValueError(f"Reading out of range for sensor {sensor}")
        
        return v


class PredictionRequest(BaseModel):
    """Base prediction request model."""
    molecule: Optional[MoleculeInput] = None
    sensor_data: Optional[SensorData] = None
    mode: PredictionMode = Field(PredictionMode.BALANCED, description="Prediction mode")
    include_features: bool = Field(False, description="Include molecular features")
    include_attention: bool = Field(False, description="Include attention weights")
    model_version: str = Field("latest", description="Model version to use")


class ScentPrediction(BaseModel):
    """Scent prediction result model."""
    primary_notes: List[str] = Field(..., description="Primary scent notes")
    descriptors: List[str] = Field(..., description="Detailed descriptors")
    intensity: float = Field(..., ge=0, le=10, description="Scent intensity (0-10)")
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence")
    chemical_family: Optional[str] = Field(None, description="Chemical family")
    categories: List[ScentCategory] = Field(default=[], description="Scent categories")


class MolecularFeatures(BaseModel):
    """Molecular features model."""
    molecular_weight: Optional[float] = None
    logp: Optional[float] = None
    tpsa: Optional[float] = None
    num_atoms: Optional[int] = None
    num_bonds: Optional[int] = None
    num_rings: Optional[int] = None
    aromatic_rings: Optional[int] = None
    rotatable_bonds: Optional[int] = None


class AttentionWeights(BaseModel):
    """Attention weights model."""
    layer_weights: List[List[float]] = Field(..., description="Layer-wise attention weights")
    head_weights: List[List[float]] = Field(..., description="Head-wise attention weights")
    token_weights: List[float] = Field(..., description="Token-wise attention weights")


class PredictionResponse(BaseModel):
    """Complete prediction response model."""
    prediction: ScentPrediction
    molecular_features: Optional[MolecularFeatures] = None
    attention_weights: Optional[AttentionWeights] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    model_version: str = Field(..., description="Model version used")
    cached: bool = Field(False, description="Whether result was cached")


class BatchPredictionRequest(BaseModel):
    """Batch prediction request model."""
    molecules: List[MoleculeInput] = Field(..., max_items=100, description="List of molecules")
    mode: PredictionMode = Field(PredictionMode.FAST, description="Prediction mode for batch")
    include_features: bool = Field(False, description="Include molecular features")
    parallel_processing: bool = Field(True, description="Enable parallel processing")
    
    @validator('molecules')
    def validate_batch_size(cls, v):
        if len(v) == 0:
            raise ValueError("Batch cannot be empty")
        if len(v) > 100:
            raise ValueError("Batch size cannot exceed 100")
        return v


class BatchPredictionResult(BaseModel):
    """Individual batch prediction result."""
    molecule: MoleculeInput
    prediction: Optional[ScentPrediction] = None
    error: Optional[str] = None
    success: bool
    processing_time_ms: float


class BatchPredictionResponse(BaseModel):
    """Batch prediction response model."""
    results: List[BatchPredictionResult]
    summary: Dict[str, Any]
    total_processing_time_ms: float
    batch_id: str


class StreamingSession(BaseModel):
    """Streaming session configuration."""
    session_id: str = Field(..., description="Unique session identifier")
    sensor_config: Dict[str, Any] = Field(..., description="Sensor configuration")
    sampling_rate: float = Field(1.0, ge=0.1, le=10.0, description="Sampling rate in Hz")
    buffer_size: int = Field(100, ge=10, le=1000, description="Buffer size for streaming")
    auto_prediction: bool = Field(True, description="Enable automatic predictions")


class StreamingPrediction(BaseModel):
    """Streaming prediction result."""
    session_id: str
    timestamp: datetime
    prediction: ScentPrediction
    sensor_data: SensorData
    sequence_number: int


class ModelInfo(BaseModel):
    """Model information model."""
    name: str
    version: str
    status: ModelStatus
    parameters: str
    architecture: str
    capabilities: List[str]
    training_data: Dict[str, Any]
    performance_metrics: Dict[str, float]
    last_updated: datetime


class APIHealth(BaseModel):
    """API health status model."""
    status: str
    version: str
    uptime_seconds: float
    models: List[ModelInfo]
    system_resources: Dict[str, float]
    active_sessions: int
    total_requests: int
    errors_last_hour: int


class APIMetrics(BaseModel):
    """API metrics model."""
    requests_per_second: float
    average_response_time_ms: float
    error_rate: float
    cache_hit_rate: float
    active_connections: int
    model_inference_time_ms: float
    queue_size: int


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str = Field(..., description="Error message")
    error_code: str = Field(..., description="Error code")
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    request_id: Optional[str] = None


class APIModels:
    """Container class for all API models."""
    
    # Input models
    MoleculeInput = MoleculeInput
    SensorData = SensorData
    PredictionRequest = PredictionRequest
    BatchPredictionRequest = BatchPredictionRequest
    StreamingSession = StreamingSession
    
    # Output models  
    ScentPrediction = ScentPrediction
    PredictionResponse = PredictionResponse
    BatchPredictionResponse = BatchPredictionResponse
    StreamingPrediction = StreamingPrediction
    
    # Feature models
    MolecularFeatures = MolecularFeatures
    AttentionWeights = AttentionWeights
    
    # System models
    ModelInfo = ModelInfo
    APIHealth = APIHealth
    APIMetrics = APIMetrics
    ErrorResponse = ErrorResponse
    
    # Enums
    ModelStatus = ModelStatus
    PredictionMode = PredictionMode
    ScentCategory = ScentCategory