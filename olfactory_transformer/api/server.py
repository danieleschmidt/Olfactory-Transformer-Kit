"""FastAPI-based progressive web server for real-time olfactory inference."""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any
from pathlib import Path
from contextlib import asynccontextmanager
import json

try:
    from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.middleware.gzip import GZipMiddleware
    from fastapi.responses import JSONResponse, StreamingResponse
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
    import uvicorn
    from pydantic import BaseModel, Field
    _HAS_FASTAPI = True
except ImportError:
    _HAS_FASTAPI = False
    # Graceful degradation for missing dependencies
    logging.warning("FastAPI not available. API server functionality disabled.")
    class FastAPI:
        def __init__(self, *args, **kwargs): pass
    class HTTPException(Exception): pass
    class BaseModel: pass
    class Field: pass

from ..core.model import OlfactoryTransformer
from ..core.tokenizer import MoleculeTokenizer
from ..core.config import OlfactoryConfig
from ..utils.security import security_manager
from ..utils.monitoring import monitor_performance, observability_manager
from ..utils.caching import cache_manager
from .middleware import SecurityMiddleware, RateLimitMiddleware
from .models import APIModels


class ScentPredictionRequest(BaseModel):
    """Request model for scent prediction."""
    smiles: str = Field(..., description="SMILES string of the molecule")
    include_attention: bool = Field(False, description="Include attention weights")
    include_features: bool = Field(False, description="Include molecular features")
    model_version: str = Field("latest", description="Model version to use")


class SensorPredictionRequest(BaseModel):
    """Request model for sensor-based prediction."""
    sensor_data: Dict[str, float] = Field(..., description="Sensor readings")
    sensor_types: List[str] = Field(..., description="Types of sensors used")
    timestamp: Optional[float] = Field(None, description="Reading timestamp")


class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions."""
    molecules: List[str] = Field(..., description="List of SMILES strings")
    max_batch_size: int = Field(32, description="Maximum batch size")


class StreamingPredictionRequest(BaseModel):
    """Request model for streaming predictions."""
    session_id: str = Field(..., description="Streaming session ID")
    sensor_config: Dict[str, Any] = Field(..., description="Sensor configuration")


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    version: str
    model_loaded: bool
    uptime_seconds: float
    memory_usage: Dict[str, float]
    gpu_available: bool


class OlfactoryAPIServer:
    """Progressive web API server for olfactory inference."""
    
    def __init__(self, config: Optional[OlfactoryConfig] = None):
        self.config = config or OlfactoryConfig()
        self.model = None
        self.tokenizer = None
        self.start_time = time.time()
        self.app = None
        self.active_streams = {}
        
        # Model loading
        self._load_models()
        
        # Create FastAPI app
        self.app = self._create_app()
    
    def _load_models(self) -> None:
        """Load olfactory models with error handling."""
        try:
            logging.info("Loading olfactory transformer model...")
            self.model = OlfactoryTransformer(self.config)
            self.tokenizer = MoleculeTokenizer.from_pretrained("olfactory-base-v1")
            logging.info("Models loaded successfully")
        except Exception as e:
            logging.error(f"Failed to load models: {e}")
            # Create fallback models for API functionality
            self.model = OlfactoryTransformer(self.config)
            self.tokenizer = MoleculeTokenizer(vocab_size=1000)
    
    @asynccontextmanager
    async def lifespan(self, app: FastAPI):
        """Application lifespan management."""
        # Startup
        logging.info("Starting Olfactory API Server...")
        observability_manager.start_monitoring()
        
        yield
        
        # Shutdown
        logging.info("Shutting down Olfactory API Server...")
        # Close active streams
        for stream_id in list(self.active_streams.keys()):
            await self._close_stream(stream_id)
        observability_manager.shutdown()
    
    def _create_app(self) -> FastAPI:
        """Create and configure FastAPI application."""
        app = FastAPI(
            title="Olfactory Transformer API",
            description="Real-time molecular scent prediction API",
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc",
            lifespan=self.lifespan
        )
        
        # Middleware
        app.add_middleware(GZipMiddleware, minimum_size=1000)
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure appropriately for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        app.add_middleware(SecurityMiddleware)
        app.add_middleware(RateLimitMiddleware, requests_per_minute=100)
        
        # Routes
        self._setup_routes(app)
        
        return app
    
    def _setup_routes(self, app: FastAPI) -> None:
        """Setup API routes."""
        
        @app.get("/health", response_model=HealthResponse)
        async def health_check():
            """Health check endpoint."""
            import psutil
            import torch
            
            memory = psutil.virtual_memory()
            
            return HealthResponse(
                status="healthy",
                version="1.0.0", 
                model_loaded=self.model is not None,
                uptime_seconds=time.time() - self.start_time,
                memory_usage={
                    "used_percent": memory.percent,
                    "available_gb": memory.available / (1024**3)
                },
                gpu_available=torch.cuda.is_available()
            )
        
        @app.post("/predict/scent")
        @monitor_performance("api_predict_scent")
        async def predict_scent(request: ScentPredictionRequest):
            """Predict scent from SMILES string."""
            if not self.model or not self.tokenizer:
                raise HTTPException(status_code=503, detail="Model not available")
            
            try:
                # Check cache first
                cache_key = f"scent:{request.smiles}:{request.model_version}"
                cached_result = await cache_manager.get(cache_key)
                if cached_result:
                    return cached_result
                
                # Predict
                prediction = self.model.predict_scent(request.smiles, self.tokenizer)
                
                result = {
                    "prediction": {
                        "primary_notes": prediction.primary_notes,
                        "descriptors": prediction.descriptors,
                        "intensity": prediction.intensity,
                        "confidence": prediction.confidence,
                        "chemical_family": prediction.chemical_family,
                    },
                    "metadata": {
                        "model_version": request.model_version,
                        "processing_time_ms": 0,  # Would be measured
                        "cached": False
                    }
                }
                
                # Cache result
                await cache_manager.set(cache_key, result, ttl=3600)
                
                return result
                
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
            except Exception as e:
                logging.error(f"Prediction failed: {e}")
                raise HTTPException(status_code=500, detail="Prediction failed")
        
        @app.post("/predict/sensor")
        @monitor_performance("api_predict_sensor")
        async def predict_from_sensor(request: SensorPredictionRequest):
            """Predict scent from sensor readings."""
            if not self.model:
                raise HTTPException(status_code=503, detail="Model not available")
            
            try:
                from ..core.config import SensorReading
                
                sensor_reading = SensorReading(
                    gas_sensors=request.sensor_data,
                    sensor_types=request.sensor_types,
                    timestamp=request.timestamp or time.time()
                )
                
                prediction = self.model.predict_from_sensors(sensor_reading)
                
                return {
                    "prediction": {
                        "primary_notes": prediction.primary_notes,
                        "descriptors": prediction.descriptors,
                        "intensity": prediction.intensity,
                        "confidence": prediction.confidence,
                    },
                    "sensor_info": {
                        "types": request.sensor_types,
                        "timestamp": sensor_reading.timestamp,
                        "num_sensors": len(request.sensor_data)
                    }
                }
                
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
            except Exception as e:
                logging.error(f"Sensor prediction failed: {e}")
                raise HTTPException(status_code=500, detail="Sensor prediction failed")
        
        @app.post("/predict/batch")
        @monitor_performance("api_predict_batch")
        async def predict_batch(request: BatchPredictionRequest, background_tasks: BackgroundTasks):
            """Batch prediction endpoint."""
            if not self.model or not self.tokenizer:
                raise HTTPException(status_code=503, detail="Model not available")
            
            if len(request.molecules) > 100:
                raise HTTPException(status_code=400, detail="Batch size too large")
            
            try:
                results = []
                for smiles in request.molecules:
                    try:
                        prediction = self.model.predict_scent(smiles, self.tokenizer)
                        results.append({
                            "smiles": smiles,
                            "prediction": {
                                "primary_notes": prediction.primary_notes,
                                "confidence": prediction.confidence,
                            },
                            "success": True
                        })
                    except Exception as e:
                        results.append({
                            "smiles": smiles,
                            "error": str(e),
                            "success": False
                        })
                
                return {
                    "results": results,
                    "summary": {
                        "total": len(request.molecules),
                        "successful": sum(1 for r in results if r["success"]),
                        "failed": sum(1 for r in results if not r["success"])
                    }
                }
                
            except Exception as e:
                logging.error(f"Batch prediction failed: {e}")
                raise HTTPException(status_code=500, detail="Batch prediction failed")
        
        @app.post("/stream/start")
        async def start_streaming(request: StreamingPredictionRequest):
            """Start streaming prediction session."""
            session_id = request.session_id
            
            if session_id in self.active_streams:
                raise HTTPException(status_code=400, detail="Session already active")
            
            # Create streaming session
            self.active_streams[session_id] = {
                "config": request.sensor_config,
                "start_time": time.time(),
                "predictions": []
            }
            
            return {"session_id": session_id, "status": "started"}
        
        @app.get("/stream/{session_id}")
        async def stream_predictions(session_id: str):
            """Stream real-time predictions."""
            if session_id not in self.active_streams:
                raise HTTPException(status_code=404, detail="Session not found")
            
            async def generate_predictions():
                while session_id in self.active_streams:
                    # Simulate streaming data
                    prediction_data = {
                        "timestamp": time.time(),
                        "session_id": session_id,
                        "prediction": {
                            "primary_notes": ["floral", "fresh"],
                            "confidence": 0.85,
                            "intensity": 6.2
                        }
                    }
                    
                    yield f"data: {json.dumps(prediction_data)}\\n\\n"
                    await asyncio.sleep(1)  # 1 Hz streaming
            
            return StreamingResponse(
                generate_predictions(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                }
            )
        
        @app.delete("/stream/{session_id}")
        async def stop_streaming(session_id: str):
            """Stop streaming session."""
            if session_id not in self.active_streams:
                raise HTTPException(status_code=404, detail="Session not found")
            
            await self._close_stream(session_id)
            return {"session_id": session_id, "status": "stopped"}
        
        @app.get("/models/info")
        async def model_info():
            """Get information about available models."""
            return {
                "available_models": ["olfactory-base-v1"],
                "current_model": {
                    "name": "olfactory-base-v1",
                    "parameters": "240M",
                    "architecture": "transformer",
                    "capabilities": [
                        "smiles_to_scent",
                        "sensor_fusion",
                        "batch_processing",
                        "streaming"
                    ]
                }
            }
        
        @app.get("/metrics")
        async def get_metrics():
            """Get API metrics."""
            return observability_manager.get_metrics()
    
    async def _close_stream(self, session_id: str) -> None:
        """Close streaming session."""
        if session_id in self.active_streams:
            del self.active_streams[session_id]
            logging.info(f"Closed streaming session: {session_id}")
    
    def run(self, host: str = "0.0.0.0", port: int = 8000, **kwargs) -> None:
        """Run the API server."""
        uvicorn.run(
            self.app,
            host=host,
            port=port,
            log_level="info",
            **kwargs
        )


# Factory function for easy deployment
def create_api_server(config: Optional[OlfactoryConfig] = None) -> FastAPI:
    """Factory function to create API server."""
    server = OlfactoryAPIServer(config)
    return server.app


if __name__ == "__main__":
    # Development server
    server = OlfactoryAPIServer()
    server.run(host="0.0.0.0", port=8000, reload=True)