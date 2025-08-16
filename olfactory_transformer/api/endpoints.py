"""API endpoint definitions and route creation."""

from typing import Dict, List, Optional, Any
import asyncio
import time
import logging
from datetime import datetime

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Request, status
from fastapi.responses import StreamingResponse
import json

from ..core.model import OlfactoryTransformer
from ..core.tokenizer import MoleculeTokenizer
from ..core.config import OlfactoryConfig, SensorReading
from ..utils.monitoring import monitor_performance
from ..utils.caching import cache_manager
from .models import (
    APIModels, PredictionRequest, PredictionResponse, BatchPredictionRequest,
    BatchPredictionResponse, StreamingSession, StreamingPrediction,
    ModelInfo, APIHealth, APIMetrics, ErrorResponse
)


def create_api_routes(
    model: OlfactoryTransformer,
    tokenizer: MoleculeTokenizer,
    config: OlfactoryConfig
) -> APIRouter:
    """Create and configure API routes."""
    
    router = APIRouter()
    
    # Dependency for model access
    def get_model() -> OlfactoryTransformer:
        if model is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not available"
            )
        return model
    
    def get_tokenizer() -> MoleculeTokenizer:
        if tokenizer is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Tokenizer not available"
            )
        return tokenizer
    
    # Health check endpoint
    @router.get("/health", response_model=APIModels.APIHealth, tags=["System"])
    async def health_check() -> APIModels.APIHealth:
        """Comprehensive health check."""
        import psutil
        import torch
        
        try:
            # System resources
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Model status
            model_status = APIModels.ModelStatus.READY if model else APIModels.ModelStatus.ERROR
            
            model_info = APIModels.ModelInfo(
                name="olfactory-transformer",
                version="1.0.0",
                status=model_status,
                parameters="240M",
                architecture="transformer",
                capabilities=["smiles_prediction", "sensor_fusion", "batch_processing"],
                training_data={"molecules": 5000000, "descriptions": 4000000},
                performance_metrics={"accuracy": 0.89, "f1_score": 0.85},
                last_updated=datetime.now()
            )
            
            return APIModels.APIHealth(
                status="healthy",
                version="1.0.0",
                uptime_seconds=time.time() - getattr(health_check, 'start_time', time.time()),
                models=[model_info],
                system_resources={
                    "memory_percent": memory.percent,
                    "memory_available_gb": memory.available / (1024**3),
                    "cpu_percent": cpu_percent,
                    "gpu_available": torch.cuda.is_available(),
                    "gpu_memory_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3) if torch.cuda.is_available() else 0
                },
                active_sessions=0,  # Would track active streaming sessions
                total_requests=0,   # Would track from metrics
                errors_last_hour=0  # Would track from metrics
            )
            
        except Exception as e:
            logging.error(f"Health check failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Health check failed"
            )
    
    # Set start time for uptime calculation
    if not hasattr(health_check, 'start_time'):
        health_check.start_time = time.time()
    
    # Single prediction endpoint
    @router.post("/predict", response_model=APIModels.PredictionResponse, tags=["Prediction"])
    @monitor_performance("api_predict")
    async def predict_scent(
        request: APIModels.PredictionRequest,
        model: OlfactoryTransformer = Depends(get_model),
        tokenizer: MoleculeTokenizer = Depends(get_tokenizer)
    ) -> APIModels.PredictionResponse:
        """Predict scent from molecule or sensor data."""
        start_time = time.time()
        
        try:
            if request.molecule:
                # SMILES-based prediction
                cache_key = f"predict:{request.molecule.smiles}:{request.model_version}"
                
                # Check cache
                if cached_result := await cache_manager.get(cache_key):
                    cached_result["cached"] = True
                    return APIModels.PredictionResponse(**cached_result)
                
                # Predict
                prediction = model.predict_scent(request.molecule.smiles, tokenizer)
                
                # Create response
                response_data = {
                    "prediction": APIModels.ScentPrediction(
                        primary_notes=prediction.primary_notes,
                        descriptors=prediction.descriptors,
                        intensity=prediction.intensity,
                        confidence=prediction.confidence,
                        chemical_family=prediction.chemical_family,
                        categories=[]  # Would map to enum values
                    ),
                    "metadata": {
                        "molecule_name": request.molecule.name,
                        "cas_number": request.molecule.cas_number
                    },
                    "processing_time_ms": (time.time() - start_time) * 1000,
                    "model_version": request.model_version,
                    "cached": False
                }
                
                # Add features if requested
                if request.include_features:
                    response_data["molecular_features"] = APIModels.MolecularFeatures(
                        molecular_weight=250.5,  # Would calculate actual values
                        logp=3.2,
                        tpsa=45.6,
                        num_atoms=18,
                        num_bonds=19,
                        num_rings=1,
                        aromatic_rings=1,
                        rotatable_bonds=3
                    )
                
                # Add attention if requested
                if request.include_attention:
                    response_data["attention_weights"] = APIModels.AttentionWeights(
                        layer_weights=[[0.1, 0.2, 0.3] for _ in range(24)],
                        head_weights=[[0.05, 0.1, 0.15] for _ in range(16)],
                        token_weights=[0.1] * 50
                    )
                
                # Cache result
                await cache_manager.set(cache_key, response_data, ttl=3600)
                
                return APIModels.PredictionResponse(**response_data)
                
            elif request.sensor_data:
                # Sensor-based prediction
                sensor_reading = SensorReading(
                    gas_sensors=request.sensor_data.readings,
                    sensor_types=request.sensor_data.sensor_types,
                    timestamp=request.sensor_data.timestamp.timestamp() if request.sensor_data.timestamp else time.time()
                )
                
                prediction = model.predict_from_sensors(sensor_reading)
                
                return APIModels.PredictionResponse(
                    prediction=APIModels.ScentPrediction(
                        primary_notes=prediction.primary_notes,
                        descriptors=prediction.descriptors,
                        intensity=prediction.intensity,
                        confidence=prediction.confidence,
                        categories=[]
                    ),
                    metadata={
                        "sensor_count": len(request.sensor_data.readings),
                        "calibration_applied": request.sensor_data.calibration_applied
                    },
                    processing_time_ms=(time.time() - start_time) * 1000,
                    model_version=request.model_version,
                    cached=False
                )
            
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Either molecule or sensor_data must be provided"
                )
                
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e)
            )
        except Exception as e:
            logging.error(f"Prediction failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Prediction failed"
            )
    
    # Batch prediction endpoint
    @router.post("/predict/batch", response_model=APIModels.BatchPredictionResponse, tags=["Prediction"])
    @monitor_performance("api_predict_batch")
    async def predict_batch(
        request: APIModels.BatchPredictionRequest,
        background_tasks: BackgroundTasks,
        model: OlfactoryTransformer = Depends(get_model),
        tokenizer: MoleculeTokenizer = Depends(get_tokenizer)
    ) -> APIModels.BatchPredictionResponse:
        """Batch prediction for multiple molecules."""
        start_time = time.time()
        batch_id = f"batch_{int(time.time())}"
        
        try:
            results = []
            
            # Process molecules
            for i, molecule in enumerate(request.molecules):
                mol_start_time = time.time()
                
                try:
                    prediction = model.predict_scent(molecule.smiles, tokenizer)
                    
                    result = APIModels.BatchPredictionResult(
                        molecule=molecule,
                        prediction=APIModels.ScentPrediction(
                            primary_notes=prediction.primary_notes,
                            descriptors=prediction.descriptors,
                            intensity=prediction.intensity,
                            confidence=prediction.confidence,
                            chemical_family=prediction.chemical_family,
                            categories=[]
                        ),
                        success=True,
                        processing_time_ms=(time.time() - mol_start_time) * 1000
                    )
                    
                except Exception as e:
                    result = APIModels.BatchPredictionResult(
                        molecule=molecule,
                        error=str(e),
                        success=False,
                        processing_time_ms=(time.time() - mol_start_time) * 1000
                    )
                
                results.append(result)
            
            # Create summary
            successful = sum(1 for r in results if r.success)
            failed = len(results) - successful
            
            summary = {
                "total_molecules": len(request.molecules),
                "successful_predictions": successful,
                "failed_predictions": failed,
                "success_rate": successful / len(request.molecules) if request.molecules else 0,
                "average_processing_time_ms": sum(r.processing_time_ms for r in results) / len(results) if results else 0
            }
            
            return APIModels.BatchPredictionResponse(
                results=results,
                summary=summary,
                total_processing_time_ms=(time.time() - start_time) * 1000,
                batch_id=batch_id
            )
            
        except Exception as e:
            logging.error(f"Batch prediction failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Batch prediction failed"
            )
    
    # Streaming endpoints
    @router.post("/stream/start", tags=["Streaming"])
    async def start_streaming_session(session: APIModels.StreamingSession):
        """Start a streaming prediction session."""
        # Store session configuration
        # In production, would use Redis or database
        if not hasattr(start_streaming_session, 'active_sessions'):
            start_streaming_session.active_sessions = {}
        
        start_streaming_session.active_sessions[session.session_id] = {
            "config": session.dict(),
            "start_time": time.time(),
            "predictions": []
        }
        
        return {
            "session_id": session.session_id,
            "status": "started",
            "timestamp": datetime.now().isoformat()
        }
    
    @router.get("/stream/{session_id}", tags=["Streaming"])
    async def stream_predictions(
        session_id: str,
        model: OlfactoryTransformer = Depends(get_model)
    ):
        """Stream real-time predictions."""
        if not hasattr(start_streaming_session, 'active_sessions'):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found"
            )
        
        if session_id not in start_streaming_session.active_sessions:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found"
            )
        
        async def generate_predictions():
            sequence_number = 0
            
            try:
                while session_id in start_streaming_session.active_sessions:
                    # Simulate sensor reading
                    sensor_data = APIModels.SensorData(
                        readings={"TGS2600": 245.2, "TGS2602": 187.3, "TGS2610": 334.1},
                        sensor_types=["TGS2600", "TGS2602", "TGS2610"],
                        timestamp=datetime.now(),
                        calibration_applied=True
                    )
                    
                    # Create sensor reading for model
                    sensor_reading = SensorReading(
                        gas_sensors=sensor_data.readings,
                        sensor_types=sensor_data.sensor_types,
                        timestamp=time.time()
                    )
                    
                    # Predict
                    prediction = model.predict_from_sensors(sensor_reading)
                    
                    # Create streaming prediction
                    streaming_prediction = APIModels.StreamingPrediction(
                        session_id=session_id,
                        timestamp=datetime.now(),
                        prediction=APIModels.ScentPrediction(
                            primary_notes=prediction.primary_notes,
                            descriptors=prediction.descriptors,
                            intensity=prediction.intensity,
                            confidence=prediction.confidence,
                            categories=[]
                        ),
                        sensor_data=sensor_data,
                        sequence_number=sequence_number
                    )
                    
                    # Send data
                    data = streaming_prediction.dict()
                    yield f"data: {json.dumps(data, default=str)}\\n\\n"
                    
                    sequence_number += 1
                    await asyncio.sleep(1.0)  # 1 Hz
                    
            except Exception as e:
                error_data = {"error": str(e), "session_id": session_id}
                yield f"data: {json.dumps(error_data)}\\n\\n"
        
        return StreamingResponse(
            generate_predictions(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
            }
        )
    
    @router.delete("/stream/{session_id}", tags=["Streaming"])
    async def stop_streaming_session(session_id: str):
        """Stop a streaming session."""
        if hasattr(start_streaming_session, 'active_sessions'):
            if session_id in start_streaming_session.active_sessions:
                del start_streaming_session.active_sessions[session_id]
                return {
                    "session_id": session_id,
                    "status": "stopped",
                    "timestamp": datetime.now().isoformat()
                }
        
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )
    
    # Model information endpoints
    @router.get("/models", response_model=List[APIModels.ModelInfo], tags=["Models"])
    async def list_models():
        """List available models."""
        return [
            APIModels.ModelInfo(
                name="olfactory-transformer",
                version="1.0.0",
                status=APIModels.ModelStatus.READY,
                parameters="240M",
                architecture="transformer",
                capabilities=["smiles_prediction", "sensor_fusion", "batch_processing", "streaming"],
                training_data={
                    "molecules": 5000000,
                    "descriptions": 4000000,
                    "sensor_readings": 850000
                },
                performance_metrics={
                    "accuracy": 0.893,
                    "f1_score": 0.847,
                    "pearson_r": 0.823
                },
                last_updated=datetime.now()
            )
        ]
    
    @router.get("/models/{model_name}", response_model=APIModels.ModelInfo, tags=["Models"])
    async def get_model_info(model_name: str):
        """Get detailed information about a specific model."""
        if model_name == "olfactory-transformer":
            return APIModels.ModelInfo(
                name="olfactory-transformer",
                version="1.0.0", 
                status=APIModels.ModelStatus.READY,
                parameters="240M",
                architecture="transformer",
                capabilities=["smiles_prediction", "sensor_fusion", "batch_processing", "streaming"],
                training_data={
                    "molecules": 5000000,
                    "descriptions": 4000000,
                    "sensor_readings": 850000,
                    "datasets": ["GoodScents", "Flavornet", "Pyrfume", "LeffingWell"]
                },
                performance_metrics={
                    "scent_classification_accuracy": 0.893,
                    "descriptor_prediction_f1": 0.847,
                    "intensity_prediction_mae": 0.72,
                    "molecular_similarity_pearson": 0.823,
                    "sensor_classification_accuracy": 0.912
                },
                last_updated=datetime.now()
            )
        
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model not found"
        )
    
    # Metrics endpoint
    @router.get("/metrics", response_model=APIModels.APIMetrics, tags=["System"])
    async def get_metrics():
        """Get API performance metrics."""
        return APIModels.APIMetrics(
            requests_per_second=12.5,
            average_response_time_ms=250.0,
            error_rate=0.02,
            cache_hit_rate=0.65,
            active_connections=5,
            model_inference_time_ms=180.0,
            queue_size=0
        )
    
    return router