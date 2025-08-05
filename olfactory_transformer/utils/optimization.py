"""Model optimization and inference acceleration utilities."""

from typing import Dict, Any, Optional, List, Union, Tuple
import logging
import time
from pathlib import Path
import json
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
import queue

import torch
import torch.nn as nn
import numpy as np

# Optional optimization imports
try:
    import torch.jit
    HAS_TORCHSCRIPT = True
except ImportError:
    HAS_TORCHSCRIPT = False

try:
    import onnx
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

try:
    import tensorrt as trt
    HAS_TENSORRT = True
except ImportError:
    HAS_TENSORRT = False

from ..core.model import OlfactoryTransformer
from ..core.config import OlfactoryConfig, ScentPrediction


class ModelOptimizer:
    """Model optimization utilities for inference acceleration."""
    
    def __init__(self, model: OlfactoryTransformer):
        self.model = model
        self.optimized_models = {}
        
    def quantize_model(
        self,
        method: str = "dynamic",
        calibration_data: Optional[torch.Tensor] = None,
        dtype: torch.dtype = torch.qint8
    ) -> nn.Module:
        """Quantize model for reduced memory and faster inference."""
        logging.info(f"Quantizing model using {method} quantization")
        
        if method == "dynamic":
            # Dynamic quantization
            quantized_model = torch.quantization.quantize_dynamic(
                self.model,
                {nn.Linear, nn.LSTM, nn.GRU},  # Layers to quantize
                dtype=dtype
            )
            
        elif method == "static":
            # Static quantization (requires calibration data)
            if calibration_data is None:
                raise ValueError("Static quantization requires calibration data")
            
            # Prepare model for quantization
            self.model.eval()
            self.model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
            torch.quantization.prepare(self.model, inplace=True)
            
            # Calibrate with data
            with torch.no_grad():
                _ = self.model(calibration_data)
            
            # Convert to quantized model
            quantized_model = torch.quantization.convert(self.model, inplace=False)
            
        else:
            raise ValueError(f"Unknown quantization method: {method}")
        
        self.optimized_models["quantized"] = quantized_model
        
        # Benchmark performance
        self._benchmark_model(quantized_model, "quantized")
        
        return quantized_model
    
    def export_torchscript(
        self,
        save_path: Union[str, Path],
        example_input: Optional[torch.Tensor] = None,
        method: str = "trace"
    ) -> Optional[torch.jit.ScriptModule]:
        """Export model to TorchScript for optimized deployment."""
        if not HAS_TORCHSCRIPT:
            logging.warning("TorchScript not available")
            return None
        
        logging.info(f"Exporting model to TorchScript using {method}")
        
        self.model.eval()
        
        if method == "trace":
            if example_input is None:
                # Create dummy input
                example_input = {
                    "input_ids": torch.randint(0, 1000, (1, 50)),
                    "attention_mask": torch.ones(1, 50),
                }
            
            try:
                traced_model = torch.jit.trace(self.model, example_input)
                traced_model.save(save_path)
                
                self.optimized_models["torchscript"] = traced_model
                return traced_model
                
            except Exception as e:
                logging.error(f"Failed to trace model: {e}")
                return None
        
        elif method == "script":
            try:
                scripted_model = torch.jit.script(self.model)
                scripted_model.save(save_path)
                
                self.optimized_models["torchscript"] = scripted_model
                return scripted_model
                
            except Exception as e:
                logging.error(f"Failed to script model: {e}")
                return None
        
        else:
            raise ValueError(f"Unknown TorchScript method: {method}")
    
    def export_onnx(
        self,
        save_path: Union[str, Path],
        example_input: Optional[Dict[str, torch.Tensor]] = None,
        opset_version: int = 15,
        optimize_for_mobile: bool = False
    ) -> bool:
        """Export model to ONNX format."""
        if not HAS_ONNX:
            logging.warning("ONNX not available")
            return False
        
        logging.info("Exporting model to ONNX format")
        
        if example_input is None:
            example_input = {
                "input_ids": torch.randint(0, 1000, (1, 50)),
                "attention_mask": torch.ones(1, 50),
            }
        
        self.model.eval()
        
        try:
            # Extract input names and values
            input_names = list(example_input.keys())
            input_values = tuple(example_input.values())
            
            # Dynamic axes for variable sequence length
            dynamic_axes = {
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "attention_mask": {0: "batch_size", 1: "sequence_length"},
            }
            
            torch.onnx.export(
                self.model,
                input_values,
                save_path,
                input_names=input_names,
                output_names=["scent_logits", "intensity", "similarity_embedding"],
                dynamic_axes=dynamic_axes,
                opset_version=opset_version,
                do_constant_folding=True,
                export_params=True,
            )
            
            # Optimize for mobile if requested
            if optimize_for_mobile:
                self._optimize_onnx_for_mobile(save_path)
            
            logging.info(f"Model exported to {save_path}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to export ONNX model: {e}")
            return False
    
    def optimize_tensorrt(
        self,
        onnx_path: Union[str, Path],
        engine_path: Union[str, Path],
        precision: str = "fp16",
        max_batch_size: int = 32,
        workspace_size: int = 1 << 30  # 1GB
    ) -> bool:
        """Optimize model with TensorRT."""
        if not HAS_TENSORRT:
            logging.warning("TensorRT not available")
            return False
        
        logging.info(f"Optimizing model with TensorRT ({precision} precision)")
        
        try:
            # Create TensorRT logger and builder
            TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
            builder = trt.Builder(TRT_LOGGER)
            
            # Create network
            network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            
            # Parse ONNX model
            parser = trt.OnnxParser(network, TRT_LOGGER)
            with open(onnx_path, 'rb') as model_file:
                if not parser.parse(model_file.read()):
                    logging.error("Failed to parse ONNX model")
                    return False
            
            # Configure builder
            config = builder.create_builder_config()
            config.max_workspace_size = workspace_size
            
            if precision == "fp16":
                config.set_flag(trt.BuilderFlag.FP16)
            elif precision == "int8":
                config.set_flag(trt.BuilderFlag.INT8)
            
            # Build engine
            engine = builder.build_engine(network, config)
            if engine is None:
                logging.error("Failed to build TensorRT engine")
                return False
            
            # Save engine
            with open(engine_path, 'wb') as f:
                f.write(engine.serialize())
            
            logging.info(f"TensorRT engine saved to {engine_path}")
            return True
            
        except Exception as e:
            logging.error(f"TensorRT optimization failed: {e}")
            return False
    
    def _optimize_onnx_for_mobile(self, onnx_path: Union[str, Path]) -> None:
        """Apply mobile-specific ONNX optimizations."""
        try:
            import onnxoptimizer
            
            # Load model
            model = onnx.load(onnx_path)
            
            # Apply optimizations
            optimized_model = onnxoptimizer.optimize(model, [
                'eliminate_deadend',
                'eliminate_identity',
                'eliminate_nop_dropout',
                'eliminate_nop_transpose',
                'fuse_add_bias_into_conv',
                'fuse_consecutive_transposes',
                'fuse_matmul_add_bias_into_gemm',
                'fuse_transpose_into_gemm',
            ])
            
            # Save optimized model
            onnx.save(optimized_model, onnx_path)
            
        except ImportError:
            logging.warning("onnxoptimizer not available for mobile optimization")
        except Exception as e:
            logging.warning(f"Mobile optimization failed: {e}")
    
    def _benchmark_model(self, model: nn.Module, model_name: str) -> Dict[str, float]:
        """Benchmark model performance."""
        logging.info(f"Benchmarking {model_name} model")
        
        # Create test input
        test_input = {
            "input_ids": torch.randint(0, 1000, (32, 50)),  # Batch size 32
            "attention_mask": torch.ones(32, 50),
        }
        
        model.eval()
        times = []
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(**test_input)
        
        # Benchmark
        with torch.no_grad():
            for _ in range(100):
                start_time = time.time()
                _ = model(**test_input)
                end_time = time.time()
                times.append(end_time - start_time)
        
        # Calculate statistics
        avg_time = np.mean(times)
        std_time = np.std(times)
        throughput = 32 / avg_time  # samples per second
        
        results = {
            "avg_inference_time": avg_time,
            "std_inference_time": std_time,
            "throughput_samples_per_sec": throughput,
            "latency_per_sample_ms": (avg_time / 32) * 1000,
        }
        
        logging.info(f"{model_name} benchmark results: {results}")
        return results
    
    def compare_optimizations(self) -> Dict[str, Dict[str, float]]:
        """Compare performance of different optimizations."""
        results = {}
        
        # Benchmark original model
        results["original"] = self._benchmark_model(self.model, "original")
        
        # Benchmark optimized models
        for name, model in self.optimized_models.items():
            results[name] = self._benchmark_model(model, name)
        
        return results


class InferenceAccelerator:
    """Accelerate inference through batching, pooling, and async processing."""
    
    def __init__(
        self,
        model: Union[OlfactoryTransformer, nn.Module],
        max_batch_size: int = 64,
        max_workers: int = 4,
        queue_timeout: float = 1.0
    ):
        self.model = model
        self.max_batch_size = max_batch_size
        self.max_workers = max_workers
        self.queue_timeout = queue_timeout
        
        # Request queue for batching
        self.request_queue = queue.Queue()
        self.response_map = {}
        self.response_lock = threading.Lock()
        
        # Worker pool
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.batch_processor = None
        self.running = False
        
        # Performance tracking
        self.stats = {
            "requests_processed": 0,
            "batches_processed": 0,
            "avg_batch_size": 0.0,
            "total_inference_time": 0.0,
        }
    
    def start_batch_processing(self) -> None:
        """Start background batch processing."""
        if self.running:
            return
        
        self.running = True
        self.batch_processor = threading.Thread(target=self._batch_processing_loop)
        self.batch_processor.daemon = True
        self.batch_processor.start()
        
        logging.info("Batch processing started")
    
    def stop_batch_processing(self) -> None:
        """Stop background batch processing."""
        self.running = False
        if self.batch_processor:
            self.batch_processor.join(timeout=5.0)
        
        logging.info("Batch processing stopped")
    
    def predict_async(
        self,
        request_id: str,
        input_data: Dict[str, torch.Tensor]
    ) -> threading.Event:
        """Submit async prediction request."""
        result_event = threading.Event()
        
        request = {
            "id": request_id,
            "data": input_data,
            "event": result_event,
            "timestamp": time.time(),
        }
        
        self.request_queue.put(request)
        return result_event
    
    def get_result(self, request_id: str) -> Optional[Any]:
        """Get prediction result."""
        with self.response_lock:
            return self.response_map.pop(request_id, None)
    
    def predict_batch_sync(
        self,
        batch_data: List[Dict[str, torch.Tensor]]
    ) -> List[Any]:
        """Synchronous batch prediction."""
        if not batch_data:
            return []
        
        # Prepare batch
        batched_input = self._prepare_batch(batch_data)
        
        # Run inference
        start_time = time.time()
        with torch.no_grad():
            batch_outputs = self.model(**batched_input)
        
        inference_time = time.time() - start_time
        
        # Process outputs
        results = self._process_batch_outputs(batch_outputs, len(batch_data))
        
        # Update stats
        self.stats["requests_processed"] += len(batch_data)
        self.stats["batches_processed"] += 1
        self.stats["total_inference_time"] += inference_time
        
        batch_size = len(batch_data)
        self.stats["avg_batch_size"] = (
            (self.stats["avg_batch_size"] * (self.stats["batches_processed"] - 1) + batch_size) /
            self.stats["batches_processed"]
        )
        
        return results
    
    def _batch_processing_loop(self) -> None:
        """Main batch processing loop."""
        while self.running:
            requests = []
            
            # Collect requests with timeout
            try:
                # Get first request (blocking)
                first_request = self.request_queue.get(timeout=self.queue_timeout)
                requests.append(first_request)
                
                # Collect additional requests (non-blocking)
                while len(requests) < self.max_batch_size:
                    try:
                        request = self.request_queue.get_nowait()
                        requests.append(request)
                    except queue.Empty:
                        break
                
                # Process batch
                if requests:
                    self._process_request_batch(requests)
                    
            except queue.Empty:
                # No requests available, continue
                continue
            except Exception as e:
                logging.error(f"Error in batch processing: {e}")
    
    def _process_request_batch(self, requests: List[Dict]) -> None:
        """Process a batch of requests."""
        try:
            # Prepare batch data
            batch_data = [req["data"] for req in requests]
            batched_input = self._prepare_batch(batch_data)
            
            # Run inference
            start_time = time.time()
            with torch.no_grad():
                batch_outputs = self.model(**batched_input)
            inference_time = time.time() - start_time
            
            # Process outputs
            results = self._process_batch_outputs(batch_outputs, len(requests))
            
            # Store results and notify
            with self.response_lock:
                for request, result in zip(requests, results):
                    self.response_map[request["id"]] = result
                    request["event"].set()
            
            # Update stats
            self.stats["requests_processed"] += len(requests)
            self.stats["batches_processed"] += 1
            self.stats["total_inference_time"] += inference_time
            
            batch_size = len(requests)
            self.stats["avg_batch_size"] = (
                (self.stats["avg_batch_size"] * (self.stats["batches_processed"] - 1) + batch_size) /
                self.stats["batches_processed"]
            )
            
        except Exception as e:
            logging.error(f"Error processing request batch: {e}")
            
            # Notify requests of error
            with self.response_lock:
                for request in requests:
                    self.response_map[request["id"]] = {"error": str(e)}
                    request["event"].set()
    
    def _prepare_batch(self, batch_data: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Prepare batch from individual inputs."""
        if not batch_data:
            return {}
        
        # Get all keys from first sample
        keys = batch_data[0].keys()
        batched = {}
        
        for key in keys:
            # Stack tensors for this key
            tensors = [sample[key] for sample in batch_data if key in sample]
            if tensors and isinstance(tensors[0], torch.Tensor):
                # Handle different tensor shapes by padding
                max_length = max(t.shape[-1] if t.dim() > 1 else t.shape[0] for t in tensors)
                
                padded_tensors = []
                for tensor in tensors:
                    if tensor.dim() == 1:
                        # Pad 1D tensor
                        pad_size = max_length - tensor.shape[0]
                        if pad_size > 0:
                            tensor = torch.cat([tensor, torch.zeros(pad_size, dtype=tensor.dtype)])
                    elif tensor.dim() == 2:
                        # Pad 2D tensor
                        pad_size = max_length - tensor.shape[1]
                        if pad_size > 0:
                            tensor = torch.cat([tensor, torch.zeros(tensor.shape[0], pad_size, dtype=tensor.dtype)], dim=1)
                    
                    padded_tensors.append(tensor)
                
                batched[key] = torch.stack(padded_tensors, dim=0)
        
        return batched
    
    def _process_batch_outputs(self, batch_outputs: Dict[str, torch.Tensor], batch_size: int) -> List[Any]:
        """Process batch outputs into individual results."""
        results = []
        
        for i in range(batch_size):
            result = {}
            
            for key, tensor in batch_outputs.items():
                if isinstance(tensor, torch.Tensor):
                    result[key] = tensor[i].cpu().numpy() if tensor.dim() > 0 else tensor.item()
                else:
                    result[key] = tensor
            
            results.append(result)
        
        return results
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = self.stats.copy()
        
        if stats["requests_processed"] > 0:
            stats["avg_inference_time_per_request"] = (
                stats["total_inference_time"] / stats["requests_processed"]
            )
        
        if stats["batches_processed"] > 0:
            stats["avg_inference_time_per_batch"] = (
                stats["total_inference_time"] / stats["batches_processed"]
            )
        
        stats["queue_size"] = self.request_queue.qsize()
        stats["pending_responses"] = len(self.response_map)
        
        return stats
    
    def __enter__(self):
        """Context manager entry."""
        self.start_batch_processing()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_batch_processing()
        self.executor.shutdown(wait=True)