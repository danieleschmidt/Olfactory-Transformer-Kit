"""
Robust Processing Utilities for Generation 2.

Implements comprehensive error handling, validation, and fault-tolerant processing:
- Input validation and sanitization
- Graceful degradation under failures
- Memory-efficient processing
- Multi-modal data handling
- Production-ready error recovery
"""

import logging
import time
import traceback
from typing import Dict, List, Optional, Any, Union, Callable, Tuple
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
import json
import hashlib
from collections import defaultdict, deque
import threading
import asyncio

# Safe imports with fallbacks
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    # Mock numpy for basic operations
    class MockNumpy:
        @staticmethod 
        def array(x):
            return list(x) if isinstance(x, (list, tuple)) else [x]
        
        @staticmethod
        def mean(x):
            if isinstance(x, (list, tuple)) and x:
                return sum(x) / len(x)
            return 0.0
        
        @staticmethod
        def std(x):
            if isinstance(x, (list, tuple)) and len(x) > 1:
                mean = sum(x) / len(x)
                variance = sum((i - mean) ** 2 for i in x) / len(x)
                return variance ** 0.5
            return 0.0
        
        @staticmethod
        def clip(x, a_min, a_max):
            if isinstance(x, (list, tuple)):
                return [max(a_min, min(a_max, i)) for i in x]
            return max(a_min, min(a_max, x))
    
    np = MockNumpy()


@dataclass
class ValidationResult:
    """Result of input validation."""
    valid: bool
    errors: List[str]
    warnings: List[str]
    sanitized_data: Any
    metadata: Dict[str, Any]


@dataclass
class ProcessingResult:
    """Result of robust processing operation."""
    success: bool
    result: Any
    errors: List[str]
    warnings: List[str]
    processing_time: float
    fallback_used: bool
    metadata: Dict[str, Any]


class InputValidator:
    """Comprehensive input validation and sanitization."""
    
    def __init__(self):
        self.validation_stats = defaultdict(int)
        self.suspicious_patterns = [
            r'<script[^>]*>.*?</script>',  # XSS
            r'javascript:',                # JS injection
            r'union\s+select',            # SQL injection
            r'drop\s+table',              # SQL injection
            r'\.\./.*etc/passwd',         # Path traversal
            r'(rm\s+-rf|del\s+/)',       # Command injection
        ]
    
    def validate_smiles(self, smiles: str) -> ValidationResult:
        """Validate and sanitize SMILES string."""
        errors = []
        warnings = []
        
        if not isinstance(smiles, str):
            errors.append("SMILES must be a string")
            return ValidationResult(False, errors, warnings, None, {})
        
        # Basic length validation
        if len(smiles) == 0:
            errors.append("SMILES cannot be empty")
        elif len(smiles) > 500:
            errors.append("SMILES too long (max 500 characters)")
            warnings.append("Very large molecule - may cause performance issues")
        
        # Character validation
        valid_chars = set('CNOPSFClBrIHc()[]=#+-0123456789@/')
        invalid_chars = set(smiles) - valid_chars
        if invalid_chars:
            warnings.append(f"Unusual characters detected: {invalid_chars}")
        
        # Sanitization
        sanitized = smiles.strip()
        
        # Remove null bytes and control characters
        sanitized = ''.join(char for char in sanitized if ord(char) >= 32 or char in '\t\n\r')
        
        # Check for suspicious patterns
        for pattern in self.suspicious_patterns:
            import re
            if re.search(pattern, sanitized.lower()):
                errors.append(f"Suspicious pattern detected: {pattern}")
        
        # Basic structure validation
        if sanitized:
            paren_count = sanitized.count('(') - sanitized.count(')')
            bracket_count = sanitized.count('[') - sanitized.count(']')
            
            if paren_count != 0:
                errors.append("Unbalanced parentheses in SMILES")
            if bracket_count != 0:
                errors.append("Unbalanced brackets in SMILES")
        
        # Update stats
        self.validation_stats['smiles_validated'] += 1
        if errors:
            self.validation_stats['smiles_errors'] += 1
        
        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            sanitized_data=sanitized if len(errors) == 0 else None,
            metadata={'original_length': len(smiles), 'sanitized_length': len(sanitized)}
        )
    
    def validate_sensor_data(self, sensor_data: Dict[str, float]) -> ValidationResult:
        """Validate sensor reading data."""
        errors = []
        warnings = []
        
        if not isinstance(sensor_data, dict):
            errors.append("Sensor data must be a dictionary")
            return ValidationResult(False, errors, warnings, None, {})
        
        if not sensor_data:
            errors.append("Sensor data cannot be empty")
        
        sanitized = {}
        
        for sensor_name, value in sensor_data.items():
            # Validate sensor name
            if not isinstance(sensor_name, str):
                errors.append(f"Sensor name must be string: {sensor_name}")
                continue
            
            if len(sensor_name) > 50:
                warnings.append(f"Long sensor name: {sensor_name}")
            
            # Validate sensor value
            try:
                float_value = float(value)
                
                # Range validation
                if not (-10000 <= float_value <= 10000):
                    warnings.append(f"Sensor {sensor_name} value out of typical range: {float_value}")
                
                # NaN/infinity check
                if not (float_value == float_value):  # NaN check
                    errors.append(f"NaN value for sensor {sensor_name}")
                    continue
                
                if abs(float_value) == float('inf'):
                    errors.append(f"Infinite value for sensor {sensor_name}")
                    continue
                
                sanitized[sensor_name] = float_value
                
            except (ValueError, TypeError):
                errors.append(f"Invalid sensor value for {sensor_name}: {value}")
        
        self.validation_stats['sensor_data_validated'] += 1
        if errors:
            self.validation_stats['sensor_data_errors'] += 1
        
        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            sanitized_data=sanitized if len(errors) == 0 else None,
            metadata={'sensor_count': len(sanitized)}
        )
    
    def get_validation_stats(self) -> Dict[str, int]:
        """Get validation statistics."""
        return dict(self.validation_stats)


class FaultTolerantProcessor:
    """Fault-tolerant processing with graceful degradation."""
    
    def __init__(self, max_retries: int = 3, timeout_seconds: float = 30.0):
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds
        self.failure_counts = defaultdict(int)
        self.circuit_breakers = {}
        self.fallback_cache = {}
        self._lock = threading.Lock()
    
    def circuit_breaker(self, operation_name: str, failure_threshold: int = 5, 
                       reset_timeout: float = 300.0):
        """Circuit breaker decorator for operations."""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                return self._execute_with_circuit_breaker(
                    operation_name, func, failure_threshold, reset_timeout, *args, **kwargs
                )
            return wrapper
        return decorator
    
    def _execute_with_circuit_breaker(self, operation_name: str, func: Callable,
                                    failure_threshold: int, reset_timeout: float,
                                    *args, **kwargs) -> Any:
        """Execute function with circuit breaker pattern."""
        with self._lock:
            breaker_state = self.circuit_breakers.get(operation_name, {
                'state': 'closed',  # closed, open, half-open
                'failure_count': 0,
                'last_failure_time': 0,
                'failure_threshold': failure_threshold,
                'reset_timeout': reset_timeout
            })
            
            current_time = time.time()
            
            # Check if circuit should be reset
            if (breaker_state['state'] == 'open' and 
                current_time - breaker_state['last_failure_time'] > reset_timeout):
                breaker_state['state'] = 'half-open'
                breaker_state['failure_count'] = 0
                logging.info(f"Circuit breaker for {operation_name} moving to half-open")
            
            # If circuit is open, fail fast
            if breaker_state['state'] == 'open':
                raise RuntimeError(f"Circuit breaker {operation_name} is open")
            
            try:
                result = func(*args, **kwargs)
                
                # Success - reset failure count
                if breaker_state['state'] == 'half-open':
                    breaker_state['state'] = 'closed'
                    logging.info(f"Circuit breaker for {operation_name} closed")
                
                breaker_state['failure_count'] = 0
                self.circuit_breakers[operation_name] = breaker_state
                
                return result
                
            except Exception as e:
                breaker_state['failure_count'] += 1
                breaker_state['last_failure_time'] = current_time
                
                if breaker_state['failure_count'] >= failure_threshold:
                    breaker_state['state'] = 'open'
                    logging.warning(f"Circuit breaker for {operation_name} opened after {failure_threshold} failures")
                
                self.circuit_breakers[operation_name] = breaker_state
                raise e
    
    def retry_with_exponential_backoff(self, operation_name: str):
        """Retry decorator with exponential backoff."""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                return self._execute_with_retry(operation_name, func, *args, **kwargs)
            return wrapper
        return decorator
    
    def _execute_with_retry(self, operation_name: str, func: Callable, *args, **kwargs) -> Any:
        """Execute function with retry logic."""
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                if attempt > 0:
                    # Exponential backoff
                    delay = min(2 ** (attempt - 1), 30)  # Max 30 seconds
                    time.sleep(delay)
                    logging.info(f"Retrying {operation_name}, attempt {attempt + 1}")
                
                return func(*args, **kwargs)
                
            except Exception as e:
                last_exception = e
                logging.warning(f"Attempt {attempt + 1} for {operation_name} failed: {e}")
                
                # Don't retry for certain types of errors
                if isinstance(e, (ValueError, TypeError)):
                    logging.info(f"Not retrying {operation_name} due to {type(e).__name__}")
                    break
        
        # All retries exhausted
        logging.error(f"All {self.max_retries + 1} attempts for {operation_name} failed")
        raise last_exception
    
    def with_fallback(self, primary_func: Callable, fallback_func: Callable, 
                     cache_key: Optional[str] = None):
        """Execute primary function with fallback on failure."""
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                # Try primary function
                result = primary_func(*args, **kwargs)
                
                # Cache successful result if key provided
                if cache_key:
                    self.fallback_cache[cache_key] = result
                
                return ProcessingResult(
                    success=True,
                    result=result,
                    errors=[],
                    warnings=[],
                    processing_time=time.time() - start_time,
                    fallback_used=False,
                    metadata={'method': 'primary'}
                )
                
            except Exception as e:
                logging.warning(f"Primary function failed: {e}, trying fallback")
                
                try:
                    # Try fallback function
                    fallback_result = fallback_func(*args, **kwargs)
                    
                    return ProcessingResult(
                        success=True,
                        result=fallback_result,
                        errors=[],
                        warnings=[f"Primary method failed: {e}"],
                        processing_time=time.time() - start_time,
                        fallback_used=True,
                        metadata={'method': 'fallback', 'primary_error': str(e)}
                    )
                    
                except Exception as fallback_error:
                    # Both failed - check cache
                    if cache_key and cache_key in self.fallback_cache:
                        cached_result = self.fallback_cache[cache_key]
                        return ProcessingResult(
                            success=True,
                            result=cached_result,
                            errors=[],
                            warnings=[f"Using cached result - primary: {e}, fallback: {fallback_error}"],
                            processing_time=time.time() - start_time,
                            fallback_used=True,
                            metadata={'method': 'cache'}
                        )
                    
                    # Complete failure
                    return ProcessingResult(
                        success=False,
                        result=None,
                        errors=[f"Primary: {e}", f"Fallback: {fallback_error}"],
                        warnings=[],
                        processing_time=time.time() - start_time,
                        fallback_used=True,
                        metadata={'method': 'failed'}
                    )
        
        return wrapper


class MemoryEfficientProcessor:
    """Memory-efficient processing for large-scale operations."""
    
    def __init__(self, max_memory_mb: int = 1024, batch_size: int = 32):
        self.max_memory_mb = max_memory_mb
        self.batch_size = batch_size
        self.memory_usage = 0
        self.processed_count = 0
    
    def estimate_memory_usage(self, data_size: int, factor: float = 1.5) -> int:
        """Estimate memory usage in MB."""
        return int((data_size * factor) / (1024 * 1024))
    
    def process_in_batches(self, data: List[Any], process_func: Callable) -> List[Any]:
        """Process data in memory-efficient batches."""
        results = []
        total_items = len(data)
        
        for i in range(0, total_items, self.batch_size):
            batch = data[i:i + self.batch_size]
            batch_start_time = time.time()
            
            try:
                # Estimate memory usage for this batch
                estimated_memory = self.estimate_memory_usage(len(str(batch)))
                
                if estimated_memory > self.max_memory_mb:
                    # Further subdivide batch
                    sub_batch_size = max(1, self.batch_size // 2)
                    logging.warning(f"Large batch detected, subdividing to {sub_batch_size}")
                    
                    for j in range(0, len(batch), sub_batch_size):
                        sub_batch = batch[j:j + sub_batch_size]
                        sub_results = process_func(sub_batch)
                        results.extend(sub_results if isinstance(sub_results, list) else [sub_results])
                else:
                    batch_results = process_func(batch)
                    results.extend(batch_results if isinstance(batch_results, list) else [batch_results])
                
                self.processed_count += len(batch)
                batch_time = time.time() - batch_start_time
                
                logging.debug(f"Processed batch {i//self.batch_size + 1}/{(total_items-1)//self.batch_size + 1} "
                            f"({len(batch)} items) in {batch_time:.2f}s")
                
                # Yield control periodically
                if i % (self.batch_size * 10) == 0:
                    time.sleep(0.001)  # Prevent CPU starvation
                    
            except Exception as e:
                logging.error(f"Batch processing failed for batch {i//self.batch_size + 1}: {e}")
                # Continue with next batch rather than failing completely
                continue
        
        return results
    
    async def async_process_in_batches(self, data: List[Any], async_process_func: Callable) -> List[Any]:
        """Async version of batch processing."""
        results = []
        total_items = len(data)
        
        for i in range(0, total_items, self.batch_size):
            batch = data[i:i + self.batch_size]
            
            try:
                batch_results = await async_process_func(batch)
                results.extend(batch_results if isinstance(batch_results, list) else [batch_results])
                
                # Yield control
                await asyncio.sleep(0.001)
                
            except Exception as e:
                logging.error(f"Async batch processing failed: {e}")
                continue
        
        return results


class MultiModalDataHandler:
    """Robust multi-modal data handling and fusion."""
    
    def __init__(self):
        self.supported_modalities = ['molecular', 'sensor', 'textual', 'spectral', 'temporal']
        self.fusion_strategies = ['concatenation', 'attention', 'averaging', 'weighted']
        self.data_cache = {}
    
    def validate_modality_data(self, modality: str, data: Any) -> ValidationResult:
        """Validate data for specific modality."""
        errors = []
        warnings = []
        
        if modality not in self.supported_modalities:
            errors.append(f"Unsupported modality: {modality}")
            return ValidationResult(False, errors, warnings, None, {})
        
        sanitized_data = None
        metadata = {'modality': modality}
        
        try:
            if modality == 'molecular':
                if isinstance(data, str):
                    # SMILES string
                    validator = InputValidator()
                    result = validator.validate_smiles(data)
                    return result
                elif isinstance(data, (list, tuple)):
                    # Molecular features
                    if HAS_NUMPY:
                        sanitized_data = np.array(data, dtype=float)
                    else:
                        sanitized_data = [float(x) for x in data]
                    metadata['feature_count'] = len(data)
                else:
                    errors.append("Molecular data must be SMILES string or feature vector")
            
            elif modality == 'sensor':
                if isinstance(data, dict):
                    validator = InputValidator()
                    result = validator.validate_sensor_data(data)
                    return result
                elif isinstance(data, (list, tuple)):
                    # Sensor array
                    if HAS_NUMPY:
                        sanitized_data = np.array(data, dtype=float)
                    else:
                        sanitized_data = [float(x) for x in data]
                    metadata['sensor_count'] = len(data)
                else:
                    errors.append("Sensor data must be dict or array")
            
            elif modality == 'textual':
                if isinstance(data, str):
                    sanitized_data = data.strip()
                    metadata['text_length'] = len(sanitized_data)
                elif isinstance(data, list):
                    sanitized_data = [str(item).strip() for item in data]
                    metadata['text_count'] = len(sanitized_data)
                else:
                    errors.append("Textual data must be string or list of strings")
            
            elif modality == 'spectral':
                if isinstance(data, (list, tuple)):
                    if HAS_NUMPY:
                        sanitized_data = np.array(data, dtype=float)
                    else:
                        sanitized_data = [float(x) for x in data]
                    metadata['spectral_points'] = len(data)
                else:
                    errors.append("Spectral data must be array of values")
            
            elif modality == 'temporal':
                if isinstance(data, (list, tuple)) and all(isinstance(x, (list, tuple)) for x in data):
                    # Time series data
                    if HAS_NUMPY:
                        sanitized_data = [np.array(series, dtype=float) for series in data]
                    else:
                        sanitized_data = [[float(x) for x in series] for series in data]
                    metadata['sequence_count'] = len(data)
                    metadata['sequence_length'] = len(data[0]) if data else 0
                else:
                    errors.append("Temporal data must be list of sequences")
            
        except Exception as e:
            errors.append(f"Data processing error for {modality}: {e}")
        
        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            sanitized_data=sanitized_data,
            metadata=metadata
        )
    
    def fuse_modalities(self, modality_data: Dict[str, Any], 
                       strategy: str = 'concatenation') -> Tuple[Any, Dict[str, Any]]:
        """Fuse data from multiple modalities."""
        if strategy not in self.fusion_strategies:
            raise ValueError(f"Unsupported fusion strategy: {strategy}")
        
        validated_data = {}
        fusion_metadata = {'strategy': strategy, 'modalities': list(modality_data.keys())}
        
        # Validate all modalities
        for modality, data in modality_data.items():
            validation_result = self.validate_modality_data(modality, data)
            if not validation_result.valid:
                logging.warning(f"Modality {modality} validation failed: {validation_result.errors}")
                continue
            validated_data[modality] = validation_result.sanitized_data
        
        if not validated_data:
            raise ValueError("No valid modality data available for fusion")
        
        fusion_metadata['valid_modalities'] = list(validated_data.keys())
        
        # Apply fusion strategy
        try:
            if strategy == 'concatenation':
                fused_data = self._concatenate_modalities(validated_data)
            elif strategy == 'attention':
                fused_data = self._attention_fusion(validated_data)
            elif strategy == 'averaging':
                fused_data = self._average_fusion(validated_data)
            elif strategy == 'weighted':
                fused_data = self._weighted_fusion(validated_data)
            else:
                fused_data = self._concatenate_modalities(validated_data)  # Default fallback
            
            return fused_data, fusion_metadata
            
        except Exception as e:
            logging.error(f"Fusion failed: {e}")
            # Fallback to simple concatenation
            fused_data = self._concatenate_modalities(validated_data)
            fusion_metadata['fallback_used'] = True
            return fused_data, fusion_metadata
    
    def _concatenate_modalities(self, data: Dict[str, Any]) -> Any:
        """Simple concatenation fusion."""
        all_features = []
        
        for modality, modal_data in data.items():
            if isinstance(modal_data, (list, tuple)):
                all_features.extend(modal_data)
            elif HAS_NUMPY and hasattr(modal_data, 'flatten'):
                all_features.extend(modal_data.flatten().tolist())
            else:
                all_features.append(float(modal_data))
        
        return all_features
    
    def _attention_fusion(self, data: Dict[str, Any]) -> Any:
        """Attention-based fusion (simplified)."""
        # Simplified attention mechanism
        modality_weights = {}
        total_weight = 0
        
        for modality, modal_data in data.items():
            # Simple heuristic: longer sequences get higher weight
            if isinstance(modal_data, (list, tuple)):
                weight = len(modal_data)
            else:
                weight = 1
            modality_weights[modality] = weight
            total_weight += weight
        
        # Normalize weights
        for modality in modality_weights:
            modality_weights[modality] /= total_weight
        
        # Weighted concatenation
        weighted_features = []
        for modality, modal_data in data.items():
            weight = modality_weights[modality]
            
            if isinstance(modal_data, (list, tuple)):
                weighted_data = [x * weight for x in modal_data]
                weighted_features.extend(weighted_data)
            else:
                weighted_features.append(float(modal_data) * weight)
        
        return weighted_features
    
    def _average_fusion(self, data: Dict[str, Any]) -> Any:
        """Average-based fusion."""
        # Align all modalities to same length by padding/truncation
        max_length = 0
        aligned_data = {}
        
        for modality, modal_data in data.items():
            if isinstance(modal_data, (list, tuple)):
                max_length = max(max_length, len(modal_data))
                aligned_data[modality] = list(modal_data)
            else:
                aligned_data[modality] = [float(modal_data)]
                max_length = max(max_length, 1)
        
        # Pad or truncate to max_length
        for modality in aligned_data:
            current_len = len(aligned_data[modality])
            if current_len < max_length:
                # Pad with zeros
                aligned_data[modality].extend([0.0] * (max_length - current_len))
            elif current_len > max_length:
                # Truncate
                aligned_data[modality] = aligned_data[modality][:max_length]
        
        # Average across modalities
        averaged = []
        num_modalities = len(aligned_data)
        
        for i in range(max_length):
            values = [aligned_data[modality][i] for modality in aligned_data]
            avg_value = sum(values) / num_modalities
            averaged.append(avg_value)
        
        return averaged
    
    def _weighted_fusion(self, data: Dict[str, Any]) -> Any:
        """Weighted fusion with learned weights."""
        # Simple learned weights based on modality reliability
        modality_reliability = {
            'molecular': 0.4,
            'sensor': 0.3,
            'textual': 0.1,
            'spectral': 0.15,
            'temporal': 0.05
        }
        
        weighted_features = []
        
        for modality, modal_data in data.items():
            weight = modality_reliability.get(modality, 0.1)
            
            if isinstance(modal_data, (list, tuple)):
                weighted_data = [x * weight for x in modal_data]
                weighted_features.extend(weighted_data)
            else:
                weighted_features.append(float(modal_data) * weight)
        
        return weighted_features


class RobustPipeline:
    """Complete robust processing pipeline."""
    
    def __init__(self):
        self.validator = InputValidator()
        self.processor = FaultTolerantProcessor()
        self.memory_processor = MemoryEfficientProcessor()
        self.multimodal_handler = MultiModalDataHandler()
        self.processing_stats = defaultdict(int)
    
    def process_request(self, request_data: Dict[str, Any]) -> ProcessingResult:
        """Process a complete request with full robustness."""
        start_time = time.time()
        
        try:
            # Step 1: Input validation
            validation_results = {}
            
            for key, value in request_data.items():
                if key == 'smiles':
                    validation_results[key] = self.validator.validate_smiles(value)
                elif key == 'sensor_data':
                    validation_results[key] = self.validator.validate_sensor_data(value)
                else:
                    # Generic validation
                    validation_results[key] = ValidationResult(
                        valid=True, errors=[], warnings=[], 
                        sanitized_data=value, metadata={}
                    )
            
            # Check if any critical validations failed
            critical_errors = []
            for key, result in validation_results.items():
                if not result.valid:
                    critical_errors.extend([f"{key}: {error}" for error in result.errors])
            
            if critical_errors:
                return ProcessingResult(
                    success=False,
                    result=None,
                    errors=critical_errors,
                    warnings=[],
                    processing_time=time.time() - start_time,
                    fallback_used=False,
                    metadata={'stage': 'validation'}
                )
            
            # Step 2: Multi-modal fusion if needed
            if len(request_data) > 1:
                try:
                    fused_data, fusion_metadata = self.multimodal_handler.fuse_modalities(
                        request_data, strategy='concatenation'
                    )
                except Exception as e:
                    logging.warning(f"Multi-modal fusion failed: {e}, processing individually")
                    fused_data = request_data
                    fusion_metadata = {'fallback': True}
            else:
                fused_data = request_data
                fusion_metadata = {'single_modality': True}
            
            # Step 3: Robust processing with fallback
            def primary_processing():
                # Simulate advanced processing
                result = {
                    'processed_data': fused_data,
                    'confidence': 0.85,
                    'processing_method': 'primary'
                }
                return result
            
            def fallback_processing():
                # Simplified fallback processing
                result = {
                    'processed_data': fused_data,
                    'confidence': 0.60,
                    'processing_method': 'fallback'
                }
                return result
            
            processing_wrapper = self.processor.with_fallback(
                primary_processing, fallback_processing, 
                cache_key=hashlib.md5(str(request_data).encode()).hexdigest()
            )
            
            processing_result = processing_wrapper()
            
            # Update stats
            self.processing_stats['requests_processed'] += 1
            if processing_result.fallback_used:
                self.processing_stats['fallback_used'] += 1
            
            return ProcessingResult(
                success=processing_result.success,
                result=processing_result.result,
                errors=processing_result.errors,
                warnings=processing_result.warnings,
                processing_time=time.time() - start_time,
                fallback_used=processing_result.fallback_used,
                metadata={
                    'validation_results': {k: v.metadata for k, v in validation_results.items()},
                    'fusion_metadata': fusion_metadata,
                    **processing_result.metadata
                }
            )
            
        except Exception as e:
            logging.error(f"Pipeline processing failed: {e}")
            traceback.print_exc()
            
            return ProcessingResult(
                success=False,
                result=None,
                errors=[f"Pipeline error: {e}"],
                warnings=[],
                processing_time=time.time() - start_time,
                fallback_used=False,
                metadata={'stage': 'pipeline_error'}
            )
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get comprehensive processing statistics."""
        stats = dict(self.processing_stats)
        stats.update(self.validator.get_validation_stats())
        
        # Circuit breaker stats
        stats['circuit_breakers'] = {
            name: breaker['state'] for name, breaker in self.processor.circuit_breakers.items()
        }
        
        return stats


# Global robust pipeline instance
robust_pipeline = RobustPipeline()