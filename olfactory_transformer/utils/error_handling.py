"""Comprehensive error handling and recovery utilities."""

import logging
import traceback
import functools
import time
from typing import Any, Callable, Dict, List, Optional, Type, Union
from dataclasses import dataclass
from enum import Enum
import threading
from collections import defaultdict, deque

import torch


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for classification."""
    VALIDATION = "validation"
    COMPUTATION = "computation"
    MEMORY = "memory"
    NETWORK = "network"
    SECURITY = "security"
    SYSTEM = "system"
    USER_INPUT = "user_input"
    MODEL = "model"


@dataclass
class ErrorRecord:
    """Record of an error occurrence."""
    timestamp: float
    error_type: str
    error_message: str
    stack_trace: str
    severity: ErrorSeverity
    category: ErrorCategory
    context: Dict[str, Any]
    function_name: str
    recovery_attempted: bool = False
    recovery_successful: bool = False


class ErrorHandler:
    """Centralized error handling and recovery system."""
    
    def __init__(self, max_errors: int = 1000):
        self.max_errors = max_errors
        self.error_history = deque(maxlen=max_errors)
        self.error_counts = defaultdict(int)
        self.error_patterns = defaultdict(list)
        self.recovery_strategies = {}
        self.lock = threading.Lock()
        
        # Setup default recovery strategies
        self._setup_default_strategies()
        
        logging.info("Error handler initialized")
    
    def _setup_default_strategies(self):
        """Setup default error recovery strategies."""
        self.recovery_strategies.update({
            "CUDA out of memory": self._handle_cuda_memory_error,
            "ConnectionError": self._handle_connection_error,
            "TimeoutError": self._handle_timeout_error,
            "ValidationError": self._handle_validation_error,
            "FileNotFoundError": self._handle_file_not_found,
            "PermissionError": self._handle_permission_error,
        })
    
    def record_error(
        self,
        error: Exception,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        category: ErrorCategory = ErrorCategory.SYSTEM,
        context: Optional[Dict[str, Any]] = None,
        function_name: str = "unknown"
    ) -> ErrorRecord:
        """Record an error occurrence."""
        
        error_record = ErrorRecord(
            timestamp=time.time(),
            error_type=type(error).__name__,
            error_message=str(error),
            stack_trace=traceback.format_exc(),
            severity=severity,
            category=category,
            context=context or {},
            function_name=function_name
        )
        
        with self.lock:
            self.error_history.append(error_record)
            self.error_counts[error_record.error_type] += 1
            self.error_patterns[error_record.error_type].append(error_record.timestamp)
        
        # Log error
        log_level = {
            ErrorSeverity.LOW: logging.INFO,
            ErrorSeverity.MEDIUM: logging.WARNING,
            ErrorSeverity.HIGH: logging.ERROR,
            ErrorSeverity.CRITICAL: logging.CRITICAL,
        }[severity]
        
        logging.log(
            log_level,
            f"[{category.value}] {error_record.error_type}: {error_record.error_message}"
        )
        
        return error_record
    
    def attempt_recovery(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Attempt to recover from an error."""
        error_type = type(error).__name__
        error_message = str(error)
        
        # Find matching recovery strategy
        strategy = None
        for pattern, recovery_func in self.recovery_strategies.items():
            if pattern in error_message or pattern == error_type:
                strategy = recovery_func
                break
        
        if strategy is None:
            logging.warning(f"No recovery strategy for error: {error_type}")
            return False
        
        try:
            logging.info(f"Attempting recovery for {error_type}")
            return strategy(error, context)
        except Exception as recovery_error:
            logging.error(f"Recovery failed: {recovery_error}")
            return False
    
    def _handle_cuda_memory_error(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Handle CUDA out of memory errors."""
        try:
            if torch.cuda.is_available():
                # Clear GPU cache
                torch.cuda.empty_cache()
                
                # Reduce batch size if available in context
                if "batch_size" in context:
                    context["batch_size"] = max(1, context["batch_size"] // 2)
                    logging.info(f"Reduced batch size to {context['batch_size']}")
                
                # Force garbage collection
                import gc
                gc.collect()
                
                logging.info("CUDA memory cleared, cache emptied")
                return True
        except Exception as e:
            logging.error(f"Failed to clear CUDA memory: {e}")
        
        return False
    
    def _handle_connection_error(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Handle network connection errors."""
        try:
            # Implement exponential backoff
            retry_count = context.get("retry_count", 0)
            if retry_count < 3:
                wait_time = (2 ** retry_count) * 1.0  # 1, 2, 4 seconds
                time.sleep(wait_time)
                context["retry_count"] = retry_count + 1
                logging.info(f"Retrying connection after {wait_time}s (attempt {retry_count + 1})")
                return True
        except Exception as e:
            logging.error(f"Connection recovery failed: {e}")
        
        return False
    
    def _handle_timeout_error(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Handle timeout errors."""
        try:
            # Increase timeout if possible
            if "timeout" in context:
                context["timeout"] = min(context["timeout"] * 1.5, 300)  # Max 5 minutes
                logging.info(f"Increased timeout to {context['timeout']}s")
                return True
        except Exception as e:
            logging.error(f"Timeout recovery failed: {e}")
        
        return False
    
    def _handle_validation_error(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Handle validation errors."""
        try:
            # Use fallback/default values if available
            if "fallback_value" in context:
                logging.info("Using fallback value for validation error")
                return True
            
            # Sanitize input if possible
            if "input_data" in context and isinstance(context["input_data"], str):
                # Basic sanitization
                sanitized = context["input_data"].strip()[:1000]  # Limit length
                context["input_data"] = sanitized
                logging.info("Input data sanitized")
                return True
        except Exception as e:
            logging.error(f"Validation recovery failed: {e}")
        
        return False
    
    def _handle_file_not_found(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Handle file not found errors."""
        try:
            # Try alternative file paths
            if "file_path" in context and "alternative_paths" in context:
                for alt_path in context["alternative_paths"]:
                    if Path(alt_path).exists():
                        context["file_path"] = alt_path
                        logging.info(f"Using alternative file path: {alt_path}")
                        return True
            
            # Create file with default content if path is provided
            if "create_default" in context and context["create_default"]:
                file_path = context.get("file_path")
                if file_path:
                    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
                    with open(file_path, 'w') as f:
                        f.write(context.get("default_content", ""))
                    logging.info(f"Created default file: {file_path}")
                    return True
        except Exception as e:
            logging.error(f"File recovery failed: {e}")
        
        return False
    
    def _handle_permission_error(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Handle permission errors."""
        try:
            # Try alternative directory with write permissions
            if "output_dir" in context:
                import tempfile
                temp_dir = tempfile.mkdtemp()
                context["output_dir"] = temp_dir
                logging.info(f"Using temporary directory: {temp_dir}")
                return True
        except Exception as e:
            logging.error(f"Permission recovery failed: {e}")
        
        return False
    
    def get_error_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get error summary for recent period."""
        cutoff_time = time.time() - (hours * 3600)
        
        with self.lock:
            recent_errors = [
                err for err in self.error_history 
                if err.timestamp >= cutoff_time
            ]
        
        # Count by type and severity
        error_types = defaultdict(int)
        severity_counts = defaultdict(int)
        category_counts = defaultdict(int)
        
        for error in recent_errors:
            error_types[error.error_type] += 1
            severity_counts[error.severity.value] += 1
            category_counts[error.category.value] += 1
        
        return {
            "period_hours": hours,
            "total_errors": len(recent_errors),
            "by_type": dict(error_types),
            "by_severity": dict(severity_counts),
            "by_category": dict(category_counts),
            "most_common_error": max(error_types.items(), key=lambda x: x[1])[0] if error_types else None,
        }
    
    def detect_error_patterns(self, window_minutes: int = 60) -> List[Dict[str, Any]]:
        """Detect error patterns and anomalies."""
        patterns = []
        window_seconds = window_minutes * 60
        current_time = time.time()
        
        with self.lock:
            for error_type, timestamps in self.error_patterns.items():
                # Count recent occurrences
                recent_timestamps = [
                    t for t in timestamps 
                    if current_time - t <= window_seconds
                ]
                
                if len(recent_timestamps) >= 5:  # Pattern threshold
                    # Calculate frequency
                    if len(recent_timestamps) > 1:
                        intervals = [
                            recent_timestamps[i] - recent_timestamps[i-1]
                            for i in range(1, len(recent_timestamps))
                        ]
                        avg_interval = sum(intervals) / len(intervals)
                        
                        patterns.append({
                            "error_type": error_type,
                            "count": len(recent_timestamps),
                            "avg_interval_seconds": avg_interval,
                            "pattern_detected": True,
                        })
        
        return patterns


def robust_operation(
    retry_count: int = 3,
    retry_delay: float = 1.0,
    fallback_value: Any = None,
    error_category: ErrorCategory = ErrorCategory.SYSTEM,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
):
    """Decorator for robust operation execution with automatic recovery."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            context = {
                "function_name": func.__name__,
                "retry_count": 0,
                "fallback_value": fallback_value,
            }
            
            last_error = None
            
            for attempt in range(retry_count + 1):
                try:
                    return func(*args, **kwargs)
                    
                except Exception as e:
                    last_error = e
                    
                    # Record error
                    error_handler.record_error(
                        e,
                        severity=severity,
                        category=error_category,
                        context=context,
                        function_name=func.__name__
                    )
                    
                    # Attempt recovery on all but last attempt
                    if attempt < retry_count:
                        recovery_successful = error_handler.attempt_recovery(e, context)
                        
                        if recovery_successful:
                            logging.info(f"Recovery successful for {func.__name__}, retrying...")
                            time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
                            continue
                        else:
                            logging.warning(f"Recovery failed for {func.__name__}")
                    
                    # If this is the last attempt or recovery failed
                    if attempt == retry_count:
                        if fallback_value is not None:
                            logging.warning(f"Using fallback value for {func.__name__}")
                            return fallback_value
                        else:
                            raise last_error
                    
                    time.sleep(retry_delay * (attempt + 1))
            
            # Should not reach here
            raise last_error
            
        return wrapper
    return decorator


def safe_execute(
    func: Callable,
    *args,
    fallback_value: Any = None,
    timeout_seconds: Optional[float] = None,
    **kwargs
) -> Any:
    """Safely execute a function with timeout and error handling."""
    
    def target(result_container, error_container):
        try:
            result = func(*args, **kwargs)
            result_container[0] = result
        except Exception as e:
            error_container[0] = e
    
    if timeout_seconds:
        import threading
        
        result_container = [None]
        error_container = [None]
        
        thread = threading.Thread(target=target, args=(result_container, error_container))
        thread.start()
        thread.join(timeout=timeout_seconds)
        
        if thread.is_alive():
            # Thread is still running, timeout occurred
            logging.error(f"Function {func.__name__} timed out after {timeout_seconds}s")
            return fallback_value
        
        if error_container[0]:
            error_handler.record_error(
                error_container[0],
                severity=ErrorSeverity.MEDIUM,
                category=ErrorCategory.COMPUTATION,
                function_name=func.__name__
            )
            if fallback_value is not None:
                return fallback_value
            raise error_container[0]
        
        return result_container[0]
    else:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_handler.record_error(
                e,
                severity=ErrorSeverity.MEDIUM,
                category=ErrorCategory.COMPUTATION,
                function_name=func.__name__
            )
            if fallback_value is not None:
                return fallback_value
            raise


class CircuitBreaker:
    """Circuit breaker pattern for preventing cascading failures."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        timeout_seconds: float = 60.0,
        expected_exception: Type[Exception] = Exception
    ):
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        
        self.lock = threading.Lock()
    
    def __call__(self, func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with self.lock:
                if self.state == "OPEN":
                    if self._should_attempt_reset():
                        self.state = "HALF_OPEN"
                    else:
                        raise Exception("Circuit breaker is OPEN")
                
                try:
                    result = func(*args, **kwargs)
                    self._on_success()
                    return result
                    
                except self.expected_exception as e:
                    self._on_failure()
                    raise
        
        return wrapper
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self.last_failure_time is None:
            return True
        return time.time() - self.last_failure_time >= self.timeout_seconds
    
    def _on_success(self):
        """Reset circuit breaker on successful operation."""
        self.failure_count = 0
        self.state = "CLOSED"
    
    def _on_failure(self):
        """Handle failure in circuit breaker."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"


# Global error handler instance
error_handler = ErrorHandler()


# Example usage decorators
predict_safely = robust_operation(
    retry_count=2,
    error_category=ErrorCategory.MODEL,
    severity=ErrorSeverity.HIGH
)

validate_safely = robust_operation(
    retry_count=1,
    error_category=ErrorCategory.VALIDATION,
    severity=ErrorSeverity.MEDIUM,
    fallback_value={"valid": False, "errors": ["Validation failed"]}
)