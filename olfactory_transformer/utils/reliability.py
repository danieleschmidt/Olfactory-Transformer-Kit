"""Reliability and resilience utilities for production deployment."""

import time
import logging
import threading
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import functools
import traceback

import torch
import psutil


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Circuit tripped, blocking calls
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5
    success_threshold: int = 3
    timeout_seconds: float = 60.0
    exception_types: List[type] = field(default_factory=lambda: [Exception])
    
    
@dataclass
class HealthCheckResult:
    """Result of a health check."""
    component: str
    healthy: bool
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    response_time_ms: Optional[float] = None
    error: Optional[str] = None


class CircuitBreaker:
    """Circuit breaker implementation for fault tolerance."""
    
    def __init__(self, name: str, config: CircuitBreakerConfig):
        self.name = name
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.next_attempt_time = 0
        self.lock = threading.Lock()
        
        # Statistics
        self.total_requests = 0
        self.total_failures = 0
        self.total_successes = 0
        self.state_changes = []
        
        logging.info(f"Circuit breaker '{name}' initialized")
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator for circuit breaker."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return self.call(func, *args, **kwargs)
        return wrapper
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        with self.lock:
            self.total_requests += 1
            
            if self.state == CircuitState.OPEN:
                if time.time() < self.next_attempt_time:
                    raise RuntimeError(f"Circuit breaker '{self.name}' is OPEN")
                else:
                    # Transition to half-open
                    self.state = CircuitState.HALF_OPEN
                    self.success_count = 0
                    self._log_state_change("HALF_OPEN")
        
        try:
            result = func(*args, **kwargs)
            self._handle_success()
            return result
            
        except Exception as e:
            if any(isinstance(e, exc_type) for exc_type in self.config.exception_types):
                self._handle_failure()
            raise
    
    def _handle_success(self):
        """Handle successful call."""
        with self.lock:
            self.total_successes += 1
            self.failure_count = 0
            
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    self.state = CircuitState.CLOSED
                    self._log_state_change("CLOSED")
    
    def _handle_failure(self):
        """Handle failed call."""
        with self.lock:
            self.total_failures += 1
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.state == CircuitState.HALF_OPEN:
                # Immediately open on failure in half-open
                self.state = CircuitState.OPEN
                self.next_attempt_time = time.time() + self.config.timeout_seconds
                self._log_state_change("OPEN")
                
            elif (self.state == CircuitState.CLOSED and 
                  self.failure_count >= self.config.failure_threshold):
                # Transition to open
                self.state = CircuitState.OPEN
                self.next_attempt_time = time.time() + self.config.timeout_seconds
                self._log_state_change("OPEN")
    
    def _log_state_change(self, new_state: str):
        """Log state change."""
        self.state_changes.append({
            "timestamp": time.time(),
            "from_state": self.state.value if hasattr(self.state, 'value') else str(self.state),
            "to_state": new_state,
            "failure_count": self.failure_count,
            "total_failures": self.total_failures,
        })
        
        logging.warning(f"Circuit breaker '{self.name}' state changed to {new_state}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        with self.lock:
            return {
                "name": self.name,
                "state": self.state.value,
                "failure_count": self.failure_count,
                "success_count": self.success_count,
                "total_requests": self.total_requests,
                "total_failures": self.total_failures,
                "total_successes": self.total_successes,
                "failure_rate": self.total_failures / max(1, self.total_requests),
                "last_failure_time": self.last_failure_time,
                "next_attempt_time": self.next_attempt_time if self.state == CircuitState.OPEN else None,
                "state_changes": self.state_changes[-10:],  # Last 10 changes
            }
    
    def reset(self):
        """Reset circuit breaker to closed state."""
        with self.lock:
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.success_count = 0
            self.next_attempt_time = 0
            logging.info(f"Circuit breaker '{self.name}' manually reset")


class HealthChecker:
    """Comprehensive system health monitoring."""
    
    def __init__(self):
        self.checks: Dict[str, Callable] = {}
        self.results: Dict[str, HealthCheckResult] = {}
        self.check_intervals: Dict[str, float] = {}
        self.last_check_times: Dict[str, float] = {}
        self.lock = threading.Lock()
        
        # Built-in checks
        self.register_check("cpu_usage", self._check_cpu_usage, interval=30)
        self.register_check("memory_usage", self._check_memory_usage, interval=30)
        self.register_check("disk_usage", self._check_disk_usage, interval=60)
        self.register_check("gpu_availability", self._check_gpu_availability, interval=60)
        
        logging.info("Health checker initialized with built-in checks")
    
    def register_check(
        self, 
        name: str, 
        check_func: Callable[[], HealthCheckResult], 
        interval: float = 60.0
    ):
        """Register a health check."""
        with self.lock:
            self.checks[name] = check_func
            self.check_intervals[name] = interval
            self.last_check_times[name] = 0
        
        logging.info(f"Health check '{name}' registered with {interval}s interval")
    
    def run_check(self, name: str, force: bool = False) -> Optional[HealthCheckResult]:
        """Run a specific health check."""
        if name not in self.checks:
            logging.error(f"Unknown health check: {name}")
            return None
        
        current_time = time.time()
        
        # Check if we need to run this check
        if not force:
            last_check = self.last_check_times.get(name, 0)
            interval = self.check_intervals.get(name, 60)
            if current_time - last_check < interval:
                return self.results.get(name)
        
        try:
            start_time = time.time()
            result = self.checks[name]()
            end_time = time.time()
            
            result.response_time_ms = (end_time - start_time) * 1000
            result.timestamp = current_time
            
            with self.lock:
                self.results[name] = result
                self.last_check_times[name] = current_time
            
            return result
            
        except Exception as e:
            error_result = HealthCheckResult(
                component=name,
                healthy=False,
                error=str(e),
                timestamp=current_time
            )
            
            with self.lock:
                self.results[name] = error_result
                self.last_check_times[name] = current_time
            
            logging.error(f"Health check '{name}' failed: {e}")
            return error_result
    
    def run_all_checks(self, force: bool = False) -> Dict[str, HealthCheckResult]:
        """Run all registered health checks."""
        results = {}
        for name in self.checks:
            result = self.run_check(name, force)
            if result:
                results[name] = result
        return results
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status."""
        results = self.run_all_checks()
        
        healthy_checks = sum(1 for r in results.values() if r.healthy)
        total_checks = len(results)
        
        system_healthy = healthy_checks == total_checks
        
        return {
            "healthy": system_healthy,
            "timestamp": time.time(),
            "summary": {
                "total_checks": total_checks,
                "healthy_checks": healthy_checks,
                "failed_checks": total_checks - healthy_checks,
                "health_score": healthy_checks / max(1, total_checks),
            },
            "checks": {name: result.__dict__ for name, result in results.items()},
        }
    
    def _check_cpu_usage(self) -> HealthCheckResult:
        """Check CPU usage."""
        cpu_percent = psutil.cpu_percent(interval=1)
        
        return HealthCheckResult(
            component="cpu_usage",
            healthy=cpu_percent < 90,  # Consider unhealthy if > 90%
            details={
                "cpu_percent": cpu_percent,
                "cpu_count": psutil.cpu_count(),
                "load_average": psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None,
            }
        )
    
    def _check_memory_usage(self) -> HealthCheckResult:
        """Check memory usage."""
        memory = psutil.virtual_memory()
        
        return HealthCheckResult(
            component="memory_usage",
            healthy=memory.percent < 85,  # Consider unhealthy if > 85%
            details={
                "percent": memory.percent,
                "total_gb": memory.total / (1024**3),
                "available_gb": memory.available / (1024**3),
                "used_gb": memory.used / (1024**3),
            }
        )
    
    def _check_disk_usage(self) -> HealthCheckResult:
        """Check disk usage."""
        disk = psutil.disk_usage('/')
        
        return HealthCheckResult(
            component="disk_usage",
            healthy=disk.percent < 90,  # Consider unhealthy if > 90%
            details={
                "percent": disk.percent,
                "total_gb": disk.total / (1024**3),
                "free_gb": disk.free / (1024**3),
                "used_gb": disk.used / (1024**3),
            }
        )
    
    def _check_gpu_availability(self) -> HealthCheckResult:
        """Check GPU availability."""
        gpu_available = torch.cuda.is_available()
        
        details = {"cuda_available": gpu_available}
        
        if gpu_available:
            details.update({
                "device_count": torch.cuda.device_count(),
                "current_device": torch.cuda.current_device(),
                "device_name": torch.cuda.get_device_name(),
            })
            
            # Check GPU memory
            try:
                memory_allocated = torch.cuda.memory_allocated()
                memory_cached = torch.cuda.memory_reserved()
                details.update({
                    "memory_allocated_mb": memory_allocated / (1024**2),
                    "memory_cached_mb": memory_cached / (1024**2),
                })
            except Exception as e:
                details["memory_check_error"] = str(e)
        
        return HealthCheckResult(
            component="gpu_availability",
            healthy=True,  # GPU availability is not critical
            details=details
        )


class RetryHandler:
    """Configurable retry mechanism with exponential backoff."""
    
    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        retry_exceptions: List[type] = None
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.retry_exceptions = retry_exceptions or [Exception]
        
        # Statistics
        self.total_attempts = 0
        self.total_retries = 0
        self.successful_calls = 0
        self.failed_calls = 0
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator for retry mechanism."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return self.call(func, *args, **kwargs)
        return wrapper
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with retry logic."""
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            self.total_attempts += 1
            
            try:
                result = func(*args, **kwargs)
                if attempt > 0:
                    self.total_retries += attempt
                self.successful_calls += 1
                return result
                
            except Exception as e:
                last_exception = e
                
                # Check if this exception should trigger retry
                if not any(isinstance(e, exc_type) for exc_type in self.retry_exceptions):
                    raise
                
                # Don't retry on last attempt
                if attempt == self.max_retries:
                    break
                
                # Calculate delay
                delay = min(
                    self.base_delay * (self.exponential_base ** attempt),
                    self.max_delay
                )
                
                # Add jitter
                if self.jitter:
                    import random
                    delay *= (0.5 + 0.5 * random.random())
                
                logging.warning(f"Retry attempt {attempt + 1}/{self.max_retries} after {delay:.2f}s delay: {e}")
                time.sleep(delay)
        
        self.failed_calls += 1
        raise last_exception
    
    def get_stats(self) -> Dict[str, Any]:
        """Get retry statistics."""
        return {
            "total_attempts": self.total_attempts,
            "total_retries": self.total_retries,
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
            "success_rate": self.successful_calls / max(1, self.successful_calls + self.failed_calls),
            "avg_retries_per_call": self.total_retries / max(1, self.successful_calls + self.failed_calls),
        }


class GracefulShutdownHandler:
    """Handle graceful shutdown of the application."""
    
    def __init__(self):
        self.shutdown_requested = False
        self.shutdown_callbacks: List[Callable] = []
        self.shutdown_lock = threading.Lock()
        
        # Register signal handlers
        import signal
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
        
        logging.info("Graceful shutdown handler initialized")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logging.info(f"Received signal {signum}, initiating graceful shutdown")
        self.initiate_shutdown()
    
    def register_shutdown_callback(self, callback: Callable):
        """Register callback for shutdown."""
        with self.shutdown_lock:
            self.shutdown_callbacks.append(callback)
        
        logging.info(f"Registered shutdown callback: {callback.__name__}")
    
    def initiate_shutdown(self):
        """Initiate graceful shutdown."""
        with self.shutdown_lock:
            if self.shutdown_requested:
                return
            
            self.shutdown_requested = True
            
            logging.info(f"Starting graceful shutdown, running {len(self.shutdown_callbacks)} callbacks")
            
            for callback in reversed(self.shutdown_callbacks):  # Reverse order
                try:
                    callback()
                    logging.info(f"Shutdown callback completed: {callback.__name__}")
                except Exception as e:
                    logging.error(f"Error in shutdown callback {callback.__name__}: {e}")
            
            logging.info("Graceful shutdown completed")
    
    def is_shutdown_requested(self) -> bool:
        """Check if shutdown was requested."""
        return self.shutdown_requested


class ReliabilityManager:
    """Central reliability and resilience management."""
    
    def __init__(self):
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.health_checker = HealthChecker()
        self.retry_handlers: Dict[str, RetryHandler] = {}
        self.shutdown_handler = GracefulShutdownHandler()
        
        # Built-in circuit breakers
        self.create_circuit_breaker("model_inference", CircuitBreakerConfig(
            failure_threshold=5,
            timeout_seconds=30.0,
            exception_types=[RuntimeError, torch.cuda.OutOfMemoryError]
        ))
        
        self.create_circuit_breaker("tokenization", CircuitBreakerConfig(
            failure_threshold=10,
            timeout_seconds=10.0,
            exception_types=[ValueError, RuntimeError]
        ))
        
        # Built-in retry handlers
        self.create_retry_handler("api_requests", RetryHandler(
            max_retries=3,
            base_delay=1.0,
            retry_exceptions=[ConnectionError, TimeoutError]
        ))
        
        logging.info("Reliability manager initialized")
    
    def create_circuit_breaker(self, name: str, config: CircuitBreakerConfig) -> CircuitBreaker:
        """Create and register circuit breaker."""
        cb = CircuitBreaker(name, config)
        self.circuit_breakers[name] = cb
        return cb
    
    def get_circuit_breaker(self, name: str) -> Optional[CircuitBreaker]:
        """Get circuit breaker by name."""
        return self.circuit_breakers.get(name)
    
    def create_retry_handler(self, name: str, handler: RetryHandler) -> RetryHandler:
        """Create and register retry handler."""
        self.retry_handlers[name] = handler
        return handler
    
    def get_retry_handler(self, name: str) -> Optional[RetryHandler]:
        """Get retry handler by name."""
        return self.retry_handlers.get(name)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        circuit_breaker_stats = {
            name: cb.get_stats() for name, cb in self.circuit_breakers.items()
        }
        
        retry_stats = {
            name: handler.get_stats() for name, handler in self.retry_handlers.items()
        }
        
        health_status = self.health_checker.get_system_health()
        
        return {
            "timestamp": time.time(),
            "healthy": health_status["healthy"],
            "shutdown_requested": self.shutdown_handler.is_shutdown_requested(),
            "circuit_breakers": circuit_breaker_stats,
            "retry_handlers": retry_stats,
            "health_checks": health_status,
        }
    
    def register_shutdown_callback(self, callback: Callable):
        """Register shutdown callback."""
        self.shutdown_handler.register_shutdown_callback(callback)
    
    def initiate_shutdown(self):
        """Initiate graceful shutdown."""
        self.shutdown_handler.initiate_shutdown()


# Global reliability manager instance
reliability_manager = ReliabilityManager()


def circuit_breaker(name: str, config: Optional[CircuitBreakerConfig] = None):
    """Decorator for circuit breaker protection."""
    def decorator(func):
        cb_config = config or CircuitBreakerConfig()
        cb = reliability_manager.create_circuit_breaker(name, cb_config)
        return cb(func)
    return decorator


def retry(name: str, handler: Optional[RetryHandler] = None):
    """Decorator for retry logic."""
    def decorator(func):
        retry_handler = handler or RetryHandler()
        reliability_manager.create_retry_handler(name, retry_handler)
        return retry_handler(func)
    return decorator