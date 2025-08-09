"""Advanced observability and monitoring for production systems."""

import time
import logging
import threading
import json
from typing import Dict, Any, Optional, List, Callable, Union
from dataclasses import dataclass, field, asdict
from collections import defaultdict, deque
from contextlib import contextmanager
import functools
import uuid
import traceback
from pathlib import Path

import torch
import psutil


@dataclass
class MetricPoint:
    """Single metric measurement."""
    name: str
    value: Union[float, int]
    timestamp: float = field(default_factory=time.time)
    tags: Dict[str, str] = field(default_factory=dict)
    unit: Optional[str] = None
    

@dataclass 
class LogEvent:
    """Structured log event."""
    timestamp: float = field(default_factory=time.time)
    level: str = "INFO"
    message: str = ""
    logger_name: str = "olfactory_transformer"
    module: Optional[str] = None
    function: Optional[str] = None
    line_number: Optional[int] = None
    correlation_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    extra_data: Dict[str, Any] = field(default_factory=dict)
    exception_info: Optional[str] = None


@dataclass
class TraceSpan:
    """Distributed tracing span."""
    span_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_span_id: Optional[str] = None
    operation_name: str = ""
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None
    status: str = "active"  # active, success, error
    tags: Dict[str, str] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None


class MetricsCollector:
    """High-performance metrics collection and aggregation."""
    
    def __init__(self, buffer_size: int = 10000, flush_interval: float = 60.0):
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval
        
        # Thread-safe collections
        self.metrics_buffer = deque(maxlen=buffer_size)
        self.metric_aggregates: Dict[str, Dict] = defaultdict(lambda: {
            "count": 0, "sum": 0.0, "min": float("inf"), "max": float("-inf"),
            "last_value": 0.0, "last_timestamp": 0.0
        })
        self.lock = threading.RLock()
        
        # Exporters
        self.exporters: List[Callable[[List[MetricPoint]], None]] = []
        
        # Background flushing
        self.flush_thread = None
        self.should_stop = threading.Event()
        self._start_background_flushing()
        
        logging.info("Metrics collector initialized")
    
    def _start_background_flushing(self):
        """Start background thread for metric flushing."""
        def flush_worker():
            while not self.should_stop.wait(self.flush_interval):
                try:
                    self.flush_metrics()
                except Exception as e:
                    logging.error(f"Error flushing metrics: {e}")
        
        self.flush_thread = threading.Thread(target=flush_worker, daemon=True)
        self.flush_thread.start()
    
    def record_metric(
        self, 
        name: str, 
        value: Union[float, int],
        tags: Optional[Dict[str, str]] = None,
        unit: Optional[str] = None
    ):
        """Record a metric measurement."""
        metric = MetricPoint(
            name=name,
            value=float(value),
            tags=tags or {},
            unit=unit
        )
        
        with self.lock:
            self.metrics_buffer.append(metric)
            
            # Update aggregates
            key = f"{name}:{json.dumps(tags or {}, sort_keys=True)}"
            agg = self.metric_aggregates[key]
            agg["count"] += 1
            agg["sum"] += value
            agg["min"] = min(agg["min"], value)
            agg["max"] = max(agg["max"], value)
            agg["last_value"] = value
            agg["last_timestamp"] = metric.timestamp
    
    def increment_counter(
        self, 
        name: str, 
        value: Union[float, int] = 1,
        tags: Optional[Dict[str, str]] = None
    ):
        """Increment a counter metric."""
        self.record_metric(f"{name}.count", value, tags, "count")
    
    def record_gauge(
        self,
        name: str,
        value: Union[float, int], 
        tags: Optional[Dict[str, str]] = None
    ):
        """Record a gauge metric."""
        self.record_metric(f"{name}.gauge", value, tags, "gauge")
    
    def record_histogram(
        self,
        name: str,
        value: Union[float, int],
        tags: Optional[Dict[str, str]] = None
    ):
        """Record a histogram metric."""
        self.record_metric(f"{name}.histogram", value, tags, "milliseconds")
    
    @contextmanager
    def time_operation(
        self, 
        name: str, 
        tags: Optional[Dict[str, str]] = None
    ):
        """Time an operation and record as histogram."""
        start_time = time.time()
        try:
            yield
        finally:
            duration_ms = (time.time() - start_time) * 1000
            self.record_histogram(f"{name}.duration", duration_ms, tags)
    
    def get_metric_aggregates(self) -> Dict[str, Dict]:
        """Get current metric aggregates."""
        with self.lock:
            return dict(self.metric_aggregates)
    
    def flush_metrics(self):
        """Flush metrics to all registered exporters."""
        if not self.exporters:
            return
        
        with self.lock:
            if not self.metrics_buffer:
                return
            
            # Convert deque to list for exporters
            metrics_to_export = list(self.metrics_buffer)
            self.metrics_buffer.clear()
        
        # Export to all registered exporters
        for exporter in self.exporters:
            try:
                exporter(metrics_to_export)
            except Exception as e:
                logging.error(f"Error in metrics exporter {exporter}: {e}")
    
    def add_exporter(self, exporter: Callable[[List[MetricPoint]], None]):
        """Add a metrics exporter."""
        self.exporters.append(exporter)
        logging.info(f"Added metrics exporter: {exporter}")
    
    def stop(self):
        """Stop the metrics collector."""
        self.should_stop.set()
        if self.flush_thread:
            self.flush_thread.join(timeout=5.0)
        
        # Final flush
        self.flush_metrics()
        logging.info("Metrics collector stopped")


class StructuredLogger:
    """Advanced structured logging with correlation tracking."""
    
    def __init__(self, name: str = "olfactory_transformer"):
        self.name = name
        self.logger = logging.getLogger(name)
        self.correlation_context = threading.local()
        
        # Setup structured formatting
        self._setup_structured_handler()
        
        # Event buffer for analysis
        self.event_buffer = deque(maxlen=1000)
        self.lock = threading.Lock()
    
    def _setup_structured_handler(self):
        """Setup structured JSON logging handler."""
        handler = logging.StreamHandler()
        
        class StructuredFormatter(logging.Formatter):
            def format(self, record):
                log_event = {
                    "timestamp": record.created,
                    "level": record.levelname,
                    "message": record.getMessage(),
                    "logger_name": record.name,
                    "module": record.module if hasattr(record, 'module') else None,
                    "function": record.funcName,
                    "line_number": record.lineno,
                    "thread_id": record.thread,
                    "process_id": record.process,
                }
                
                # Add correlation info if available
                if hasattr(self, 'correlation_context'):
                    correlation_id = getattr(self.correlation_context, 'correlation_id', None)
                    if correlation_id:
                        log_event["correlation_id"] = correlation_id
                
                # Add exception info
                if record.exc_info:
                    log_event["exception_info"] = self.formatException(record.exc_info)
                
                # Add extra fields
                for key, value in record.__dict__.items():
                    if key not in log_event and not key.startswith('_'):
                        log_event[key] = value
                
                return json.dumps(log_event, default=str)
        
        handler.setFormatter(StructuredFormatter())
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def set_correlation_id(self, correlation_id: str):
        """Set correlation ID for current thread."""
        self.correlation_context.correlation_id = correlation_id
    
    def get_correlation_id(self) -> Optional[str]:
        """Get correlation ID for current thread."""
        return getattr(self.correlation_context, 'correlation_id', None)
    
    def info(self, message: str, **kwargs):
        """Log info message with structured data."""
        self._log("INFO", message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message with structured data."""
        self._log("WARNING", message, **kwargs)
    
    def error(self, message: str, exception: Optional[Exception] = None, **kwargs):
        """Log error message with structured data."""
        extra_data = kwargs
        if exception:
            extra_data['exception_type'] = type(exception).__name__
            extra_data['exception_message'] = str(exception)
            extra_data['stack_trace'] = traceback.format_exc()
        
        self._log("ERROR", message, **extra_data)
    
    def _log(self, level: str, message: str, **kwargs):
        """Internal logging method."""
        log_event = LogEvent(
            level=level,
            message=message,
            logger_name=self.name,
            correlation_id=self.get_correlation_id(),
            extra_data=kwargs
        )
        
        # Buffer event for analysis
        with self.lock:
            self.event_buffer.append(log_event)
        
        # Use standard logger
        log_method = getattr(self.logger, level.lower())
        log_method(message, extra=kwargs)
    
    def get_recent_events(self, limit: int = 100) -> List[LogEvent]:
        """Get recent log events."""
        with self.lock:
            return list(self.event_buffer)[-limit:]


class DistributedTracing:
    """Lightweight distributed tracing implementation."""
    
    def __init__(self):
        self.active_spans: Dict[str, TraceSpan] = {}
        self.completed_spans: deque = deque(maxlen=1000)
        self.lock = threading.RLock()
        
        # Trace context per thread
        self.trace_context = threading.local()
    
    @contextmanager
    def trace_operation(
        self,
        operation_name: str,
        tags: Optional[Dict[str, str]] = None,
        parent_span_id: Optional[str] = None
    ):
        """Trace an operation with automatic span management."""
        span = self.start_span(operation_name, tags, parent_span_id)
        
        try:
            yield span
            self.finish_span(span.span_id, "success")
        except Exception as e:
            self.finish_span(span.span_id, "error", str(e))
            raise
    
    def start_span(
        self,
        operation_name: str,
        tags: Optional[Dict[str, str]] = None,
        parent_span_id: Optional[str] = None
    ) -> TraceSpan:
        """Start a new tracing span."""
        # Get trace context
        current_trace_id = getattr(self.trace_context, 'trace_id', None)
        if not current_trace_id:
            current_trace_id = str(uuid.uuid4())
            self.trace_context.trace_id = current_trace_id
        
        span = TraceSpan(
            trace_id=current_trace_id,
            parent_span_id=parent_span_id or getattr(self.trace_context, 'current_span_id', None),
            operation_name=operation_name,
            tags=tags or {}
        )
        
        with self.lock:
            self.active_spans[span.span_id] = span
        
        # Set as current span
        self.trace_context.current_span_id = span.span_id
        
        return span
    
    def finish_span(
        self,
        span_id: str,
        status: str = "success",
        error: Optional[str] = None
    ):
        """Finish a tracing span."""
        with self.lock:
            span = self.active_spans.get(span_id)
            if not span:
                return
            
            span.end_time = time.time()
            span.duration_ms = (span.end_time - span.start_time) * 1000
            span.status = status
            span.error = error
            
            # Move to completed spans
            self.completed_spans.append(span)
            del self.active_spans[span_id]
        
        # Update trace context
        if getattr(self.trace_context, 'current_span_id', None) == span_id:
            self.trace_context.current_span_id = span.parent_span_id
    
    def add_span_log(self, span_id: str, message: str, **kwargs):
        """Add log entry to span."""
        with self.lock:
            span = self.active_spans.get(span_id)
            if span:
                span.logs.append({
                    "timestamp": time.time(),
                    "message": message,
                    **kwargs
                })
    
    def get_trace(self, trace_id: str) -> List[TraceSpan]:
        """Get all spans for a trace."""
        with self.lock:
            # Include both active and completed spans
            all_spans = list(self.active_spans.values()) + list(self.completed_spans)
            return [span for span in all_spans if span.trace_id == trace_id]
    
    def get_current_span(self) -> Optional[TraceSpan]:
        """Get current span for thread."""
        current_span_id = getattr(self.trace_context, 'current_span_id', None)
        if current_span_id:
            with self.lock:
                return self.active_spans.get(current_span_id)
        return None


class SystemMetricsCollector:
    """Collect system-level metrics for monitoring."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
        self.collection_interval = 30.0  # 30 seconds
        self.should_stop = threading.Event()
        self.collection_thread = None
        
        self._start_collection()
    
    def _start_collection(self):
        """Start background system metrics collection."""
        def collect_worker():
            while not self.should_stop.wait(self.collection_interval):
                try:
                    self._collect_system_metrics()
                except Exception as e:
                    logging.error(f"Error collecting system metrics: {e}")
        
        self.collection_thread = threading.Thread(target=collect_worker, daemon=True)
        self.collection_thread.start()
        
        logging.info("System metrics collection started")
    
    def _collect_system_metrics(self):
        """Collect various system metrics."""
        timestamp = time.time()
        
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        self.metrics.record_gauge("system.cpu.usage_percent", cpu_percent)
        
        # Memory metrics  
        memory = psutil.virtual_memory()
        self.metrics.record_gauge("system.memory.usage_percent", memory.percent)
        self.metrics.record_gauge("system.memory.available_gb", memory.available / (1024**3))
        self.metrics.record_gauge("system.memory.used_gb", memory.used / (1024**3))
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        self.metrics.record_gauge("system.disk.usage_percent", disk.percent)
        self.metrics.record_gauge("system.disk.free_gb", disk.free / (1024**3))
        
        # GPU metrics (if available)
        if torch.cuda.is_available():
            try:
                gpu_memory_allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
                gpu_memory_reserved = torch.cuda.memory_reserved() / (1024**3)   # GB
                
                self.metrics.record_gauge("system.gpu.memory_allocated_gb", gpu_memory_allocated)
                self.metrics.record_gauge("system.gpu.memory_reserved_gb", gpu_memory_reserved)
                
                # GPU utilization (if nvidia-ml-py3 available)
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    self.metrics.record_gauge("system.gpu.utilization_percent", gpu_util.gpu)
                except ImportError:
                    pass  # nvidia-ml-py3 not available
                except Exception as e:
                    logging.debug(f"GPU utilization collection failed: {e}")
                    
            except Exception as e:
                logging.debug(f"GPU metrics collection failed: {e}")
    
    def stop(self):
        """Stop system metrics collection."""
        self.should_stop.set()
        if self.collection_thread:
            self.collection_thread.join(timeout=5.0)
        logging.info("System metrics collection stopped")


class ObservabilityManager:
    """Central observability and monitoring coordination."""
    
    def __init__(self):
        self.metrics = MetricsCollector()
        self.logger = StructuredLogger()
        self.tracing = DistributedTracing()
        self.system_metrics = SystemMetricsCollector(self.metrics)
        
        # Add built-in exporters
        self._setup_default_exporters()
        
        logging.info("Observability manager initialized")
    
    def _setup_default_exporters(self):
        """Setup default metrics exporters."""
        def console_exporter(metrics: List[MetricPoint]):
            if metrics:
                logging.info(f"Exported {len(metrics)} metrics to console")
                for metric in metrics[-5:]:  # Show last 5 metrics
                    logging.debug(f"Metric: {metric.name}={metric.value} @{metric.timestamp}")
        
        self.metrics.add_exporter(console_exporter)
    
    def add_file_exporter(self, file_path: Path):
        """Add file-based metrics exporter."""
        def file_exporter(metrics: List[MetricPoint]):
            if not metrics:
                return
            
            try:
                with open(file_path, 'a') as f:
                    for metric in metrics:
                        f.write(f"{json.dumps(asdict(metric))}\n")
            except Exception as e:
                logging.error(f"File exporter error: {e}")
        
        self.metrics.add_exporter(file_exporter)
        logging.info(f"Added file metrics exporter: {file_path}")
    
    def create_correlation_context(self) -> str:
        """Create new correlation context for request tracking."""
        correlation_id = str(uuid.uuid4())
        self.logger.set_correlation_id(correlation_id)
        return correlation_id
    
    @contextmanager
    def trace_request(self, operation_name: str, **tags):
        """Trace entire request with metrics and logs."""
        correlation_id = self.create_correlation_context()
        
        with self.tracing.trace_operation(operation_name, tags):
            start_time = time.time()
            
            try:
                self.logger.info(f"Starting {operation_name}", correlation_id=correlation_id)
                yield correlation_id
                
                # Record success metrics
                duration_ms = (time.time() - start_time) * 1000
                self.metrics.record_histogram(f"{operation_name}.duration", duration_ms, tags)
                self.metrics.increment_counter(f"{operation_name}.success", 1, tags)
                
                self.logger.info(f"Completed {operation_name}", 
                               duration_ms=duration_ms, correlation_id=correlation_id)
                
            except Exception as e:
                # Record error metrics
                duration_ms = (time.time() - start_time) * 1000
                self.metrics.increment_counter(f"{operation_name}.error", 1, {**tags, "error_type": type(e).__name__})
                
                self.logger.error(f"Failed {operation_name}", 
                                exception=e, duration_ms=duration_ms, correlation_id=correlation_id)
                raise
    
    def get_observability_status(self) -> Dict[str, Any]:
        """Get comprehensive observability status."""
        return {
            "timestamp": time.time(),
            "metrics_buffer_size": len(self.metrics.metrics_buffer),
            "active_spans": len(self.tracing.active_spans),
            "completed_spans": len(self.tracing.completed_spans),
            "recent_log_events": len(self.logger.event_buffer),
            "metrics_aggregates_count": len(self.metrics.metric_aggregates),
        }
    
    def stop(self):
        """Stop all observability components."""
        self.metrics.stop()
        self.system_metrics.stop()
        logging.info("Observability manager stopped")


# Global observability manager instance
observability_manager = ObservabilityManager()


def trace_operation(operation_name: str, **tags):
    """Decorator for distributed tracing."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with observability_manager.tracing.trace_operation(operation_name, tags):
                return func(*args, **kwargs)
        return wrapper
    return decorator


def monitor_performance(operation_name: str, **tags):
    """Decorator for performance monitoring."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                
                # Record success metrics
                duration_ms = (time.time() - start_time) * 1000
                observability_manager.metrics.record_histogram(f"{operation_name}.duration", duration_ms, tags)
                observability_manager.metrics.increment_counter(f"{operation_name}.success", 1, tags)
                
                return result
                
            except Exception as e:
                # Record error metrics
                observability_manager.metrics.increment_counter(
                    f"{operation_name}.error", 1, 
                    {**tags, "error_type": type(e).__name__}
                )
                raise
        return wrapper
    return decorator