"""Comprehensive logging configuration for the Olfactory Transformer system."""

import logging
import logging.handlers
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import threading
import queue


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Add custom fields
        if hasattr(record, "user_id"):
            log_data["user_id"] = record.user_id
        if hasattr(record, "session_id"):
            log_data["session_id"] = record.session_id
        if hasattr(record, "operation"):
            log_data["operation"] = record.operation
        if hasattr(record, "duration_ms"):
            log_data["duration_ms"] = record.duration_ms
        
        return json.dumps(log_data)


class SecurityLogFilter(logging.Filter):
    """Filter for security-related log entries."""
    
    def __init__(self):
        super().__init__()
        self.security_keywords = [
            "security", "violation", "attack", "malicious", "unauthorized",
            "injection", "xss", "sql", "traversal", "blocked", "suspicious"
        ]
    
    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage().lower()
        return any(keyword in message for keyword in self.security_keywords)


class PerformanceLogFilter(logging.Filter):
    """Filter for performance-related log entries."""
    
    def filter(self, record: logging.LogRecord) -> bool:
        return hasattr(record, "duration_ms") or "performance" in record.getMessage().lower()


class AsyncLogHandler(logging.Handler):
    """Asynchronous log handler to prevent I/O blocking."""
    
    def __init__(self, target_handler: logging.Handler, queue_size: int = 1000):
        super().__init__()
        self.target_handler = target_handler
        self.log_queue = queue.Queue(maxsize=queue_size)
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()
        self._stop_event = threading.Event()
    
    def emit(self, record: logging.LogRecord):
        try:
            self.log_queue.put_nowait(record)
        except queue.Full:
            # Drop the log record if queue is full
            pass
    
    def _worker(self):
        while not self._stop_event.is_set():
            try:
                record = self.log_queue.get(timeout=1.0)
                self.target_handler.emit(record)
                self.log_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                # Log to stderr to avoid recursion
                print(f"Error in async log handler: {e}", file=sys.stderr)
    
    def close(self):
        self._stop_event.set()
        self.worker_thread.join(timeout=5.0)
        super().close()


class LoggingConfig:
    """Central logging configuration manager."""
    
    def __init__(self, log_dir: Optional[Path] = None, log_level: str = "INFO"):
        self.log_dir = log_dir or Path("logs")
        self.log_level = getattr(logging, log_level.upper())
        self.handlers_configured = False
        
        # Ensure log directory exists
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Log file paths
        self.main_log_file = self.log_dir / "olfactory_transformer.log"
        self.error_log_file = self.log_dir / "errors.log"
        self.security_log_file = self.log_dir / "security.log"
        self.performance_log_file = self.log_dir / "performance.log"
        self.access_log_file = self.log_dir / "access.log"
    
    def setup_logging(self, 
                     enable_console: bool = True,
                     enable_file: bool = True,
                     enable_json: bool = False,
                     enable_async: bool = True) -> None:
        """Setup comprehensive logging configuration."""
        
        if self.handlers_configured:
            return
        
        # Remove existing handlers
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Set root logger level
        root_logger.setLevel(self.log_level)
        
        # Standard formatter
        standard_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Detailed formatter
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(funcName)s:%(lineno)d - %(message)s'
        )
        
        # JSON formatter
        json_formatter = JSONFormatter()
        
        # Console handler
        if enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(self.log_level)
            console_handler.setFormatter(standard_formatter)
            
            if enable_async:
                console_handler = AsyncLogHandler(console_handler)
            
            root_logger.addHandler(console_handler)
        
        # File handlers
        if enable_file:
            # Main log file (rotating)
            main_handler = logging.handlers.RotatingFileHandler(
                self.main_log_file,
                maxBytes=50 * 1024 * 1024,  # 50MB
                backupCount=5,
                encoding='utf-8'
            )
            main_handler.setLevel(self.log_level)
            formatter = json_formatter if enable_json else detailed_formatter
            main_handler.setFormatter(formatter)
            
            if enable_async:
                main_handler = AsyncLogHandler(main_handler)
            
            root_logger.addHandler(main_handler)
            
            # Error log file (errors and above)
            error_handler = logging.handlers.RotatingFileHandler(
                self.error_log_file,
                maxBytes=10 * 1024 * 1024,  # 10MB
                backupCount=3,
                encoding='utf-8'
            )
            error_handler.setLevel(logging.ERROR)
            error_handler.setFormatter(detailed_formatter)
            
            if enable_async:
                error_handler = AsyncLogHandler(error_handler)
            
            root_logger.addHandler(error_handler)
        
        # Setup specialized loggers
        self._setup_security_logging(enable_async)
        self._setup_performance_logging(enable_async)
        self._setup_access_logging(enable_async)
        
        self.handlers_configured = True
        logging.info("Logging configuration completed")
    
    def _setup_security_logging(self, enable_async: bool = True):
        """Setup security-specific logging."""
        security_logger = logging.getLogger('security')
        security_logger.setLevel(logging.WARNING)
        
        # Security log handler
        security_handler = logging.handlers.RotatingFileHandler(
            self.security_log_file,
            maxBytes=20 * 1024 * 1024,  # 20MB
            backupCount=5,
            encoding='utf-8'
        )
        security_handler.setLevel(logging.WARNING)
        security_handler.setFormatter(JSONFormatter())
        security_handler.addFilter(SecurityLogFilter())
        
        if enable_async:
            security_handler = AsyncLogHandler(security_handler)
        
        security_logger.addHandler(security_handler)
        security_logger.propagate = False  # Don't propagate to root logger
    
    def _setup_performance_logging(self, enable_async: bool = True):
        """Setup performance-specific logging."""
        performance_logger = logging.getLogger('performance')
        performance_logger.setLevel(logging.INFO)
        
        # Performance log handler
        perf_handler = logging.handlers.RotatingFileHandler(
            self.performance_log_file,
            maxBytes=30 * 1024 * 1024,  # 30MB
            backupCount=3,
            encoding='utf-8'
        )
        perf_handler.setLevel(logging.INFO)
        perf_handler.setFormatter(JSONFormatter())
        perf_handler.addFilter(PerformanceLogFilter())
        
        if enable_async:
            perf_handler = AsyncLogHandler(perf_handler)
        
        performance_logger.addHandler(perf_handler)
        performance_logger.propagate = False
    
    def _setup_access_logging(self, enable_async: bool = True):
        """Setup access logging for API requests."""
        access_logger = logging.getLogger('access')
        access_logger.setLevel(logging.INFO)
        
        # Access log handler
        access_handler = logging.handlers.RotatingFileHandler(
            self.access_log_file,
            maxBytes=50 * 1024 * 1024,  # 50MB
            backupCount=7,
            encoding='utf-8'
        )
        access_handler.setLevel(logging.INFO)
        
        # Access log format
        access_formatter = logging.Formatter(
            '%(asctime)s - %(remote_addr)s - %(method)s %(path)s - %(status)s - %(response_time)sms'
        )
        access_handler.setFormatter(access_formatter)
        
        if enable_async:
            access_handler = AsyncLogHandler(access_handler)
        
        access_logger.addHandler(access_handler)
        access_logger.propagate = False
    
    def get_logger(self, name: str) -> logging.Logger:
        """Get a configured logger instance."""
        return logging.getLogger(name)
    
    def log_performance(self, operation: str, duration_ms: float, **kwargs):
        """Log performance metrics."""
        perf_logger = logging.getLogger('performance')
        
        extra = {
            'operation': operation,
            'duration_ms': duration_ms,
            **kwargs
        }
        
        perf_logger.info(f"Performance: {operation} took {duration_ms:.2f}ms", extra=extra)
    
    def log_security_event(self, event_type: str, details: Dict[str, Any], severity: str = "warning"):
        """Log security events."""
        security_logger = logging.getLogger('security')
        
        extra = {
            'event_type': event_type,
            'details': details
        }
        
        level = getattr(logging, severity.upper(), logging.WARNING)
        security_logger.log(level, f"Security event: {event_type}", extra=extra)
    
    def log_access(self, method: str, path: str, status: int, response_time: float, 
                   remote_addr: str = "unknown", user_id: Optional[str] = None):
        """Log API access."""
        access_logger = logging.getLogger('access')
        
        extra = {
            'method': method,
            'path': path,
            'status': status,
            'response_time': response_time,
            'remote_addr': remote_addr,
            'user_id': user_id
        }
        
        access_logger.info("API access", extra=extra)
    
    def cleanup(self):
        """Cleanup logging handlers."""
        # Close all async handlers
        for handler in logging.getLogger().handlers:
            if isinstance(handler, AsyncLogHandler):
                handler.close()


class ContextualLogger:
    """Logger with automatic context injection."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.context = {}
    
    def set_context(self, **kwargs):
        """Set contextual information."""
        self.context.update(kwargs)
    
    def clear_context(self):
        """Clear contextual information."""
        self.context.clear()
    
    def _log_with_context(self, level: int, msg: str, *args, **kwargs):
        """Log with automatic context injection."""
        extra = kwargs.get('extra', {})
        extra.update(self.context)
        kwargs['extra'] = extra
        
        self.logger.log(level, msg, *args, **kwargs)
    
    def debug(self, msg: str, *args, **kwargs):
        self._log_with_context(logging.DEBUG, msg, *args, **kwargs)
    
    def info(self, msg: str, *args, **kwargs):
        self._log_with_context(logging.INFO, msg, *args, **kwargs)
    
    def warning(self, msg: str, *args, **kwargs):
        self._log_with_context(logging.WARNING, msg, *args, **kwargs)
    
    def error(self, msg: str, *args, **kwargs):
        self._log_with_context(logging.ERROR, msg, *args, **kwargs)
    
    def critical(self, msg: str, *args, **kwargs):
        self._log_with_context(logging.CRITICAL, msg, *args, **kwargs)


def get_contextual_logger(name: str) -> ContextualLogger:
    """Get a contextual logger instance."""
    return ContextualLogger(logging.getLogger(name))


# Performance logging decorator
def log_performance(operation_name: Optional[str] = None):
    """Decorator to automatically log function performance."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            operation = operation_name or f"{func.__module__}.{func.__name__}"
            
            try:
                result = func(*args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000
                
                # Log successful operation
                perf_logger = logging.getLogger('performance')
                perf_logger.info(
                    f"Operation completed: {operation}",
                    extra={
                        'operation': operation,
                        'duration_ms': duration_ms,
                        'status': 'success'
                    }
                )
                
                return result
                
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                
                # Log failed operation
                perf_logger = logging.getLogger('performance')
                perf_logger.error(
                    f"Operation failed: {operation}",
                    extra={
                        'operation': operation,
                        'duration_ms': duration_ms,
                        'status': 'error',
                        'error_type': type(e).__name__,
                        'error_message': str(e)
                    }
                )
                
                raise
        
        return wrapper
    return decorator


# Global logging config instance
logging_config = LoggingConfig()


def setup_production_logging():
    """Setup production-ready logging configuration."""
    logging_config.setup_logging(
        enable_console=True,
        enable_file=True,
        enable_json=True,
        enable_async=True
    )


def setup_development_logging():
    """Setup development-friendly logging configuration."""
    logging_config.setup_logging(
        enable_console=True,
        enable_file=True,
        enable_json=False,
        enable_async=False
    )