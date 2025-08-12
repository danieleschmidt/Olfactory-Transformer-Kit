"""Performance optimization and monitoring utilities."""

import time
import threading
import multiprocessing
import concurrent.futures
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, field
from collections import deque, defaultdict
import logging
import functools
import asyncio
from pathlib import Path
import json

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


@dataclass
class PerformanceMetrics:
    """Performance metrics container."""
    operation_name: str
    start_time: float
    end_time: float
    duration: float
    memory_usage_mb: float = 0.0
    cpu_percent: float = 0.0
    throughput: float = 0.0  # items/second
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchProcessingConfig:
    """Configuration for batch processing optimization."""
    max_batch_size: int = 32
    max_workers: int = None  # Auto-detect
    use_threading: bool = True  # vs multiprocessing
    timeout_seconds: float = 300.0
    memory_limit_mb: float = 1024.0
    adaptive_batching: bool = True


class PerformanceMonitor:
    """Monitor and track performance metrics."""
    
    def __init__(self, history_size: int = 1000):
        self.history_size = history_size
        self.metrics_history = deque(maxlen=history_size)
        self.operation_stats = defaultdict(list)
        self.lock = threading.Lock()
        
    def time_operation(self, operation_name: str):
        """Context manager for timing operations."""
        return _OperationTimer(self, operation_name)
    
    def record_metric(self, metric: PerformanceMetrics):
        """Record a performance metric."""
        with self.lock:
            self.metrics_history.append(metric)
            self.operation_stats[metric.operation_name].append(metric)
            
            # Limit per-operation history
            if len(self.operation_stats[metric.operation_name]) > 100:
                self.operation_stats[metric.operation_name] = \
                    self.operation_stats[metric.operation_name][-100:]
    
    def get_stats(self, operation_name: Optional[str] = None) -> Dict[str, Any]:
        """Get performance statistics."""
        with self.lock:
            if operation_name:
                metrics = self.operation_stats.get(operation_name, [])
            else:
                metrics = list(self.metrics_history)
            
            if not metrics:
                return {"operation": operation_name, "count": 0}
            
            durations = [m.duration for m in metrics]
            memory_usage = [m.memory_usage_mb for m in metrics if m.memory_usage_mb > 0]
            throughputs = [m.throughput for m in metrics if m.throughput > 0]
            
            stats = {
                "operation": operation_name,
                "count": len(metrics),
                "avg_duration": sum(durations) / len(durations),
                "min_duration": min(durations),
                "max_duration": max(durations),
                "total_duration": sum(durations),
            }
            
            if memory_usage:
                stats.update({
                    "avg_memory_mb": sum(memory_usage) / len(memory_usage),
                    "max_memory_mb": max(memory_usage),
                })
            
            if throughputs:
                stats.update({
                    "avg_throughput": sum(throughputs) / len(throughputs),
                    "max_throughput": max(throughputs),
                })
            
            return stats
    
    def get_recent_performance(self, minutes: int = 5) -> Dict[str, Any]:
        """Get performance metrics for recent time window."""
        cutoff_time = time.time() - (minutes * 60)
        
        with self.lock:
            recent_metrics = [
                m for m in self.metrics_history 
                if m.start_time >= cutoff_time
            ]
            
            if not recent_metrics:
                return {"recent_minutes": minutes, "operations": 0}
            
            # Group by operation
            by_operation = defaultdict(list)
            for metric in recent_metrics:
                by_operation[metric.operation_name].append(metric)
            
            operation_stats = {}
            for op_name, op_metrics in by_operation.items():
                durations = [m.duration for m in op_metrics]
                operation_stats[op_name] = {
                    "count": len(op_metrics),
                    "avg_duration": sum(durations) / len(durations),
                    "ops_per_minute": len(op_metrics) / minutes,
                }
            
            return {
                "recent_minutes": minutes,
                "operations": len(recent_metrics),
                "unique_operations": len(by_operation),
                "by_operation": operation_stats,
            }


class _OperationTimer:
    """Context manager for timing operations."""
    
    def __init__(self, monitor: PerformanceMonitor, operation_name: str):
        self.monitor = monitor
        self.operation_name = operation_name
        self.start_time = None
        self.start_memory = None
        
    def __enter__(self):
        self.start_time = time.time()
        if HAS_PSUTIL:
            try:
                process = psutil.Process()
                self.start_memory = process.memory_info().rss / 1024 / 1024  # MB
            except:
                self.start_memory = 0
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.time()
        duration = end_time - self.start_time
        
        memory_usage = 0.0
        if HAS_PSUTIL and self.start_memory:
            try:
                process = psutil.Process()
                end_memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_usage = end_memory - self.start_memory
            except:
                pass
        
        metric = PerformanceMetrics(
            operation_name=self.operation_name,
            start_time=self.start_time,
            end_time=end_time,
            duration=duration,
            memory_usage_mb=memory_usage,
        )
        
        self.monitor.record_metric(metric)


class BatchProcessor:
    """Optimized batch processing for high throughput."""
    
    def __init__(self, config: BatchProcessingConfig = None):
        self.config = config or BatchProcessingConfig()
        if self.config.max_workers is None:
            self.config.max_workers = min(8, multiprocessing.cpu_count())
        
        self.performance_monitor = PerformanceMonitor()
        self.processed_count = 0
        self.error_count = 0
        
    def process_batch_sync(
        self,
        items: List[Any],
        process_func: Callable,
        *args,
        **kwargs
    ) -> List[Any]:
        """Process items in batches synchronously."""
        
        with self.performance_monitor.time_operation("batch_process_sync"):
            results = []
            errors = []
            
            # Split into batches
            batches = self._create_batches(items)
            
            if self.config.use_threading:
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=self.config.max_workers
                ) as executor:
                    futures = []
                    for batch in batches:
                        future = executor.submit(
                            self._process_single_batch,
                            batch, process_func, *args, **kwargs
                        )
                        futures.append(future)
                    
                    # Collect results
                    for future in concurrent.futures.as_completed(
                        futures, timeout=self.config.timeout_seconds
                    ):
                        try:
                            batch_results = future.result()
                            results.extend(batch_results)
                            self.processed_count += len(batch_results)
                        except Exception as e:
                            logging.error(f"Batch processing error: {e}")
                            errors.append(e)
                            self.error_count += 1
            else:
                # Process sequentially for simplicity
                for batch in batches:
                    try:
                        batch_results = self._process_single_batch(
                            batch, process_func, *args, **kwargs
                        )
                        results.extend(batch_results)
                        self.processed_count += len(batch_results)
                    except Exception as e:
                        logging.error(f"Batch processing error: {e}")
                        errors.append(e)
                        self.error_count += 1
            
            if errors:
                logging.warning(f"Batch processing completed with {len(errors)} errors")
            
            return results
    
    async def process_batch_async(
        self,
        items: List[Any],
        process_func: Callable,
        *args,
        **kwargs
    ) -> List[Any]:
        """Process items in batches asynchronously."""
        
        results = []
        batches = self._create_batches(items)
        
        semaphore = asyncio.Semaphore(self.config.max_workers)
        
        async def process_batch_with_semaphore(batch):
            async with semaphore:
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(
                    None, self._process_single_batch,
                    batch, process_func, *args, **kwargs
                )
        
        tasks = [process_batch_with_semaphore(batch) for batch in batches]
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for batch_result in batch_results:
            if isinstance(batch_result, Exception):
                logging.error(f"Async batch error: {batch_result}")
                self.error_count += 1
            else:
                results.extend(batch_result)
                self.processed_count += len(batch_result)
        
        return results
    
    def _create_batches(self, items: List[Any]) -> List[List[Any]]:
        """Create batches from items list."""
        batch_size = self.config.max_batch_size
        
        if self.config.adaptive_batching and HAS_PSUTIL:
            # Adjust batch size based on memory usage
            try:
                memory_percent = psutil.virtual_memory().percent
                if memory_percent > 80:
                    batch_size = max(1, batch_size // 2)
                elif memory_percent < 50:
                    batch_size = min(batch_size * 2, len(items))
            except:
                pass
        
        batches = []
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            batches.append(batch)
        
        return batches
    
    def _process_single_batch(
        self,
        batch: List[Any],
        process_func: Callable,
        *args,
        **kwargs
    ) -> List[Any]:
        """Process a single batch of items."""
        results = []
        
        for item in batch:
            try:
                result = process_func(item, *args, **kwargs)
                results.append(result)
            except Exception as e:
                logging.debug(f"Item processing error: {e}")
                # Continue with other items
                results.append(None)
        
        return results
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = self.performance_monitor.get_stats("batch_process_sync")
        stats.update({
            "total_processed": self.processed_count,
            "total_errors": self.error_count,
            "error_rate": self.error_count / max(1, self.processed_count + self.error_count),
        })
        return stats


class CacheManager:
    """Intelligent caching for performance optimization."""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: float = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache = {}
        self.access_times = {}
        self.access_counts = defaultdict(int)
        self.lock = threading.Lock()
        
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        with self.lock:
            if key not in self.cache:
                return None
            
            # Check TTL
            if time.time() - self.access_times[key] > self.ttl_seconds:
                self._remove_key(key)
                return None
            
            # Update access stats
            self.access_times[key] = time.time()
            self.access_counts[key] += 1
            
            return self.cache[key]
    
    def put(self, key: str, value: Any) -> None:
        """Put item in cache."""
        with self.lock:
            # Check if we need to evict
            if len(self.cache) >= self.max_size and key not in self.cache:
                self._evict_lru()
            
            self.cache[key] = value
            self.access_times[key] = time.time()
            self.access_counts[key] += 1
    
    def _evict_lru(self) -> None:
        """Evict least recently used item."""
        if not self.cache:
            return
        
        # Find LRU key
        lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        self._remove_key(lru_key)
    
    def _remove_key(self, key: str) -> None:
        """Remove key from cache."""
        if key in self.cache:
            del self.cache[key]
            del self.access_times[key]
            if key in self.access_counts:
                del self.access_counts[key]
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self.lock:
            self.cache.clear()
            self.access_times.clear()
            self.access_counts.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total_accesses = sum(self.access_counts.values())
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "utilization": len(self.cache) / self.max_size,
                "total_accesses": total_accesses,
                "average_accesses": total_accesses / max(1, len(self.cache)),
            }


def performance_profile(operation_name: str = None):
    """Decorator for performance profiling."""
    def decorator(func: Callable) -> Callable:
        op_name = operation_name or f"{func.__module__}.{func.__name__}"
        monitor = PerformanceMonitor()
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with monitor.time_operation(op_name):
                return func(*args, **kwargs)
        
        # Attach monitor to function for stats access
        wrapper._performance_monitor = monitor
        return wrapper
    
    return decorator


def memoize_with_ttl(ttl_seconds: float = 3600, max_size: int = 128):
    """Decorator for memoization with TTL."""
    def decorator(func: Callable) -> Callable:
        cache = CacheManager(max_size=max_size, ttl_seconds=ttl_seconds)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from arguments
            key = _create_cache_key(func.__name__, args, kwargs)
            
            # Try cache first
            result = cache.get(key)
            if result is not None:
                return result
            
            # Compute and cache
            result = func(*args, **kwargs)
            cache.put(key, result)
            return result
        
        wrapper._cache = cache
        return wrapper
    
    return decorator


def _create_cache_key(func_name: str, args: tuple, kwargs: dict) -> str:
    """Create a cache key from function arguments."""
    import hashlib
    
    # Convert arguments to string representation
    args_str = str(args)
    kwargs_str = str(sorted(kwargs.items()))
    combined = f"{func_name}:{args_str}:{kwargs_str}"
    
    # Hash for consistent key length
    return hashlib.md5(combined.encode()).hexdigest()


class ResourceMonitor:
    """Monitor system resources during operations."""
    
    def __init__(self, check_interval: float = 1.0):
        self.check_interval = check_interval
        self.monitoring = False
        self.resource_history = deque(maxlen=1000)
        self.monitor_thread = None
        
    def start_monitoring(self):
        """Start resource monitoring."""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
    def stop_monitoring(self):
        """Stop resource monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring:
            try:
                if HAS_PSUTIL:
                    snapshot = {
                        "timestamp": time.time(),
                        "cpu_percent": psutil.cpu_percent(),
                        "memory_percent": psutil.virtual_memory().percent,
                        "disk_io": psutil.disk_io_counters()._asdict() if psutil.disk_io_counters() else {},
                        "network_io": psutil.net_io_counters()._asdict() if psutil.net_io_counters() else {},
                    }
                    self.resource_history.append(snapshot)
                
                time.sleep(self.check_interval)
            except Exception as e:
                logging.debug(f"Resource monitoring error: {e}")
                time.sleep(self.check_interval)
    
    def get_current_usage(self) -> Dict[str, Any]:
        """Get current resource usage."""
        if not HAS_PSUTIL:
            return {"error": "psutil not available"}
        
        try:
            return {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "memory_used_gb": psutil.virtual_memory().used / 1024**3,
                "memory_total_gb": psutil.virtual_memory().total / 1024**3,
                "disk_usage_percent": psutil.disk_usage('/').percent,
                "load_average": psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None,
            }
        except Exception as e:
            return {"error": str(e)}
    
    def get_peak_usage(self, minutes: int = 5) -> Dict[str, Any]:
        """Get peak resource usage over time window."""
        if not self.resource_history:
            return {}
        
        cutoff_time = time.time() - (minutes * 60)
        recent_snapshots = [
            s for s in self.resource_history 
            if s["timestamp"] >= cutoff_time
        ]
        
        if not recent_snapshots:
            return {}
        
        cpu_values = [s["cpu_percent"] for s in recent_snapshots]
        memory_values = [s["memory_percent"] for s in recent_snapshots]
        
        return {
            "peak_cpu_percent": max(cpu_values),
            "peak_memory_percent": max(memory_values),
            "avg_cpu_percent": sum(cpu_values) / len(cpu_values),
            "avg_memory_percent": sum(memory_values) / len(memory_values),
            "samples": len(recent_snapshots),
        }


# Global instances
_global_performance_monitor = PerformanceMonitor()
_global_resource_monitor = ResourceMonitor()

def get_global_performance_monitor() -> PerformanceMonitor:
    """Get global performance monitor instance."""
    return _global_performance_monitor

def get_global_resource_monitor() -> ResourceMonitor:
    """Get global resource monitor instance."""
    return _global_resource_monitor