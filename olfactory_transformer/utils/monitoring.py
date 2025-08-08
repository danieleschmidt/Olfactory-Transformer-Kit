"""Performance monitoring and resource tracking utilities."""

from typing import Dict, List, Optional, Any, Callable, Union
import time
import threading
import logging
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import deque, defaultdict
import subprocess
import psutil
import functools

import torch
import numpy as np


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    timestamp: float
    inference_time: float
    throughput: float
    memory_usage_mb: float
    gpu_memory_mb: Optional[float] = None
    cpu_usage_percent: float = 0.0
    gpu_usage_percent: Optional[float] = None
    batch_size: int = 1
    model_name: str = "unknown"
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ResourceSnapshot:
    """Snapshot of system resources."""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_total_gb: float
    disk_usage_percent: float
    network_sent_mb: float
    network_recv_mb: float
    gpu_memory_used_mb: Optional[float] = None
    gpu_memory_total_mb: Optional[float] = None
    gpu_utilization: Optional[float] = None
    temperature: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PerformanceMonitor:
    """Real-time performance monitoring for olfactory models."""
    
    def __init__(
        self,
        window_size: int = 100,
        log_interval: float = 60.0,
        enable_gpu_monitoring: bool = True
    ):
        self.window_size = window_size
        self.log_interval = log_interval
        self.enable_gpu_monitoring = enable_gpu_monitoring and torch.cuda.is_available()
        
        # Metrics storage
        self.metrics_history = deque(maxlen=window_size)
        self.aggregated_metrics = {}
        
        # Timing context managers
        self.active_timers = {}
        self.timer_lock = threading.Lock()
        
        # Background monitoring
        self.monitoring_active = False
        self.monitoring_thread = None
        
        # GPU monitoring setup
        if self.enable_gpu_monitoring:
            self._setup_gpu_monitoring()
    
    def _setup_gpu_monitoring(self) -> None:
        """Setup GPU monitoring utilities."""
        try:
            import pynvml
            pynvml.nvmlInit()
            self.gpu_count = pynvml.nvmlDeviceGetCount()
            self.gpu_handles = [
                pynvml.nvmlDeviceGetHandleByIndex(i) 
                for i in range(self.gpu_count)
            ]
            self.has_pynvml = True
        except ImportError:
            logging.warning("pynvml not available, GPU monitoring limited")
            self.has_pynvml = False
            self.gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    
    def time_inference(self, model_name: str = "model"):
        """Context manager for timing inference."""
        return InferenceTimer(self, model_name)
    
    def record_metrics(self, metrics: PerformanceMetrics) -> None:
        """Record performance metrics."""
        self.metrics_history.append(metrics)
        
        # Update aggregated metrics
        self._update_aggregated_metrics(metrics)
    
    def _update_aggregated_metrics(self, metrics: PerformanceMetrics) -> None:
        """Update running aggregated metrics."""
        model_name = metrics.model_name
        
        if model_name not in self.aggregated_metrics:
            self.aggregated_metrics[model_name] = {
                "count": 0,
                "total_time": 0.0,
                "total_throughput": 0.0,
                "max_memory": 0.0,
                "avg_batch_size": 0.0,
            }
        
        agg = self.aggregated_metrics[model_name]
        agg["count"] += 1
        agg["total_time"] += metrics.inference_time
        agg["total_throughput"] += metrics.throughput
        agg["max_memory"] = max(agg["max_memory"], metrics.memory_usage_mb)
        agg["avg_batch_size"] = (
            (agg["avg_batch_size"] * (agg["count"] - 1) + metrics.batch_size) / agg["count"]
        )
    
    def get_current_stats(self) -> Dict[str, Any]:
        """Get current performance statistics."""
        if not self.metrics_history:
            return {}
        
        recent_metrics = list(self.metrics_history)[-min(10, len(self.metrics_history)):]
        
        stats = {
            "recent_avg_inference_time": np.mean([m.inference_time for m in recent_metrics]),
            "recent_avg_throughput": np.mean([m.throughput for m in recent_metrics]),
            "recent_max_memory": max(m.memory_usage_mb for m in recent_metrics),
            "total_inferences": len(self.metrics_history),
        }
        
        # Add per-model statistics
        for model_name, agg in self.aggregated_metrics.items():
            stats[f"{model_name}_avg_time"] = agg["total_time"] / agg["count"]
            stats[f"{model_name}_avg_throughput"] = agg["total_throughput"] / agg["count"]
            stats[f"{model_name}_max_memory"] = agg["max_memory"]
            stats[f"{model_name}_count"] = agg["count"]
        
        return stats
    
    def start_background_monitoring(self) -> None:
        """Start background resource monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
        logging.info("Background monitoring started")
    
    def stop_background_monitoring(self) -> None:
        """Stop background resource monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        
        logging.info("Background monitoring stopped")
    
    def _monitoring_loop(self) -> None:
        """Background monitoring loop."""
        last_log_time = time.time()
        
        while self.monitoring_active:
            try:
                # Log stats periodically
                current_time = time.time()
                if current_time - last_log_time >= self.log_interval:
                    stats = self.get_current_stats()
                    if stats:
                        logging.info(f"Performance stats: {stats}")
                    last_log_time = current_time
                
                time.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logging.error(f"Error in monitoring loop: {e}")
                time.sleep(5)
    
    def export_metrics(self, file_path: Union[str, Path]) -> None:
        """Export metrics to file."""
        metrics_data = {
            "metrics_history": [m.to_dict() for m in self.metrics_history],
            "aggregated_metrics": self.aggregated_metrics,
            "export_timestamp": time.time(),
        }
        
        with open(file_path, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        
        logging.info(f"Metrics exported to {file_path}")


class InferenceTimer:
    """Context manager for timing inference operations."""
    
    def __init__(self, monitor: PerformanceMonitor, model_name: str = "model"):
        self.monitor = monitor
        self.model_name = model_name
        self.start_time = None
        self.start_memory = None
        
        # GPU monitoring
        self.start_gpu_memory = None
        
    def __enter__(self):
        """Start timing."""
        self.start_time = time.time()
        
        # Record memory usage
        process = psutil.Process()
        self.start_memory = process.memory_info().rss / (1024 * 1024)  # MB
        
        # GPU memory
        if self.monitor.enable_gpu_monitoring and torch.cuda.is_available():
            self.start_gpu_memory = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """End timing and record metrics."""
        end_time = time.time()
        inference_time = end_time - self.start_time
        
        # Memory usage
        process = psutil.Process()
        end_memory = process.memory_info().rss / (1024 * 1024)  # MB
        memory_usage = max(end_memory, self.start_memory)
        
        # GPU memory
        gpu_memory = None
        if self.monitor.enable_gpu_monitoring and torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
        
        # CPU usage
        cpu_usage = psutil.cpu_percent()
        
        # Create metrics
        metrics = PerformanceMetrics(
            timestamp=end_time,
            inference_time=inference_time,
            throughput=1.0 / inference_time,  # Basic throughput
            memory_usage_mb=memory_usage,
            gpu_memory_mb=gpu_memory,
            cpu_usage_percent=cpu_usage,
            model_name=self.model_name,
        )
        
        self.monitor.record_metrics(metrics)


def performance_monitor(model_name: str = "model"):
    """Decorator for monitoring function performance."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get or create monitor from global state
            monitor = getattr(wrapper, '_monitor', None)
            if monitor is None:
                monitor = PerformanceMonitor()
                wrapper._monitor = monitor
            
            with monitor.time_inference(model_name):
                return func(*args, **kwargs)
        
        return wrapper
    return decorator


class ResourceTracker:
    """System resource tracking and alerting."""
    
    def __init__(
        self,
        memory_threshold: float = 80.0,  # percent
        cpu_threshold: float = 90.0,     # percent
        disk_threshold: float = 85.0,    # percent
        gpu_memory_threshold: float = 90.0,  # percent
    ):
        self.memory_threshold = memory_threshold
        self.cpu_threshold = cpu_threshold
        self.disk_threshold = disk_threshold
        self.gpu_memory_threshold = gpu_memory_threshold
        
        # Resource history
        self.resource_history = deque(maxlen=1000)
        
        # Alert callbacks
        self.alert_callbacks = []
        
        # Monitoring state
        self.monitoring_active = False
        self.monitoring_thread = None
        
        # Network baseline
        self.network_baseline = self._get_network_stats()
        
    def add_alert_callback(self, callback: Callable[[str, Dict[str, Any]], None]) -> None:
        """Add callback for resource alerts."""
        self.alert_callbacks.append(callback)
    
    def get_resource_snapshot(self) -> ResourceSnapshot:
        """Get current resource snapshot."""
        # CPU and memory
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        # Disk usage
        disk = psutil.disk_usage('/')
        
        # Network stats
        network_current = self._get_network_stats()
        network_sent_mb = (network_current['bytes_sent'] - self.network_baseline['bytes_sent']) / (1024 * 1024)
        network_recv_mb = (network_current['bytes_recv'] - self.network_baseline['bytes_recv']) / (1024 * 1024)
        
        # GPU stats
        gpu_memory_used = None
        gpu_memory_total = None
        gpu_utilization = None
        temperature = None
        
        if torch.cuda.is_available():
            try:
                gpu_memory_used = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
                gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)  # MB
                
                # Try to get more detailed GPU stats
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    
                    # GPU utilization
                    gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_utilization = gpu_util.gpu
                    
                    # Temperature
                    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    temperature = temp
                    
                except ImportError:
                    pass
                    
            except Exception as e:
                logging.warning(f"Failed to get GPU stats: {e}")
        
        snapshot = ResourceSnapshot(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used_gb=memory.used / (1024 ** 3),
            memory_total_gb=memory.total / (1024 ** 3),
            disk_usage_percent=disk.percent,
            network_sent_mb=network_sent_mb,
            network_recv_mb=network_recv_mb,
            gpu_memory_used_mb=gpu_memory_used,
            gpu_memory_total_mb=gpu_memory_total,
            gpu_utilization=gpu_utilization,
            temperature=temperature,
        )
        
        return snapshot
    
    def _get_network_stats(self) -> Dict[str, int]:
        """Get network statistics."""
        net_io = psutil.net_io_counters()
        return {
            'bytes_sent': net_io.bytes_sent,
            'bytes_recv': net_io.bytes_recv,
        }
    
    def check_thresholds(self, snapshot: ResourceSnapshot) -> List[Dict[str, Any]]:
        """Check resource thresholds and return alerts."""
        alerts = []
        
        # Memory alert
        if snapshot.memory_percent > self.memory_threshold:
            alerts.append({
                "type": "high_memory",
                "value": snapshot.memory_percent,
                "threshold": self.memory_threshold,
                "message": f"Memory usage {snapshot.memory_percent:.1f}% exceeds threshold {self.memory_threshold}%"
            })
        
        # CPU alert
        if snapshot.cpu_percent > self.cpu_threshold:
            alerts.append({
                "type": "high_cpu",
                "value": snapshot.cpu_percent,
                "threshold": self.cpu_threshold,
                "message": f"CPU usage {snapshot.cpu_percent:.1f}% exceeds threshold {self.cpu_threshold}%"
            })
        
        # Disk alert
        if snapshot.disk_usage_percent > self.disk_threshold:
            alerts.append({
                "type": "high_disk",
                "value": snapshot.disk_usage_percent,
                "threshold": self.disk_threshold,
                "message": f"Disk usage {snapshot.disk_usage_percent:.1f}% exceeds threshold {self.disk_threshold}%"
            })
        
        # GPU memory alert
        if (snapshot.gpu_memory_used_mb and snapshot.gpu_memory_total_mb):
            gpu_memory_percent = (snapshot.gpu_memory_used_mb / snapshot.gpu_memory_total_mb) * 100
            if gpu_memory_percent > self.gpu_memory_threshold:
                alerts.append({
                    "type": "high_gpu_memory",
                    "value": gpu_memory_percent,
                    "threshold": self.gpu_memory_threshold,
                    "message": f"GPU memory usage {gpu_memory_percent:.1f}% exceeds threshold {self.gpu_memory_threshold}%"
                })
        
        return alerts
    
    def start_monitoring(self, interval: float = 30.0) -> None:
        """Start resource monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval,)
        )
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
        logging.info(f"Resource monitoring started (interval: {interval}s)")
    
    def stop_monitoring(self) -> None:
        """Stop resource monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        
        logging.info("Resource monitoring stopped")
    
    def _monitoring_loop(self, interval: float) -> None:
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                # Get resource snapshot
                snapshot = self.get_resource_snapshot()
                self.resource_history.append(snapshot)
                
                # Check thresholds
                alerts = self.check_thresholds(snapshot)
                
                # Trigger alert callbacks
                for alert in alerts:
                    for callback in self.alert_callbacks:
                        try:
                            callback(alert["type"], alert)
                        except Exception as e:
                            logging.error(f"Alert callback failed: {e}")
                
                time.sleep(interval)
                
            except Exception as e:
                logging.error(f"Error in resource monitoring: {e}")
                time.sleep(10)
    
    def get_resource_summary(self, hours: int = 1) -> Dict[str, Any]:
        """Get resource usage summary for the last N hours."""
        cutoff_time = time.time() - (hours * 3600)
        recent_snapshots = [
            s for s in self.resource_history 
            if s.timestamp >= cutoff_time
        ]
        
        if not recent_snapshots:
            return {}
        
        return {
            "period_hours": hours,
            "snapshots_count": len(recent_snapshots),
            "avg_cpu_percent": np.mean([s.cpu_percent for s in recent_snapshots]),
            "max_cpu_percent": max(s.cpu_percent for s in recent_snapshots),
            "avg_memory_percent": np.mean([s.memory_percent for s in recent_snapshots]),
            "max_memory_percent": max(s.memory_percent for s in recent_snapshots),
            "avg_gpu_utilization": np.mean([
                s.gpu_utilization for s in recent_snapshots 
                if s.gpu_utilization is not None
            ]) if any(s.gpu_utilization for s in recent_snapshots) else None,
            "max_temperature": max([
                s.temperature for s in recent_snapshots 
                if s.temperature is not None
            ]) if any(s.temperature for s in recent_snapshots) else None,
            "total_network_sent_mb": sum(s.network_sent_mb for s in recent_snapshots),
            "total_network_recv_mb": sum(s.network_recv_mb for s in recent_snapshots),
        }
    
    def export_resource_data(self, file_path: Union[str, Path]) -> None:
        """Export resource data to file."""
        data = {
            "snapshots": [s.to_dict() for s in self.resource_history],
            "thresholds": {
                "memory": self.memory_threshold,
                "cpu": self.cpu_threshold,
                "disk": self.disk_threshold,
                "gpu_memory": self.gpu_memory_threshold,
            },
            "export_timestamp": time.time(),
        }
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logging.info(f"Resource data exported to {file_path}")


# Example alert callback
def log_alert(alert_type: str, alert_data: Dict[str, Any]) -> None:
    """Simple logging alert callback."""
    logging.warning(f"RESOURCE ALERT [{alert_type}]: {alert_data['message']}")


# Example usage context manager
class ResourceMonitoringContext:
    """Context manager for temporary resource monitoring."""
    
    def __init__(self, tracker: ResourceTracker, interval: float = 10.0):
        self.tracker = tracker
        self.interval = interval
        
    def __enter__(self):
        self.tracker.start_monitoring(self.interval)
        return self.tracker
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.tracker.stop_monitoring()