"""Auto-scaling and load balancing utilities for the Olfactory Transformer system."""

import time
import threading
import logging
import json
import subprocess
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from collections import deque, defaultdict
from pathlib import Path
import queue
import psutil

try:
    import docker
    HAS_DOCKER = True
except ImportError:
    HAS_DOCKER = False

try:
    import kubernetes
    from kubernetes import client, config
    HAS_KUBERNETES = True
except ImportError:
    HAS_KUBERNETES = False


@dataclass
class MetricSnapshot:
    """System metrics snapshot for scaling decisions."""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    active_requests: int
    response_time_ms: float
    queue_length: int
    error_rate: float
    throughput_rps: float  # Requests per second
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ScalingEvent:
    """Record of a scaling event."""
    timestamp: float
    action: str  # scale_up, scale_down
    current_instances: int
    target_instances: int
    trigger_metric: str
    metric_value: float
    threshold: float
    reason: str


class LoadBalancer:
    """Simple round-robin load balancer for distributed inference."""
    
    def __init__(self):
        self.instances = []
        self.current_index = 0
        self.instance_health = {}
        self.lock = threading.Lock()
        
        # Health check settings
        self.health_check_interval = 30.0  # seconds
        self.health_checker_thread = None
        self.health_check_running = False
        
        logging.info("Load balancer initialized")
    
    def add_instance(self, instance_id: str, endpoint: str, weight: float = 1.0):
        """Add an instance to the load balancer."""
        with self.lock:
            instance = {
                "id": instance_id,
                "endpoint": endpoint,
                "weight": weight,
                "active": True,
                "request_count": 0,
                "last_used": time.time(),
            }
            
            self.instances.append(instance)
            self.instance_health[instance_id] = {
                "healthy": True,
                "last_check": time.time(),
                "consecutive_failures": 0,
            }
            
            logging.info(f"Added instance {instance_id} at {endpoint}")
    
    def remove_instance(self, instance_id: str):
        """Remove an instance from the load balancer."""
        with self.lock:
            self.instances = [inst for inst in self.instances if inst["id"] != instance_id]
            self.instance_health.pop(instance_id, None)
            
            logging.info(f"Removed instance {instance_id}")
    
    def get_next_instance(self) -> Optional[Dict[str, Any]]:
        """Get next available instance using round-robin with health checks."""
        with self.lock:
            if not self.instances:
                return None
            
            # Filter healthy instances
            healthy_instances = [
                inst for inst in self.instances
                if inst["active"] and self.instance_health.get(inst["id"], {}).get("healthy", False)
            ]
            
            if not healthy_instances:
                logging.warning("No healthy instances available")
                return None
            
            # Round-robin selection
            selected = healthy_instances[self.current_index % len(healthy_instances)]
            self.current_index = (self.current_index + 1) % len(healthy_instances)
            
            # Update usage stats
            selected["request_count"] += 1
            selected["last_used"] = time.time()
            
            return selected
    
    def mark_instance_unhealthy(self, instance_id: str):
        """Mark an instance as unhealthy."""
        with self.lock:
            if instance_id in self.instance_health:
                health = self.instance_health[instance_id]
                health["healthy"] = False
                health["consecutive_failures"] += 1
                
                logging.warning(f"Instance {instance_id} marked unhealthy")
    
    def start_health_checks(self):
        """Start background health checking."""
        if self.health_check_running:
            return
        
        self.health_check_running = True
        self.health_checker_thread = threading.Thread(target=self._health_check_loop, daemon=True)
        self.health_checker_thread.start()
        
        logging.info("Health checking started")
    
    def stop_health_checks(self):
        """Stop background health checking."""
        self.health_check_running = False
        if self.health_checker_thread:
            self.health_checker_thread.join(timeout=5.0)
    
    def _health_check_loop(self):
        """Background health checking loop."""
        while self.health_check_running:
            try:
                with self.lock:
                    instances_to_check = list(self.instances)
                
                for instance in instances_to_check:
                    self._check_instance_health(instance)
                
                time.sleep(self.health_check_interval)
                
            except Exception as e:
                logging.error(f"Health check error: {e}")
                time.sleep(5.0)
    
    def _check_instance_health(self, instance: Dict[str, Any]):
        """Check health of a specific instance."""
        instance_id = instance["id"]
        endpoint = instance["endpoint"]
        
        try:
            # Simple HTTP health check (would use proper health endpoint)
            import requests
            response = requests.get(f"{endpoint}/health", timeout=5.0)
            
            if response.status_code == 200:
                # Mark as healthy
                health = self.instance_health[instance_id]
                health["healthy"] = True
                health["consecutive_failures"] = 0
                health["last_check"] = time.time()
            else:
                self.mark_instance_unhealthy(instance_id)
                
        except Exception as e:
            logging.warning(f"Health check failed for {instance_id}: {e}")
            self.mark_instance_unhealthy(instance_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get load balancer statistics."""
        with self.lock:
            total_instances = len(self.instances)
            healthy_instances = sum(
                1 for inst_id in self.instances
                if self.instance_health.get(inst_id, {}).get("healthy", False)
            )
            
            total_requests = sum(inst["request_count"] for inst in self.instances)
            
            return {
                "total_instances": total_instances,
                "healthy_instances": healthy_instances,
                "total_requests": total_requests,
                "instance_details": [
                    {
                        "id": inst["id"],
                        "endpoint": inst["endpoint"],
                        "active": inst["active"],
                        "request_count": inst["request_count"],
                        "healthy": self.instance_health.get(inst["id"], {}).get("healthy", False),
                    }
                    for inst in self.instances
                ]
            }


class AutoScaler:
    """Automatic scaling controller based on system metrics."""
    
    def __init__(self, 
                 min_instances: int = 1,
                 max_instances: int = 10,
                 target_cpu: float = 70.0,
                 target_memory: float = 80.0,
                 scale_up_threshold: float = 80.0,
                 scale_down_threshold: float = 50.0,
                 cooldown_seconds: float = 300.0):
        
        self.min_instances = min_instances
        self.max_instances = max_instances
        self.target_cpu = target_cpu
        self.target_memory = target_memory
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.cooldown_seconds = cooldown_seconds
        
        # Current state
        self.current_instances = min_instances
        self.last_scaling_event = 0
        self.metrics_history = deque(maxlen=100)
        self.scaling_events = deque(maxlen=50)
        
        # Monitoring
        self.monitoring_active = False
        self.monitoring_thread = None
        self.lock = threading.Lock()
        
        # Scaling callbacks
        self.scale_up_callback: Optional[Callable[[int], bool]] = None
        self.scale_down_callback: Optional[Callable[[int], bool]] = None
        
        logging.info(f"AutoScaler initialized: {min_instances}-{max_instances} instances")
    
    def set_scale_callbacks(self, 
                           scale_up_callback: Callable[[int], bool],
                           scale_down_callback: Callable[[int], bool]):
        """Set callbacks for scaling operations."""
        self.scale_up_callback = scale_up_callback
        self.scale_down_callback = scale_down_callback
    
    def record_metrics(self, metrics: MetricSnapshot):
        """Record system metrics for scaling decisions."""
        with self.lock:
            self.metrics_history.append(metrics)
            
            # Check if scaling is needed
            if self._should_scale():
                self._perform_scaling(metrics)
    
    def _should_scale(self) -> bool:
        """Check if scaling action should be taken."""
        # Check cooldown period
        if time.time() - self.last_scaling_event < self.cooldown_seconds:
            return False
        
        # Need sufficient metrics
        if len(self.metrics_history) < 3:
            return False
        
        # Check recent metrics trend
        recent_metrics = list(self.metrics_history)[-3:]
        
        # Average metrics over recent period
        avg_cpu = sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics)
        avg_memory = sum(m.memory_percent for m in recent_metrics) / len(recent_metrics)
        avg_response_time = sum(m.response_time_ms for m in recent_metrics) / len(recent_metrics)
        
        # Scale up conditions
        if (avg_cpu > self.scale_up_threshold or 
            avg_memory > self.scale_up_threshold or
            avg_response_time > 5000):  # 5 second response time threshold
            return self.current_instances < self.max_instances
        
        # Scale down conditions
        if (avg_cpu < self.scale_down_threshold and 
            avg_memory < self.scale_down_threshold and
            avg_response_time < 1000):  # 1 second response time
            return self.current_instances > self.min_instances
        
        return False
    
    def _perform_scaling(self, current_metrics: MetricSnapshot):
        """Perform scaling action based on current metrics."""
        recent_metrics = list(self.metrics_history)[-3:]
        avg_cpu = sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics)
        avg_memory = sum(m.memory_percent for m in recent_metrics) / len(recent_metrics)
        avg_response_time = sum(m.response_time_ms for m in recent_metrics) / len(recent_metrics)
        
        action = None
        target_instances = self.current_instances
        trigger_metric = ""
        metric_value = 0
        threshold = 0
        
        # Determine scaling action
        if (avg_cpu > self.scale_up_threshold or 
            avg_memory > self.scale_up_threshold or
            avg_response_time > 5000):
            
            action = "scale_up"
            target_instances = min(self.current_instances + 1, self.max_instances)
            
            if avg_cpu > self.scale_up_threshold:
                trigger_metric = "cpu"
                metric_value = avg_cpu
                threshold = self.scale_up_threshold
            elif avg_memory > self.scale_up_threshold:
                trigger_metric = "memory"
                metric_value = avg_memory
                threshold = self.scale_up_threshold
            else:
                trigger_metric = "response_time"
                metric_value = avg_response_time
                threshold = 5000
        
        elif (avg_cpu < self.scale_down_threshold and 
              avg_memory < self.scale_down_threshold and
              avg_response_time < 1000):
            
            action = "scale_down"
            target_instances = max(self.current_instances - 1, self.min_instances)
            trigger_metric = "cpu_memory"
            metric_value = max(avg_cpu, avg_memory)
            threshold = self.scale_down_threshold
        
        if action and target_instances != self.current_instances:
            success = self._execute_scaling(action, target_instances)
            
            if success:
                # Record scaling event
                event = ScalingEvent(
                    timestamp=time.time(),
                    action=action,
                    current_instances=self.current_instances,
                    target_instances=target_instances,
                    trigger_metric=trigger_metric,
                    metric_value=metric_value,
                    threshold=threshold,
                    reason=f"{trigger_metric} {metric_value:.1f} {'>' if action == 'scale_up' else '<'} {threshold}"
                )
                
                self.scaling_events.append(event)
                self.current_instances = target_instances
                self.last_scaling_event = time.time()
                
                logging.info(f"Scaling {action}: {event.current_instances} -> {target_instances} "
                           f"(trigger: {event.reason})")
    
    def _execute_scaling(self, action: str, target_instances: int) -> bool:
        """Execute the scaling action."""
        try:
            if action == "scale_up" and self.scale_up_callback:
                return self.scale_up_callback(target_instances)
            elif action == "scale_down" and self.scale_down_callback:
                return self.scale_down_callback(target_instances)
            else:
                logging.warning(f"No callback configured for {action}")
                return False
        except Exception as e:
            logging.error(f"Scaling execution failed: {e}")
            return False
    
    def start_monitoring(self, interval_seconds: float = 30.0):
        """Start automated monitoring and scaling."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval_seconds,),
            daemon=True
        )
        self.monitoring_thread.start()
        
        logging.info(f"AutoScaler monitoring started (interval: {interval_seconds}s)")
    
    def stop_monitoring(self):
        """Stop automated monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        
        logging.info("AutoScaler monitoring stopped")
    
    def _monitoring_loop(self, interval_seconds: float):
        """Background monitoring loop."""
        while self.monitoring_active:
            try:
                # Collect current metrics
                metrics = self._collect_system_metrics()
                self.record_metrics(metrics)
                
                time.sleep(interval_seconds)
                
            except Exception as e:
                logging.error(f"Monitoring loop error: {e}")
                time.sleep(10.0)
    
    def _collect_system_metrics(self) -> MetricSnapshot:
        """Collect current system metrics."""
        # Get basic system metrics
        cpu_percent = psutil.cpu_percent(interval=1.0)
        memory = psutil.virtual_memory()
        
        # Placeholder for application-specific metrics
        # In production, these would come from your monitoring system
        active_requests = 0  # Would track from request queue
        response_time_ms = 100.0  # Would track from performance monitor
        queue_length = 0  # Would track from request processing queue
        error_rate = 0.0  # Would calculate from error counts
        throughput_rps = 10.0  # Would calculate from request counts
        
        return MetricSnapshot(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            active_requests=active_requests,
            response_time_ms=response_time_ms,
            queue_length=queue_length,
            error_rate=error_rate,
            throughput_rps=throughput_rps,
        )
    
    def get_scaling_stats(self) -> Dict[str, Any]:
        """Get scaling statistics and current state."""
        with self.lock:
            recent_events = list(self.scaling_events)[-10:]  # Last 10 events
            
            return {
                "current_instances": self.current_instances,
                "min_instances": self.min_instances,
                "max_instances": self.max_instances,
                "last_scaling_event": self.last_scaling_event,
                "total_scaling_events": len(self.scaling_events),
                "recent_events": [asdict(event) for event in recent_events],
                "thresholds": {
                    "scale_up": self.scale_up_threshold,
                    "scale_down": self.scale_down_threshold,
                    "cooldown_seconds": self.cooldown_seconds,
                },
                "monitoring_active": self.monitoring_active,
            }


class DockerScaler:
    """Docker-based container scaling implementation."""
    
    def __init__(self, image_name: str, container_prefix: str = "olfactory"):
        if not HAS_DOCKER:
            raise RuntimeError("Docker not available")
        
        self.client = docker.from_env()
        self.image_name = image_name
        self.container_prefix = container_prefix
        self.running_containers = {}
        
        logging.info(f"Docker scaler initialized for image: {image_name}")
    
    def scale_up(self, target_instances: int) -> bool:
        """Scale up Docker containers."""
        try:
            current_count = len(self.running_containers)
            
            for i in range(current_count, target_instances):
                container_name = f"{self.container_prefix}-{i}"
                
                container = self.client.containers.run(
                    self.image_name,
                    name=container_name,
                    detach=True,
                    ports={'8000/tcp': None},  # Dynamic port mapping
                    environment={
                        'INSTANCE_ID': str(i),
                        'LOG_LEVEL': 'INFO',
                    },
                    restart_policy={"Name": "unless-stopped"},
                )
                
                self.running_containers[container_name] = container
                logging.info(f"Started container: {container_name}")
            
            return True
            
        except Exception as e:
            logging.error(f"Failed to scale up containers: {e}")
            return False
    
    def scale_down(self, target_instances: int) -> bool:
        """Scale down Docker containers."""
        try:
            current_count = len(self.running_containers)
            containers_to_remove = current_count - target_instances
            
            # Remove excess containers (LIFO)
            container_items = list(self.running_containers.items())
            for i in range(containers_to_remove):
                container_name, container = container_items[-(i+1)]
                
                container.stop(timeout=30)
                container.remove()
                del self.running_containers[container_name]
                
                logging.info(f"Stopped and removed container: {container_name}")
            
            return True
            
        except Exception as e:
            logging.error(f"Failed to scale down containers: {e}")
            return False
    
    def get_container_stats(self) -> Dict[str, Any]:
        """Get statistics for running containers."""
        stats = {}
        
        for container_name, container in self.running_containers.items():
            try:
                container.reload()
                stats[container_name] = {
                    "status": container.status,
                    "ports": container.ports,
                    "created": container.attrs['Created'],
                }
            except Exception as e:
                logging.warning(f"Failed to get stats for {container_name}: {e}")
                stats[container_name] = {"status": "unknown", "error": str(e)}
        
        return stats


class MetricsCollector:
    """Collect and aggregate metrics from multiple sources."""
    
    def __init__(self):
        self.metrics_buffer = deque(maxlen=1000)
        self.aggregated_metrics = defaultdict(list)
        self.lock = threading.Lock()
        
        # Collection settings
        self.collection_interval = 10.0  # seconds
        self.aggregation_window = 60.0  # seconds
        
        logging.info("Metrics collector initialized")
    
    def add_metric(self, metric_name: str, value: float, timestamp: Optional[float] = None):
        """Add a metric data point."""
        if timestamp is None:
            timestamp = time.time()
        
        with self.lock:
            metric_point = {
                "name": metric_name,
                "value": value,
                "timestamp": timestamp,
            }
            
            self.metrics_buffer.append(metric_point)
            self.aggregated_metrics[metric_name].append((timestamp, value))
            
            # Clean old data
            cutoff_time = timestamp - self.aggregation_window
            self.aggregated_metrics[metric_name] = [
                (t, v) for t, v in self.aggregated_metrics[metric_name]
                if t > cutoff_time
            ]
    
    def get_aggregated_metrics(self, window_seconds: float = 60.0) -> Dict[str, Dict[str, float]]:
        """Get aggregated metrics over time window."""
        cutoff_time = time.time() - window_seconds
        aggregated = {}
        
        with self.lock:
            for metric_name, data_points in self.aggregated_metrics.items():
                recent_points = [(t, v) for t, v in data_points if t > cutoff_time]
                
                if recent_points:
                    values = [v for t, v in recent_points]
                    aggregated[metric_name] = {
                        "count": len(values),
                        "mean": sum(values) / len(values),
                        "min": min(values),
                        "max": max(values),
                        "latest": values[-1],
                    }
        
        return aggregated
    
    def export_metrics(self, file_path: Path):
        """Export metrics to file."""
        with self.lock:
            metrics_data = {
                "export_timestamp": time.time(),
                "metrics_buffer": list(self.metrics_buffer),
                "aggregated_metrics": {
                    name: list(data) for name, data in self.aggregated_metrics.items()
                }
            }
        
        with open(file_path, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        
        logging.info(f"Metrics exported to {file_path}")


# Global instances
load_balancer = LoadBalancer()
metrics_collector = MetricsCollector()


def setup_auto_scaling(image_name: str, 
                      min_instances: int = 1,
                      max_instances: int = 5) -> AutoScaler:
    """Setup complete auto-scaling system."""
    
    # Create Docker scaler
    docker_scaler = DockerScaler(image_name)
    
    # Create auto-scaler
    auto_scaler = AutoScaler(
        min_instances=min_instances,
        max_instances=max_instances
    )
    
    # Set scaling callbacks
    auto_scaler.set_scale_callbacks(
        docker_scaler.scale_up,
        docker_scaler.scale_down
    )
    
    # Start monitoring
    auto_scaler.start_monitoring()
    load_balancer.start_health_checks()
    
    logging.info("Auto-scaling system setup completed")
    return auto_scaler