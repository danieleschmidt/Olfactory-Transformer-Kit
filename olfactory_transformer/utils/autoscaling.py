"""Auto-scaling and load balancing for production deployments."""

import time
import logging
import threading
import json
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, field
from collections import deque, defaultdict
from enum import Enum
import multiprocessing as mp
import asyncio
from concurrent.futures import ThreadPoolExecutor
import hashlib
import uuid

import torch
import psutil


class ScalingEvent(Enum):
    """Auto-scaling events."""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    INSTANCE_START = "instance_start"
    INSTANCE_STOP = "instance_stop"
    HEALTH_CHECK = "health_check"


@dataclass
class InstanceMetrics:
    """Metrics for a single inference instance."""
    instance_id: str
    timestamp: float = field(default_factory=time.time)
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    gpu_utilization: float = 0.0
    active_requests: int = 0
    requests_per_second: float = 0.0
    average_latency_ms: float = 0.0
    error_rate: float = 0.0
    queue_depth: int = 0
    health_status: str = "healthy"  # healthy, degraded, unhealthy


@dataclass
class ScalingPolicy:
    """Configuration for auto-scaling behavior."""
    min_instances: int = 1
    max_instances: int = 10
    target_cpu_utilization: float = 70.0
    target_memory_utilization: float = 80.0
    target_latency_ms: float = 100.0
    scale_up_threshold: float = 85.0
    scale_down_threshold: float = 30.0
    scale_up_cooldown_seconds: float = 300.0  # 5 minutes
    scale_down_cooldown_seconds: float = 600.0  # 10 minutes
    evaluation_window_seconds: float = 300.0  # 5 minutes
    warmup_time_seconds: float = 60.0  # 1 minute
    health_check_interval: float = 30.0
    load_balancing_strategy: str = "round_robin"  # round_robin, least_connections, least_latency


class InferenceInstance:
    """Single inference instance for the olfactory transformer."""
    
    def __init__(self, instance_id: str, model_config: Dict[str, Any]):
        self.instance_id = instance_id
        self.model_config = model_config
        self.model = None
        self.tokenizer = None
        
        # Instance state
        self.status = "initializing"  # initializing, ready, busy, stopping, stopped
        self.start_time = time.time()
        self.request_queue = asyncio.Queue()
        self.active_requests = 0
        
        # Performance tracking
        self.request_history = deque(maxlen=1000)
        self.latency_history = deque(maxlen=100)
        self.error_count = 0
        self.total_requests = 0
        
        # Health monitoring
        self.last_health_check = 0
        self.consecutive_failures = 0
        
        logging.info(f"Inference instance {instance_id} created")
    
    async def initialize(self):
        """Initialize the model and tokenizer."""
        try:
            # Import at runtime to avoid circular imports
            from ..core.model import OlfactoryTransformer
            from ..core.tokenizer import MoleculeTokenizer
            from ..core.config import OlfactoryConfig
            
            # Create model
            config = OlfactoryConfig(**self.model_config)
            self.model = OlfactoryTransformer(config)
            self.tokenizer = MoleculeTokenizer(vocab_size=config.vocab_size)
            
            # Build basic vocabulary
            sample_molecules = ['CCO', 'CC(C)O', 'CCC', 'CCCC', 'C1=CC=CC=C1']
            self.tokenizer.build_vocab_from_smiles(sample_molecules)
            
            self.status = "ready"
            logging.info(f"Instance {self.instance_id} initialized successfully")
            
        except Exception as e:
            self.status = "failed"
            logging.error(f"Failed to initialize instance {self.instance_id}: {e}")
            raise
    
    async def predict(self, smiles: str, request_id: str) -> Dict[str, Any]:
        """Process prediction request."""
        start_time = time.time()
        
        try:
            self.active_requests += 1
            self.total_requests += 1
            
            # Check if model is ready
            if self.status != "ready":
                raise RuntimeError(f"Instance {self.instance_id} not ready (status: {self.status})")
            
            # Make prediction
            prediction = self.model.predict_scent(smiles, self.tokenizer)
            
            # Record successful request
            latency_ms = (time.time() - start_time) * 1000
            self.latency_history.append(latency_ms)
            self.request_history.append({
                "request_id": request_id,
                "timestamp": time.time(),
                "latency_ms": latency_ms,
                "success": True
            })
            
            return {
                "success": True,
                "prediction": prediction.__dict__,
                "instance_id": self.instance_id,
                "latency_ms": latency_ms
            }
            
        except Exception as e:
            # Record failed request
            self.error_count += 1
            latency_ms = (time.time() - start_time) * 1000
            self.request_history.append({
                "request_id": request_id,
                "timestamp": time.time(),
                "latency_ms": latency_ms,
                "success": False,
                "error": str(e)
            })
            
            logging.error(f"Prediction failed in instance {self.instance_id}: {e}")
            return {
                "success": False,
                "error": str(e),
                "instance_id": self.instance_id,
                "latency_ms": latency_ms
            }
            
        finally:
            self.active_requests -= 1
    
    def get_metrics(self) -> InstanceMetrics:
        """Get current instance metrics."""
        current_time = time.time()
        
        # Calculate request rate (requests in last minute)
        recent_requests = [r for r in self.request_history 
                          if current_time - r["timestamp"] <= 60]
        requests_per_second = len(recent_requests) / 60.0
        
        # Calculate average latency
        avg_latency = np.mean(self.latency_history) if self.latency_history else 0.0
        
        # Calculate error rate
        recent_errors = sum(1 for r in recent_requests if not r["success"])
        error_rate = recent_errors / max(1, len(recent_requests))
        
        return InstanceMetrics(
            instance_id=self.instance_id,
            timestamp=current_time,
            cpu_percent=psutil.cpu_percent(),
            memory_percent=psutil.virtual_memory().percent,
            active_requests=self.active_requests,
            requests_per_second=requests_per_second,
            average_latency_ms=avg_latency,
            error_rate=error_rate,
            queue_depth=self.request_queue.qsize(),
            health_status=self._determine_health_status()
        )
    
    def _determine_health_status(self) -> str:
        """Determine instance health status."""
        if self.status != "ready":
            return "unhealthy"
        
        if self.consecutive_failures > 3:
            return "unhealthy"
        
        if self.error_count / max(1, self.total_requests) > 0.1:  # >10% error rate
            return "degraded"
        
        if self.active_requests > 10:  # High load
            return "degraded"
        
        return "healthy"
    
    async def stop(self):
        """Stop the instance gracefully."""
        self.status = "stopping"
        
        # Wait for active requests to complete
        while self.active_requests > 0:
            await asyncio.sleep(0.1)
        
        self.status = "stopped"
        logging.info(f"Instance {self.instance_id} stopped")


class LoadBalancer:
    """Load balancer for distributing requests across instances."""
    
    def __init__(self, strategy: str = "round_robin"):
        self.strategy = strategy
        self.instances: Dict[str, InferenceInstance] = {}
        self.current_index = 0
        self.lock = threading.Lock()
        
        logging.info(f"Load balancer initialized with {strategy} strategy")
    
    def add_instance(self, instance: InferenceInstance):
        """Add instance to load balancer."""
        with self.lock:
            self.instances[instance.instance_id] = instance
        logging.info(f"Added instance {instance.instance_id} to load balancer")
    
    def remove_instance(self, instance_id: str):
        """Remove instance from load balancer."""
        with self.lock:
            if instance_id in self.instances:
                del self.instances[instance_id]
        logging.info(f"Removed instance {instance_id} from load balancer")
    
    def select_instance(self) -> Optional[InferenceInstance]:
        """Select best instance based on load balancing strategy."""
        with self.lock:
            available_instances = [
                instance for instance in self.instances.values()
                if instance.status == "ready" and instance._determine_health_status() != "unhealthy"
            ]
            
            if not available_instances:
                return None
            
            if self.strategy == "round_robin":
                # Round-robin selection
                instance = available_instances[self.current_index % len(available_instances)]
                self.current_index += 1
                return instance
                
            elif self.strategy == "least_connections":
                # Select instance with fewest active requests
                return min(available_instances, key=lambda x: x.active_requests)
                
            elif self.strategy == "least_latency":
                # Select instance with lowest average latency
                def avg_latency(instance):
                    return np.mean(instance.latency_history) if instance.latency_history else 0
                
                return min(available_instances, key=avg_latency)
            
            else:
                # Default to round-robin
                return available_instances[0]
    
    def get_all_metrics(self) -> List[InstanceMetrics]:
        """Get metrics from all instances."""
        with self.lock:
            return [instance.get_metrics() for instance in self.instances.values()]


class AutoScaler:
    """Auto-scaling controller for managing inference instances."""
    
    def __init__(self, policy: ScalingPolicy):
        self.policy = policy
        self.load_balancer = LoadBalancer(policy.load_balancing_strategy)
        self.instances: Dict[str, InferenceInstance] = {}
        
        # Scaling state
        self.last_scale_up_time = 0
        self.last_scale_down_time = 0
        self.target_instances = policy.min_instances
        
        # Monitoring
        self.metrics_history = deque(maxlen=1000)
        self.scaling_events = deque(maxlen=100)
        
        # Control loop
        self.should_stop = threading.Event()
        self.control_thread = None
        
        logging.info("Auto-scaler initialized")
    
    def start(self, model_config: Dict[str, Any]):
        """Start auto-scaling with initial instances."""
        self.model_config = model_config
        
        # Create initial instances
        for i in range(self.policy.min_instances):
            self._create_instance()
        
        # Start control loop
        self.control_thread = threading.Thread(target=self._control_loop, daemon=True)
        self.control_thread.start()
        
        logging.info(f"Auto-scaler started with {self.policy.min_instances} initial instances")
    
    def stop(self):
        """Stop auto-scaler and all instances."""
        self.should_stop.set()
        
        if self.control_thread:
            self.control_thread.join(timeout=10.0)
        
        # Stop all instances
        for instance in self.instances.values():
            asyncio.create_task(instance.stop())
        
        logging.info("Auto-scaler stopped")
    
    def _create_instance(self) -> str:
        """Create new inference instance."""
        instance_id = f"instance_{uuid.uuid4().hex[:8]}"
        instance = InferenceInstance(instance_id, self.model_config)
        
        # Initialize instance asynchronously
        asyncio.create_task(self._initialize_instance(instance))
        
        self.instances[instance_id] = instance
        self.load_balancer.add_instance(instance)
        
        # Record scaling event
        self.scaling_events.append({
            "timestamp": time.time(),
            "event": ScalingEvent.INSTANCE_START.value,
            "instance_id": instance_id,
            "total_instances": len(self.instances)
        })
        
        logging.info(f"Created instance {instance_id}")
        return instance_id
    
    async def _initialize_instance(self, instance: InferenceInstance):
        """Initialize instance asynchronously."""
        try:
            await instance.initialize()
        except Exception as e:
            logging.error(f"Failed to initialize instance {instance.instance_id}: {e}")
            # Remove failed instance
            self._remove_instance(instance.instance_id)
    
    def _remove_instance(self, instance_id: str):
        """Remove and stop instance."""
        if instance_id in self.instances:
            instance = self.instances[instance_id]
            
            # Stop instance
            asyncio.create_task(instance.stop())
            
            # Remove from load balancer and instances
            self.load_balancer.remove_instance(instance_id)
            del self.instances[instance_id]
            
            # Record scaling event
            self.scaling_events.append({
                "timestamp": time.time(),
                "event": ScalingEvent.INSTANCE_STOP.value,
                "instance_id": instance_id,
                "total_instances": len(self.instances)
            })
            
            logging.info(f"Removed instance {instance_id}")
    
    def _control_loop(self):
        """Main control loop for auto-scaling decisions."""
        while not self.should_stop.wait(self.policy.health_check_interval):
            try:
                # Collect metrics
                all_metrics = self.load_balancer.get_all_metrics()
                if not all_metrics:
                    continue
                
                # Aggregate metrics
                aggregated = self._aggregate_metrics(all_metrics)
                self.metrics_history.append(aggregated)
                
                # Make scaling decision
                self._evaluate_scaling(aggregated)
                
            except Exception as e:
                logging.error(f"Error in auto-scaler control loop: {e}")
    
    def _aggregate_metrics(self, metrics: List[InstanceMetrics]) -> Dict[str, float]:
        """Aggregate metrics across all instances."""
        if not metrics:
            return {}
        
        healthy_instances = [m for m in metrics if m.health_status == "healthy"]
        
        return {
            "timestamp": time.time(),
            "total_instances": len(metrics),
            "healthy_instances": len(healthy_instances),
            "avg_cpu_percent": np.mean([m.cpu_percent for m in healthy_instances]) if healthy_instances else 0,
            "avg_memory_percent": np.mean([m.memory_percent for m in healthy_instances]) if healthy_instances else 0,
            "total_active_requests": sum(m.active_requests for m in metrics),
            "avg_latency_ms": np.mean([m.average_latency_ms for m in healthy_instances if m.average_latency_ms > 0]) or 0,
            "total_rps": sum(m.requests_per_second for m in metrics),
            "avg_error_rate": np.mean([m.error_rate for m in healthy_instances]) if healthy_instances else 0,
            "total_queue_depth": sum(m.queue_depth for m in metrics),
        }
    
    def _evaluate_scaling(self, metrics: Dict[str, float]):
        """Evaluate if scaling action is needed."""
        current_time = time.time()
        current_instances = metrics.get("total_instances", 0)
        healthy_instances = metrics.get("healthy_instances", 0)
        
        # Check if we have enough recent metrics for decision
        recent_metrics = [m for m in self.metrics_history 
                         if current_time - m.get("timestamp", 0) <= self.policy.evaluation_window_seconds]
        
        if len(recent_metrics) < 3:  # Need at least 3 data points
            return
        
        # Calculate average metrics over evaluation window
        avg_cpu = np.mean([m.get("avg_cpu_percent", 0) for m in recent_metrics])
        avg_memory = np.mean([m.get("avg_memory_percent", 0) for m in recent_metrics])
        avg_latency = np.mean([m.get("avg_latency_ms", 0) for m in recent_metrics])
        avg_error_rate = np.mean([m.get("avg_error_rate", 0) for m in recent_metrics])
        
        # Scale up conditions
        scale_up_needed = (
            (avg_cpu > self.policy.scale_up_threshold or
             avg_memory > self.policy.scale_up_threshold or
             avg_latency > self.policy.target_latency_ms * 2.0 or
             avg_error_rate > 0.05) and  # 5% error rate threshold
            current_instances < self.policy.max_instances and
            healthy_instances < current_instances and  # Have unhealthy instances
            current_time - self.last_scale_up_time > self.policy.scale_up_cooldown_seconds
        )
        
        # Scale down conditions  
        scale_down_needed = (
            avg_cpu < self.policy.scale_down_threshold and
            avg_memory < self.policy.scale_down_threshold and
            avg_latency < self.policy.target_latency_ms * 0.5 and
            avg_error_rate < 0.01 and  # 1% error rate threshold
            current_instances > self.policy.min_instances and
            current_time - self.last_scale_down_time > self.policy.scale_down_cooldown_seconds
        )
        
        if scale_up_needed:
            self._scale_up()
        elif scale_down_needed:
            self._scale_down()
    
    def _scale_up(self):
        """Scale up by adding instance."""
        self._create_instance()
        self.last_scale_up_time = time.time()
        
        # Record scaling event
        self.scaling_events.append({
            "timestamp": time.time(),
            "event": ScalingEvent.SCALE_UP.value,
            "total_instances": len(self.instances),
            "reason": "High resource utilization or latency"
        })
        
        logging.info(f"Scaled up to {len(self.instances)} instances")
    
    def _scale_down(self):
        """Scale down by removing instance."""
        # Find least utilized instance
        metrics = self.load_balancer.get_all_metrics()
        if not metrics or len(self.instances) <= self.policy.min_instances:
            return
        
        # Remove instance with lowest utilization
        least_utilized = min(metrics, key=lambda m: m.active_requests + m.queue_depth)
        self._remove_instance(least_utilized.instance_id)
        self.last_scale_down_time = time.time()
        
        # Record scaling event
        self.scaling_events.append({
            "timestamp": time.time(),
            "event": ScalingEvent.SCALE_DOWN.value,
            "total_instances": len(self.instances),
            "reason": "Low resource utilization"
        })
        
        logging.info(f"Scaled down to {len(self.instances)} instances")
    
    async def predict(self, smiles: str, request_id: Optional[str] = None) -> Dict[str, Any]:
        """Make prediction using load-balanced instances."""
        if request_id is None:
            request_id = str(uuid.uuid4())
        
        # Select instance
        instance = self.load_balancer.select_instance()
        if not instance:
            return {
                "success": False,
                "error": "No healthy instances available",
                "request_id": request_id
            }
        
        # Make prediction
        return await instance.predict(smiles, request_id)
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive auto-scaler status."""
        all_metrics = self.load_balancer.get_all_metrics()
        
        return {
            "timestamp": time.time(),
            "policy": {
                "min_instances": self.policy.min_instances,
                "max_instances": self.policy.max_instances,
                "target_cpu": self.policy.target_cpu_utilization,
                "target_latency_ms": self.policy.target_latency_ms,
            },
            "current_state": {
                "total_instances": len(self.instances),
                "healthy_instances": sum(1 for m in all_metrics if m.health_status == "healthy"),
                "active_requests": sum(m.active_requests for m in all_metrics),
                "total_rps": sum(m.requests_per_second for m in all_metrics),
            },
            "recent_events": list(self.scaling_events)[-10:],
            "instance_metrics": [m.__dict__ for m in all_metrics],
        }


# Factory function for creating auto-scaler
def create_autoscaler(
    min_instances: int = 1,
    max_instances: int = 5,
    model_config: Optional[Dict[str, Any]] = None
) -> AutoScaler:
    """Create auto-scaler with default configuration."""
    policy = ScalingPolicy(
        min_instances=min_instances,
        max_instances=max_instances,
        target_cpu_utilization=70.0,
        target_latency_ms=100.0,
    )
    
    scaler = AutoScaler(policy)
    
    if model_config:
        scaler.start(model_config)
    
    return scaler