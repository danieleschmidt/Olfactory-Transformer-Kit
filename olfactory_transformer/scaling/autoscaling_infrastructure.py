"""
Auto-scaling Infrastructure for Generation 3 Production Deployment.

Intelligent auto-scaling system that dynamically adjusts resources based on:
- Real-time load monitoring
- Prediction request patterns
- Performance metrics and thresholds
- Cost optimization strategies
- Global deployment coordination
"""

import time
import logging
import threading
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    # Mock psutil for basic functionality
    class MockPsutil:
        @staticmethod
        def cpu_percent():
            return 45.0
        
        @staticmethod
        def virtual_memory():
            return type('MockMemory', (), {
                'used': 4 * 1024 * 1024 * 1024,  # 4GB
                'percent': 60.0,
                'available': 2 * 1024 * 1024 * 1024  # 2GB
            })()
        
        @staticmethod
        def disk_usage(path):
            return type('MockDisk', (), {'percent': 45.0})()
        
        @staticmethod
        def net_connections():
            return [None] * 20  # Mock 20 connections
        
        @staticmethod
        def pids():
            return list(range(100))  # Mock 100 processes
        
        @staticmethod
        def cpu_count():
            return 4
    
    psutil = MockPsutil()
from collections import deque, defaultdict
import statistics

try:
    from olfactory_transformer.utils.dependency_manager import dependency_manager
    np = dependency_manager.mock_implementations.get('numpy')
    if np is None:
        import numpy as np
except ImportError:
    import numpy as np


class ScalingDecision(Enum):
    """Scaling decision types."""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    MAINTAIN = "maintain"
    EMERGENCY_SCALE = "emergency_scale"


class ResourceType(Enum):
    """Resource types for scaling."""
    CPU = "cpu"
    MEMORY = "memory"
    GPU = "gpu"
    NETWORK = "network"
    STORAGE = "storage"


@dataclass
class ScalingMetrics:
    """System metrics for scaling decisions."""
    timestamp: float
    cpu_utilization: float
    memory_utilization: float
    request_rate: float
    response_time: float
    error_rate: float
    queue_size: int
    active_connections: int
    throughput: float
    resource_usage: Dict[ResourceType, float] = field(default_factory=dict)


@dataclass
class ScalingConfiguration:
    """Auto-scaling configuration parameters."""
    min_instances: int = 1
    max_instances: int = 10
    target_cpu_utilization: float = 70.0
    target_memory_utilization: float = 80.0
    target_response_time_ms: float = 200.0
    scale_up_threshold: float = 80.0
    scale_down_threshold: float = 30.0
    cooldown_period_sec: int = 300
    evaluation_period_sec: int = 60
    emergency_threshold: float = 95.0
    cost_optimization_enabled: bool = True


@dataclass
class ScalingEvent:
    """Scaling event record."""
    timestamp: float
    decision: ScalingDecision
    current_instances: int
    target_instances: int
    trigger_metrics: ScalingMetrics
    reason: str
    success: bool
    execution_time: float


class MetricsCollector:
    """Real-time metrics collection and aggregation."""
    
    def __init__(self, collection_interval: float = 5.0):
        self.collection_interval = collection_interval
        self.metrics_history = deque(maxlen=1000)  # Keep last 1000 metrics
        self.collecting = False
        self.request_counter = 0
        self.response_times = deque(maxlen=100)
        self.error_counter = 0
        
    def start_collection(self):
        """Start metrics collection in background thread."""
        self.collecting = True
        self.collection_thread = threading.Thread(target=self._collection_loop)
        self.collection_thread.daemon = True
        self.collection_thread.start()
        logging.info("Metrics collection started")
    
    def stop_collection(self):
        """Stop metrics collection."""
        self.collecting = False
        if hasattr(self, 'collection_thread'):
            self.collection_thread.join(timeout=2.0)
        logging.info("Metrics collection stopped")
    
    def _collection_loop(self):
        """Background metrics collection loop."""
        while self.collecting:
            try:
                metrics = self._collect_system_metrics()
                self.metrics_history.append(metrics)
                time.sleep(self.collection_interval)
            except Exception as e:
                logging.warning(f"Metrics collection error: {e}")
                time.sleep(self.collection_interval)
    
    def _collect_system_metrics(self) -> ScalingMetrics:
        """Collect current system metrics."""
        # System resource utilization
        cpu_percent = psutil.cpu_percent()
        memory_info = psutil.virtual_memory()
        memory_percent = memory_info.percent
        
        # Application-specific metrics
        current_time = time.time()
        time_window = 60  # 1 minute window
        
        # Calculate request rate (requests per second)
        recent_requests = sum(1 for t in self.response_times 
                            if current_time - t < time_window) if self.response_times else 0
        request_rate = recent_requests / time_window
        
        # Calculate average response time
        recent_response_times = [rt for rt in self.response_times 
                               if current_time - rt < time_window]
        avg_response_time = statistics.mean(recent_response_times) if recent_response_times else 0
        
        # Error rate calculation
        total_requests = len(self.response_times) if self.response_times else 1
        error_rate = self.error_counter / total_requests
        
        # Mock queue size and connections (would be real in production)
        queue_size = max(0, int(request_rate * 2 - 10))  # Simulated queue
        active_connections = min(100, int(request_rate * 5))  # Simulated connections
        
        return ScalingMetrics(
            timestamp=current_time,
            cpu_utilization=cpu_percent,
            memory_utilization=memory_percent,
            request_rate=request_rate,
            response_time=avg_response_time,
            error_rate=error_rate,
            queue_size=queue_size,
            active_connections=active_connections,
            throughput=request_rate,  # Simplified throughput
            resource_usage={
                ResourceType.CPU: cpu_percent,
                ResourceType.MEMORY: memory_percent,
                ResourceType.NETWORK: min(100, request_rate * 10),  # Simulated
                ResourceType.STORAGE: 20  # Simulated storage usage
            }
        )
    
    def record_request(self, response_time: float, error: bool = False):
        """Record a request for metrics."""
        current_time = time.time()
        self.response_times.append(current_time)
        self.request_counter += 1
        if error:
            self.error_counter += 1
    
    def get_current_metrics(self) -> Optional[ScalingMetrics]:
        """Get most recent metrics."""
        return self.metrics_history[-1] if self.metrics_history else None
    
    def get_metrics_trend(self, duration_sec: int = 300) -> List[ScalingMetrics]:
        """Get metrics trend over specified duration."""
        current_time = time.time()
        cutoff_time = current_time - duration_sec
        
        return [m for m in self.metrics_history if m.timestamp >= cutoff_time]
    
    def get_aggregated_metrics(self, duration_sec: int = 300) -> Optional[ScalingMetrics]:
        """Get aggregated metrics over specified duration."""
        trend_metrics = self.get_metrics_trend(duration_sec)
        
        if not trend_metrics:
            return None
        
        # Aggregate metrics
        return ScalingMetrics(
            timestamp=time.time(),
            cpu_utilization=statistics.mean(m.cpu_utilization for m in trend_metrics),
            memory_utilization=statistics.mean(m.memory_utilization for m in trend_metrics),
            request_rate=statistics.mean(m.request_rate for m in trend_metrics),
            response_time=statistics.mean(m.response_time for m in trend_metrics),
            error_rate=statistics.mean(m.error_rate for m in trend_metrics),
            queue_size=int(statistics.mean(m.queue_size for m in trend_metrics)),
            active_connections=int(statistics.mean(m.active_connections for m in trend_metrics)),
            throughput=statistics.mean(m.throughput for m in trend_metrics),
            resource_usage={
                ResourceType.CPU: statistics.mean(m.resource_usage.get(ResourceType.CPU, 0) for m in trend_metrics),
                ResourceType.MEMORY: statistics.mean(m.resource_usage.get(ResourceType.MEMORY, 0) for m in trend_metrics),
                ResourceType.NETWORK: statistics.mean(m.resource_usage.get(ResourceType.NETWORK, 0) for m in trend_metrics),
                ResourceType.STORAGE: statistics.mean(m.resource_usage.get(ResourceType.STORAGE, 0) for m in trend_metrics)
            }
        )


class ScalingDecisionEngine:
    """Intelligent scaling decision engine."""
    
    def __init__(self, config: ScalingConfiguration):
        self.config = config
        self.last_scaling_time = 0
        self.current_instances = config.min_instances
        self.scaling_history = []
        
    def make_scaling_decision(self, metrics: ScalingMetrics) -> Tuple[ScalingDecision, int, str]:
        """Make intelligent scaling decision based on metrics."""
        current_time = time.time()
        
        # Check cooldown period
        if current_time - self.last_scaling_time < self.config.cooldown_period_sec:
            return ScalingDecision.MAINTAIN, self.current_instances, "Cooling down"
        
        # Emergency scaling check
        if (metrics.cpu_utilization > self.config.emergency_threshold or 
            metrics.memory_utilization > self.config.emergency_threshold or
            metrics.response_time > self.config.target_response_time_ms * 2):
            
            target_instances = min(self.config.max_instances, 
                                 self.current_instances + 2)  # Emergency scale by 2
            return ScalingDecision.EMERGENCY_SCALE, target_instances, "Emergency scaling triggered"
        
        # Calculate scaling factors
        cpu_factor = metrics.cpu_utilization / self.config.target_cpu_utilization
        memory_factor = metrics.memory_utilization / self.config.target_memory_utilization
        response_time_factor = metrics.response_time / self.config.target_response_time_ms
        
        # Determine primary scaling factor
        primary_factor = max(cpu_factor, memory_factor, response_time_factor)
        
        # Scale up conditions
        if (metrics.cpu_utilization > self.config.scale_up_threshold or
            metrics.memory_utilization > self.config.scale_up_threshold or
            metrics.response_time > self.config.target_response_time_ms * 1.5 or
            metrics.queue_size > 10):
            
            # Calculate target instances based on load
            scale_factor = max(1.2, primary_factor)  # At least 20% increase
            target_instances = min(self.config.max_instances,
                                 max(self.current_instances + 1,
                                     int(self.current_instances * scale_factor)))
            
            return ScalingDecision.SCALE_UP, target_instances, f"High load detected (factor: {primary_factor:.2f})"
        
        # Scale down conditions
        elif (metrics.cpu_utilization < self.config.scale_down_threshold and
              metrics.memory_utilization < self.config.scale_down_threshold and
              metrics.response_time < self.config.target_response_time_ms * 0.5 and
              metrics.queue_size == 0 and
              self.current_instances > self.config.min_instances):
            
            # Conservative scale down - reduce by 1 instance
            target_instances = max(self.config.min_instances,
                                 self.current_instances - 1)
            
            return ScalingDecision.SCALE_DOWN, target_instances, "Low load detected"
        
        else:
            return ScalingDecision.MAINTAIN, self.current_instances, "Load within target range"
    
    def evaluate_cost_optimization(self, metrics: ScalingMetrics) -> Dict[str, Any]:
        """Evaluate cost optimization opportunities."""
        if not self.config.cost_optimization_enabled:
            return {'enabled': False}
        
        # Calculate efficiency metrics
        cpu_efficiency = min(100, metrics.cpu_utilization / self.current_instances)
        memory_efficiency = min(100, metrics.memory_utilization / self.current_instances)
        
        # Cost per request estimation (simplified)
        cost_per_instance = 0.10  # $0.10 per hour per instance
        hourly_requests = metrics.request_rate * 3600 if metrics.request_rate > 0 else 1
        cost_per_request = (cost_per_instance * self.current_instances) / hourly_requests
        
        return {
            'enabled': True,
            'current_instances': self.current_instances,
            'cpu_efficiency': cpu_efficiency,
            'memory_efficiency': memory_efficiency,
            'estimated_hourly_cost': cost_per_instance * self.current_instances,
            'cost_per_request': cost_per_request,
            'optimization_opportunity': cpu_efficiency < 50 and memory_efficiency < 50
        }
    
    def record_scaling_event(self, event: ScalingEvent):
        """Record scaling event for analysis."""
        self.scaling_history.append(event)
        if event.success:
            self.current_instances = event.target_instances
            self.last_scaling_time = event.timestamp
        
        # Keep only recent history
        if len(self.scaling_history) > 100:
            self.scaling_history = self.scaling_history[-100:]


class ResourceManager:
    """Resource management and provisioning interface."""
    
    def __init__(self):
        self.provisioning_time = {}  # Track provisioning times per resource type
        self.active_resources = {}
        
    async def provision_instances(self, target_count: int, 
                                current_count: int) -> Dict[str, Any]:
        """Provision or deprovision instances."""
        if target_count > current_count:
            return await self._scale_up_instances(target_count - current_count)
        elif target_count < current_count:
            return await self._scale_down_instances(current_count - target_count)
        else:
            return {'action': 'maintain', 'instances': current_count}
    
    async def _scale_up_instances(self, additional_instances: int) -> Dict[str, Any]:
        """Scale up by provisioning new instances."""
        start_time = time.time()
        
        # Simulate instance provisioning time
        provisioning_time = 30 + (additional_instances * 10)  # 30-90 seconds
        await asyncio.sleep(min(2, provisioning_time / 10))  # Simulated delay
        
        execution_time = time.time() - start_time
        
        logging.info(f"Provisioned {additional_instances} new instances in {execution_time:.2f}s")
        
        return {
            'action': 'scale_up',
            'instances_added': additional_instances,
            'provisioning_time_sec': execution_time,
            'estimated_ready_time_sec': provisioning_time,
            'success': True
        }
    
    async def _scale_down_instances(self, instances_to_remove: int) -> Dict[str, Any]:
        """Scale down by terminating instances."""
        start_time = time.time()
        
        # Simulate graceful shutdown time
        shutdown_time = 10 + (instances_to_remove * 5)  # 10-60 seconds
        await asyncio.sleep(min(1, shutdown_time / 10))  # Simulated delay
        
        execution_time = time.time() - start_time
        
        logging.info(f"Terminated {instances_to_remove} instances in {execution_time:.2f}s")
        
        return {
            'action': 'scale_down',
            'instances_removed': instances_to_remove,
            'shutdown_time_sec': execution_time,
            'estimated_cleanup_time_sec': shutdown_time,
            'success': True
        }
    
    def get_resource_status(self) -> Dict[str, Any]:
        """Get current resource status."""
        return {
            'timestamp': time.time(),
            'total_cpu_cores': psutil.cpu_count(),
            'available_memory_gb': psutil.virtual_memory().available / (1024**3),
            'disk_usage_percent': psutil.disk_usage('/').percent,
            'network_connections': len(psutil.net_connections()),
            'active_processes': len(psutil.pids())
        }


class GlobalLoadBalancer:
    """Global load balancing and regional coordination."""
    
    def __init__(self):
        self.regions = {
            'us-east-1': {'instances': 2, 'load': 0.3, 'latency': 50},
            'us-west-2': {'instances': 2, 'load': 0.4, 'latency': 45},
            'eu-west-1': {'instances': 1, 'load': 0.2, 'latency': 60},
            'ap-southeast-1': {'instances': 1, 'load': 0.1, 'latency': 80}
        }
        self.traffic_distribution = defaultdict(float)
    
    def optimize_global_distribution(self, total_requests: float) -> Dict[str, Dict[str, Any]]:
        """Optimize traffic distribution across regions."""
        # Calculate optimal distribution based on capacity and latency
        regional_recommendations = {}
        
        total_capacity = sum(region['instances'] * 100 for region in self.regions.values())
        
        for region_name, region_data in self.regions.items():
            current_load = region_data['load']
            current_capacity = region_data['instances'] * 100  # 100 requests per instance
            latency_factor = 100 / region_data['latency']  # Lower latency = higher score
            
            # Calculate recommended load based on capacity and performance
            load_capacity_ratio = current_load / (current_capacity / 100) if current_capacity > 0 else 1
            
            # Scaling recommendation
            if load_capacity_ratio > 0.8:  # Over 80% capacity
                recommended_instances = region_data['instances'] + 1
                scaling_action = 'scale_up'
            elif load_capacity_ratio < 0.3 and region_data['instances'] > 1:  # Under 30% capacity
                recommended_instances = max(1, region_data['instances'] - 1)
                scaling_action = 'scale_down'
            else:
                recommended_instances = region_data['instances']
                scaling_action = 'maintain'
            
            regional_recommendations[region_name] = {
                'current_instances': region_data['instances'],
                'recommended_instances': recommended_instances,
                'scaling_action': scaling_action,
                'current_load': current_load,
                'capacity_utilization': load_capacity_ratio,
                'latency_ms': region_data['latency'],
                'performance_score': latency_factor * (1 - load_capacity_ratio)
            }
        
        return regional_recommendations
    
    def calculate_traffic_routing(self, user_location: str = 'us-east') -> Dict[str, float]:
        """Calculate optimal traffic routing weights."""
        # Simplified routing based on location and capacity
        routing_weights = {}
        
        for region_name, region_data in self.regions.items():
            # Base weight on geographic proximity (simplified)
            if user_location.startswith('us') and region_name.startswith('us'):
                proximity_weight = 0.8
            elif user_location.startswith('eu') and region_name.startswith('eu'):
                proximity_weight = 0.8
            elif user_location.startswith('ap') and region_name.startswith('ap'):
                proximity_weight = 0.8
            else:
                proximity_weight = 0.3
            
            # Adjust for current load
            load_factor = max(0.1, 1.0 - region_data['load'])
            
            # Final weight
            routing_weights[region_name] = proximity_weight * load_factor
        
        # Normalize weights
        total_weight = sum(routing_weights.values())
        if total_weight > 0:
            routing_weights = {k: v/total_weight for k, v in routing_weights.items()}
        
        return routing_weights


class AutoScalingOrchestrator:
    """Main auto-scaling orchestration system."""
    
    def __init__(self, config: ScalingConfiguration):
        self.config = config
        self.metrics_collector = MetricsCollector()
        self.decision_engine = ScalingDecisionEngine(config)
        self.resource_manager = ResourceManager()
        self.load_balancer = GlobalLoadBalancer()
        
        self.orchestrating = False
        self.scaling_events = []
        
    async def start_orchestration(self):
        """Start auto-scaling orchestration."""
        logging.info("Starting auto-scaling orchestration...")
        
        self.orchestrating = True
        self.metrics_collector.start_collection()
        
        # Start orchestration loop
        orchestration_task = asyncio.create_task(self._orchestration_loop())
        
        try:
            await orchestration_task
        except asyncio.CancelledError:
            logging.info("Auto-scaling orchestration cancelled")
        finally:
            await self.stop_orchestration()
    
    async def stop_orchestration(self):
        """Stop auto-scaling orchestration."""
        logging.info("Stopping auto-scaling orchestration...")
        
        self.orchestrating = False
        self.metrics_collector.stop_collection()
    
    async def _orchestration_loop(self):
        """Main orchestration loop."""
        while self.orchestrating:
            try:
                # Get current metrics
                current_metrics = self.metrics_collector.get_current_metrics()
                if not current_metrics:
                    await asyncio.sleep(self.config.evaluation_period_sec)
                    continue
                
                # Make scaling decision
                decision, target_instances, reason = self.decision_engine.make_scaling_decision(
                    current_metrics
                )
                
                # Execute scaling if needed
                if decision != ScalingDecision.MAINTAIN:
                    scaling_event = await self._execute_scaling(
                        decision, target_instances, current_metrics, reason
                    )
                    self.scaling_events.append(scaling_event)
                
                # Optimize global load balancing
                global_optimization = self.load_balancer.optimize_global_distribution(
                    current_metrics.request_rate
                )
                
                # Log status
                logging.info(f"Scaling status: {decision.value}, "
                           f"instances: {self.decision_engine.current_instances}, "
                           f"CPU: {current_metrics.cpu_utilization:.1f}%, "
                           f"Memory: {current_metrics.memory_utilization:.1f}%")
                
                await asyncio.sleep(self.config.evaluation_period_sec)
                
            except Exception as e:
                logging.error(f"Orchestration loop error: {e}")
                await asyncio.sleep(self.config.evaluation_period_sec)
    
    async def _execute_scaling(self, decision: ScalingDecision, 
                             target_instances: int,
                             metrics: ScalingMetrics,
                             reason: str) -> ScalingEvent:
        """Execute scaling decision."""
        start_time = time.time()
        
        try:
            # Execute resource provisioning
            provisioning_result = await self.resource_manager.provision_instances(
                target_instances, self.decision_engine.current_instances
            )
            
            execution_time = time.time() - start_time
            success = provisioning_result.get('success', True)
            
            # Create scaling event
            scaling_event = ScalingEvent(
                timestamp=start_time,
                decision=decision,
                current_instances=self.decision_engine.current_instances,
                target_instances=target_instances,
                trigger_metrics=metrics,
                reason=reason,
                success=success,
                execution_time=execution_time
            )
            
            # Record event
            self.decision_engine.record_scaling_event(scaling_event)
            
            if success:
                logging.info(f"Scaling executed successfully: {decision.value} "
                           f"from {scaling_event.current_instances} to {target_instances}")
            else:
                logging.error(f"Scaling failed: {reason}")
            
            return scaling_event
            
        except Exception as e:
            execution_time = time.time() - start_time
            logging.error(f"Scaling execution error: {e}")
            
            return ScalingEvent(
                timestamp=start_time,
                decision=decision,
                current_instances=self.decision_engine.current_instances,
                target_instances=target_instances,
                trigger_metrics=metrics,
                reason=f"Execution failed: {str(e)}",
                success=False,
                execution_time=execution_time
            )
    
    def get_scaling_status(self) -> Dict[str, Any]:
        """Get current scaling status and statistics."""
        current_metrics = self.metrics_collector.get_current_metrics()
        cost_analysis = self.decision_engine.evaluate_cost_optimization(
            current_metrics or ScalingMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0)
        )
        
        # Calculate scaling statistics
        successful_scalings = sum(1 for event in self.scaling_events if event.success)
        avg_execution_time = statistics.mean([e.execution_time for e in self.scaling_events]) if self.scaling_events else 0
        
        return {
            'orchestration_active': self.orchestrating,
            'current_instances': self.decision_engine.current_instances,
            'target_range': f"{self.config.min_instances}-{self.config.max_instances}",
            'current_metrics': current_metrics.__dict__ if current_metrics else None,
            'cost_analysis': cost_analysis,
            'scaling_statistics': {
                'total_scaling_events': len(self.scaling_events),
                'successful_scalings': successful_scalings,
                'success_rate': successful_scalings / len(self.scaling_events) if self.scaling_events else 1.0,
                'avg_execution_time_sec': avg_execution_time
            },
            'configuration': {
                'evaluation_period_sec': self.config.evaluation_period_sec,
                'cooldown_period_sec': self.config.cooldown_period_sec,
                'target_cpu_percent': self.config.target_cpu_utilization,
                'target_response_time_ms': self.config.target_response_time_ms
            }
        }
    
    def generate_scaling_report(self) -> str:
        """Generate comprehensive scaling report."""
        status = self.get_scaling_status()
        
        report = [
            "# Auto-scaling Infrastructure Report",
            "",
            "## Current Status",
            f"- Active Instances: {status['current_instances']}",
            f"- Instance Range: {status['target_range']}",
            f"- Orchestration Active: {'✅' if status['orchestration_active'] else '❌'}",
            ""
        ]
        
        if status['current_metrics']:
            metrics = status['current_metrics']
            report.extend([
                "## Current Performance",
                f"- CPU Utilization: {metrics['cpu_utilization']:.1f}%",
                f"- Memory Utilization: {metrics['memory_utilization']:.1f}%",
                f"- Request Rate: {metrics['request_rate']:.1f} req/sec",
                f"- Response Time: {metrics['response_time']:.2f} ms",
                f"- Error Rate: {metrics['error_rate']:.2%}",
                ""
            ])
        
        stats = status['scaling_statistics']
        report.extend([
            "## Scaling Statistics",
            f"- Total Scaling Events: {stats['total_scaling_events']}",
            f"- Success Rate: {stats['success_rate']:.2%}",
            f"- Average Execution Time: {stats['avg_execution_time_sec']:.2f} seconds",
            ""
        ])
        
        cost = status['cost_analysis']
        if cost.get('enabled'):
            report.extend([
                "## Cost Analysis",
                f"- Current Instances: {cost['current_instances']}",
                f"- Estimated Hourly Cost: ${cost['estimated_hourly_cost']:.2f}",
                f"- Cost per Request: ${cost['cost_per_request']:.4f}",
                f"- CPU Efficiency: {cost['cpu_efficiency']:.1f}%",
                f"- Memory Efficiency: {cost['memory_efficiency']:.1f}%",
                ""
            ])
        
        report.extend([
            "## Configuration",
            f"- Evaluation Period: {status['configuration']['evaluation_period_sec']} seconds",
            f"- Cooldown Period: {status['configuration']['cooldown_period_sec']} seconds",
            f"- Target CPU: {status['configuration']['target_cpu_percent']}%",
            f"- Target Response Time: {status['configuration']['target_response_time_ms']} ms",
            "",
            f"*Report generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}*"
        ])
        
        return "\n".join(report)
    
    def export_scaling_data(self, output_path: Path):
        """Export scaling data for analysis."""
        export_data = {
            'autoscaling_version': '2025.1',
            'timestamp': time.time(),
            'configuration': self.config.__dict__,
            'current_status': self.get_scaling_status(),
            'scaling_events': [event.__dict__ for event in self.scaling_events],
            'metrics_history': [m.__dict__ for m in list(self.metrics_collector.metrics_history)],
            'global_distribution': self.load_balancer.regions
        }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logging.info(f"Scaling data exported to {output_path}")


async def main():
    """Demonstrate auto-scaling infrastructure."""
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Configure auto-scaling
    config = ScalingConfiguration(
        min_instances=1,
        max_instances=5,
        target_cpu_utilization=70.0,
        target_response_time_ms=200.0,
        evaluation_period_sec=10,  # Faster evaluation for demo
        cooldown_period_sec=30     # Shorter cooldown for demo
    )
    
    # Initialize orchestrator
    orchestrator = AutoScalingOrchestrator(config)
    
    # Simulate load patterns
    async def simulate_load():
        """Simulate varying load patterns."""
        patterns = [
            # (requests_per_sec, duration_sec, description)
            (5, 30, "baseline load"),
            (15, 20, "increased load"),
            (25, 15, "high load"),
            (35, 10, "peak load"),
            (10, 20, "reduced load"),
            (2, 15, "low load")
        ]
        
        for request_rate, duration, description in patterns:
            logging.info(f"Simulating {description}: {request_rate} req/sec for {duration}s")
            
            # Simulate requests
            end_time = time.time() + duration
            while time.time() < end_time:
                # Record simulated requests
                response_time = 0.05 + (request_rate / 100)  # Simulate load impact
                error = request_rate > 30 and np.random.random() < 0.05  # Errors under high load
                
                orchestrator.metrics_collector.record_request(response_time, error)
                
                await asyncio.sleep(1.0 / max(1, request_rate))
    
    try:
        # Start auto-scaling orchestration
        orchestration_task = asyncio.create_task(orchestrator.start_orchestration())
        
        # Start load simulation
        load_task = asyncio.create_task(simulate_load())
        
        # Wait for load simulation to complete
        await load_task
        
        # Let orchestrator run a bit longer
        await asyncio.sleep(30)
        
        # Stop orchestration
        orchestration_task.cancel()
        
        try:
            await orchestration_task
        except asyncio.CancelledError:
            pass
        
        # Generate and display report
        report = orchestrator.generate_scaling_report()
        print("\n" + "="*80)
        print(report)
        print("="*80)
        
        # Export scaling data
        orchestrator.export_scaling_data(Path("autoscaling_results.json"))
        
        logging.info("Auto-scaling infrastructure demonstration completed!")
        
    except Exception as e:
        logging.error(f"Auto-scaling demonstration failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())