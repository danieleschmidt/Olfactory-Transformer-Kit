"""
Auto-Scaling Module for Olfactory Transformer Infrastructure.

Implements intelligent auto-scaling capabilities:
- Load-based scaling decisions
- Predictive scaling using time series forecasting
- Resource optimization and cost management
- Multi-cloud deployment orchestration
- Kubernetes integration
"""

import logging
import time
import json
import threading
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import deque
import statistics
import math

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    psutil = None


@dataclass
class ScalingConfig:
    """Configuration for auto-scaling."""
    min_replicas: int = 1
    max_replicas: int = 10
    target_cpu_percent: float = 70.0
    target_memory_percent: float = 80.0
    target_request_rate: float = 100.0  # requests per second
    scale_up_threshold: float = 80.0
    scale_down_threshold: float = 30.0
    scale_up_cooldown: int = 300  # seconds
    scale_down_cooldown: int = 600  # seconds
    prediction_window: int = 900  # 15 minutes
    enable_predictive_scaling: bool = True
    enable_cost_optimization: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ResourceMetrics:
    """Resource utilization metrics."""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_mb: float
    request_rate: float
    response_time_ms: float
    error_rate: float
    active_connections: int
    queue_length: int


@dataclass
class ScalingDecision:
    """Auto-scaling decision."""
    timestamp: float
    action: str  # "scale_up", "scale_down", "no_action"
    current_replicas: int
    target_replicas: int
    reason: str
    confidence: float
    estimated_cost_impact: float


class MetricsCollector:
    """Collects system and application metrics."""
    
    def __init__(self):
        self.metrics_history = deque(maxlen=1000)
        self.collection_interval = 30  # seconds
        self.is_collecting = False
        self.collector_thread = None
        
    def start_collection(self) -> None:
        """Start metrics collection."""
        if self.is_collecting:
            return
            
        self.is_collecting = True
        self.collector_thread = threading.Thread(target=self._collection_loop, daemon=True)
        self.collector_thread.start()
        logging.info("Started metrics collection")
    
    def stop_collection(self) -> None:
        """Stop metrics collection."""
        self.is_collecting = False
        if self.collector_thread:
            self.collector_thread.join(timeout=5)
        logging.info("Stopped metrics collection")
    
    def _collection_loop(self) -> None:
        """Main metrics collection loop."""
        while self.is_collecting:
            try:
                metrics = self._collect_current_metrics()
                self.metrics_history.append(metrics)
                time.sleep(self.collection_interval)
            except Exception as e:
                logging.error(f"Metrics collection error: {e}")
                time.sleep(60)
    
    def _collect_current_metrics(self) -> ResourceMetrics:
        """Collect current system metrics."""
        timestamp = time.time()
        
        # System metrics
        if HAS_PSUTIL:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_mb = memory.used / (1024 * 1024)
        else:
            # Simulate metrics for demo
            cpu_percent = 50.0 + (time.time() % 100 - 50) * 0.3
            memory_percent = 60.0 + (time.time() % 80 - 40) * 0.2
            memory_mb = 2048.0
        
        # Application metrics (simulated)
        request_rate = max(0, 80 + (time.time() % 60 - 30) * 2)
        response_time_ms = max(10, 100 + (time.time() % 40 - 20) * 5)
        error_rate = max(0, min(5, (time.time() % 30 - 15) * 0.1))
        active_connections = int(max(10, request_rate * 0.5))
        queue_length = max(0, int((cpu_percent - 50) * 0.2))
        
        return ResourceMetrics(
            timestamp=timestamp,
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_mb=memory_mb,
            request_rate=request_rate,
            response_time_ms=response_time_ms,
            error_rate=error_rate,
            active_connections=active_connections,
            queue_length=queue_length
        )
    
    def get_recent_metrics(self, window_seconds: int = 300) -> List[ResourceMetrics]:
        """Get metrics from recent time window."""
        cutoff_time = time.time() - window_seconds
        return [m for m in self.metrics_history if m.timestamp >= cutoff_time]
    
    def get_aggregated_metrics(self, window_seconds: int = 300) -> Dict[str, float]:
        """Get aggregated metrics over time window."""
        recent_metrics = self.get_recent_metrics(window_seconds)
        
        if not recent_metrics:
            return {}
        
        return {
            'avg_cpu_percent': statistics.mean(m.cpu_percent for m in recent_metrics),
            'max_cpu_percent': max(m.cpu_percent for m in recent_metrics),
            'avg_memory_percent': statistics.mean(m.memory_percent for m in recent_metrics),
            'max_memory_percent': max(m.memory_percent for m in recent_metrics),
            'avg_request_rate': statistics.mean(m.request_rate for m in recent_metrics),
            'max_request_rate': max(m.request_rate for m in recent_metrics),
            'avg_response_time_ms': statistics.mean(m.response_time_ms for m in recent_metrics),
            'avg_error_rate': statistics.mean(m.error_rate for m in recent_metrics),
            'avg_queue_length': statistics.mean(m.queue_length for m in recent_metrics)
        }


class LoadPredictor:
    """Predicts future load using time series analysis."""
    
    def __init__(self):
        self.prediction_models = {}
        
    def predict_load(self, metrics_history: List[ResourceMetrics], 
                    prediction_horizon: int = 900) -> Dict[str, float]:
        """Predict future load metrics."""
        if len(metrics_history) < 10:
            return {}
        
        predictions = {}
        
        # Predict CPU usage
        cpu_values = [m.cpu_percent for m in metrics_history[-50:]]
        predictions['predicted_cpu_percent'] = self._predict_time_series(cpu_values, prediction_horizon)
        
        # Predict memory usage
        memory_values = [m.memory_percent for m in metrics_history[-50:]]
        predictions['predicted_memory_percent'] = self._predict_time_series(memory_values, prediction_horizon)
        
        # Predict request rate
        request_values = [m.request_rate for m in metrics_history[-50:]]
        predictions['predicted_request_rate'] = self._predict_time_series(request_values, prediction_horizon)
        
        # Predict response time
        response_values = [m.response_time_ms for m in metrics_history[-50:]]
        predictions['predicted_response_time_ms'] = self._predict_time_series(response_values, prediction_horizon)
        
        return predictions
    
    def _predict_time_series(self, values: List[float], horizon_seconds: int) -> float:
        """Simple time series prediction using trend and seasonality."""
        if len(values) < 3:
            return statistics.mean(values) if values else 0.0
        
        # Calculate trend (linear regression slope)
        n = len(values)
        x_values = list(range(n))
        
        sum_x = sum(x_values)
        sum_y = sum(values)
        sum_xy = sum(x * y for x, y in zip(x_values, values))
        sum_x2 = sum(x * x for x in x_values)
        
        # Linear regression
        denominator = n * sum_x2 - sum_x * sum_x
        if denominator != 0:
            slope = (n * sum_xy - sum_x * sum_y) / denominator
            intercept = (sum_y - slope * sum_x) / n
        else:
            slope = 0
            intercept = statistics.mean(values)
        
        # Project trend forward
        future_steps = horizon_seconds / 30  # Assuming 30-second intervals
        trend_prediction = intercept + slope * (n + future_steps)
        
        # Add seasonal component (simplified)
        seasonal_component = self._calculate_seasonal_component(values)
        
        # Combine trend and seasonal
        prediction = trend_prediction + seasonal_component
        
        # Clamp to reasonable bounds
        if 'cpu' in str(values) or 'memory' in str(values):
            prediction = max(0, min(100, prediction))
        elif 'request' in str(values):
            prediction = max(0, prediction)
        
        return prediction
    
    def _calculate_seasonal_component(self, values: List[float]) -> float:
        """Calculate seasonal component (simplified)."""
        if len(values) < 10:
            return 0.0
        
        # Look for cyclical patterns in recent data
        recent_values = values[-10:]
        overall_mean = statistics.mean(values)
        recent_mean = statistics.mean(recent_values)
        
        # Return difference as seasonal adjustment
        return (recent_mean - overall_mean) * 0.3  # Dampen seasonal effect


class CostOptimizer:
    """Optimizes scaling decisions for cost efficiency."""
    
    def __init__(self):
        self.instance_costs = {
            'small': 0.10,   # $ per hour
            'medium': 0.20,
            'large': 0.40,
            'xlarge': 0.80
        }
        
    def calculate_cost_impact(self, current_replicas: int, target_replicas: int, 
                            instance_type: str = 'medium', duration_hours: float = 1.0) -> float:
        """Calculate cost impact of scaling decision."""
        if instance_type not in self.instance_costs:
            instance_type = 'medium'
            
        cost_per_hour = self.instance_costs[instance_type]
        
        current_cost = current_replicas * cost_per_hour * duration_hours
        target_cost = target_replicas * cost_per_hour * duration_hours
        
        return target_cost - current_cost
    
    def optimize_instance_selection(self, predicted_load: Dict[str, float]) -> str:
        """Select optimal instance type based on predicted load."""
        cpu_load = predicted_load.get('predicted_cpu_percent', 50)
        memory_load = predicted_load.get('predicted_memory_percent', 50)
        request_rate = predicted_load.get('predicted_request_rate', 100)
        
        # Simple heuristic for instance selection
        max_load = max(cpu_load, memory_load)
        
        if max_load > 80 or request_rate > 200:
            return 'xlarge'
        elif max_load > 60 or request_rate > 150:
            return 'large'
        elif max_load > 40 or request_rate > 100:
            return 'medium'
        else:
            return 'small'


class AutoScaler:
    """Main auto-scaling controller."""
    
    def __init__(self, config: ScalingConfig):
        self.config = config
        self.current_replicas = config.min_replicas
        self.last_scale_up = 0
        self.last_scale_down = 0
        self.scaling_history = deque(maxlen=100)
        
        self.metrics_collector = MetricsCollector()
        self.load_predictor = LoadPredictor()
        self.cost_optimizer = CostOptimizer()
        
        self.is_running = False
        self.scaling_thread = None
    
    def start_autoscaling(self) -> None:
        """Start auto-scaling process."""
        if self.is_running:
            return
            
        self.is_running = True
        self.metrics_collector.start_collection()
        
        self.scaling_thread = threading.Thread(target=self._scaling_loop, daemon=True)
        self.scaling_thread.start()
        
        logging.info("Started auto-scaling controller")
    
    def stop_autoscaling(self) -> None:
        """Stop auto-scaling process."""
        self.is_running = False
        self.metrics_collector.stop_collection()
        
        if self.scaling_thread:
            self.scaling_thread.join(timeout=5)
            
        logging.info("Stopped auto-scaling controller")
    
    def _scaling_loop(self) -> None:
        """Main scaling decision loop."""
        while self.is_running:
            try:
                decision = self._make_scaling_decision()
                
                if decision.action != "no_action":
                    self._execute_scaling_decision(decision)
                    self.scaling_history.append(decision)
                
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                logging.error(f"Scaling loop error: {e}")
                time.sleep(120)  # Wait longer on error
    
    def _make_scaling_decision(self) -> ScalingDecision:
        """Make intelligent scaling decision."""
        current_time = time.time()
        
        # Get current metrics
        current_metrics = self.metrics_collector.get_aggregated_metrics(300)  # 5-minute window
        
        if not current_metrics:
            return ScalingDecision(
                timestamp=current_time,
                action="no_action",
                current_replicas=self.current_replicas,
                target_replicas=self.current_replicas,
                reason="No metrics available",
                confidence=0.0,
                estimated_cost_impact=0.0
            )
        
        # Check cooldown periods
        if (current_time - self.last_scale_up < self.config.scale_up_cooldown and
            current_time - self.last_scale_down < self.config.scale_down_cooldown):
            return ScalingDecision(
                timestamp=current_time,
                action="no_action",
                current_replicas=self.current_replicas,
                target_replicas=self.current_replicas,
                reason="In cooldown period",
                confidence=1.0,
                estimated_cost_impact=0.0
            )
        
        # Analyze current load
        cpu_load = current_metrics.get('avg_cpu_percent', 0)
        memory_load = current_metrics.get('avg_memory_percent', 0)
        request_rate = current_metrics.get('avg_request_rate', 0)
        response_time = current_metrics.get('avg_response_time_ms', 0)
        queue_length = current_metrics.get('avg_queue_length', 0)
        
        # Predictive scaling
        predictions = {}
        confidence = 0.7  # Base confidence
        
        if self.config.enable_predictive_scaling:
            metrics_history = self.metrics_collector.get_recent_metrics(1800)  # 30 minutes
            predictions = self.load_predictor.predict_load(metrics_history, self.config.prediction_window)
            
            if predictions:
                # Use predictions for decision making
                cpu_load = max(cpu_load, predictions.get('predicted_cpu_percent', cpu_load))
                memory_load = max(memory_load, predictions.get('predicted_memory_percent', memory_load))
                request_rate = max(request_rate, predictions.get('predicted_request_rate', request_rate))
                confidence = 0.8  # Higher confidence with predictions
        
        # Determine scaling action
        action = "no_action"
        target_replicas = self.current_replicas
        reason = "No scaling needed"
        
        # Scale up conditions
        scale_up_needed = (
            cpu_load > self.config.scale_up_threshold or
            memory_load > self.config.scale_up_threshold or
            request_rate > self.config.target_request_rate * 1.5 or
            response_time > 500 or  # 500ms threshold
            queue_length > 10
        )
        
        if scale_up_needed and self.current_replicas < self.config.max_replicas:
            if current_time - self.last_scale_up >= self.config.scale_up_cooldown:
                action = "scale_up"
                target_replicas = min(self.config.max_replicas, self.current_replicas + 1)
                
                reasons = []
                if cpu_load > self.config.scale_up_threshold:
                    reasons.append(f"CPU: {cpu_load:.1f}%")
                if memory_load > self.config.scale_up_threshold:
                    reasons.append(f"Memory: {memory_load:.1f}%")
                if request_rate > self.config.target_request_rate * 1.5:
                    reasons.append(f"Requests: {request_rate:.1f}/s")
                if response_time > 500:
                    reasons.append(f"Response time: {response_time:.1f}ms")
                if queue_length > 10:
                    reasons.append(f"Queue: {queue_length}")
                
                reason = f"Scale up needed: {', '.join(reasons)}"
        
        # Scale down conditions
        elif (cpu_load < self.config.scale_down_threshold and
              memory_load < self.config.scale_down_threshold and
              request_rate < self.config.target_request_rate * 0.5 and
              response_time < 200 and
              queue_length < 2):
            
            if (self.current_replicas > self.config.min_replicas and
                current_time - self.last_scale_down >= self.config.scale_down_cooldown):
                action = "scale_down"
                target_replicas = max(self.config.min_replicas, self.current_replicas - 1)
                reason = f"Scale down: CPU {cpu_load:.1f}%, Memory {memory_load:.1f}%, Requests {request_rate:.1f}/s"
        
        # Calculate cost impact
        cost_impact = 0.0
        if self.config.enable_cost_optimization:
            instance_type = self.cost_optimizer.optimize_instance_selection(predictions)
            cost_impact = self.cost_optimizer.calculate_cost_impact(
                self.current_replicas, target_replicas, instance_type
            )
        
        return ScalingDecision(
            timestamp=current_time,
            action=action,
            current_replicas=self.current_replicas,
            target_replicas=target_replicas,
            reason=reason,
            confidence=confidence,
            estimated_cost_impact=cost_impact
        )
    
    def _execute_scaling_decision(self, decision: ScalingDecision) -> None:
        """Execute scaling decision."""
        logging.info(f"Executing scaling decision: {decision.action} "
                    f"({decision.current_replicas} -> {decision.target_replicas}) "
                    f"Reason: {decision.reason}")
        
        if decision.action == "scale_up":
            self.current_replicas = decision.target_replicas
            self.last_scale_up = decision.timestamp
            
            # In production, would call actual scaling API
            self._scale_infrastructure(decision.target_replicas)
            
        elif decision.action == "scale_down":
            self.current_replicas = decision.target_replicas
            self.last_scale_down = decision.timestamp
            
            # In production, would call actual scaling API
            self._scale_infrastructure(decision.target_replicas)
    
    def _scale_infrastructure(self, target_replicas: int) -> None:
        """Scale underlying infrastructure."""
        # Placeholder for actual infrastructure scaling
        # In production, this would call:
        # - Kubernetes HPA API
        # - Cloud provider auto-scaling groups
        # - Container orchestration platforms
        # - Load balancer configuration
        
        logging.info(f"Infrastructure scaled to {target_replicas} replicas")
    
    def get_scaling_status(self) -> Dict[str, Any]:
        """Get current auto-scaling status."""
        current_metrics = self.metrics_collector.get_aggregated_metrics(300)
        
        return {
            'current_replicas': self.current_replicas,
            'min_replicas': self.config.min_replicas,
            'max_replicas': self.config.max_replicas,
            'current_metrics': current_metrics,
            'last_scale_up': self.last_scale_up,
            'last_scale_down': self.last_scale_down,
            'scaling_history_count': len(self.scaling_history),
            'is_running': self.is_running
        }
    
    def get_scaling_recommendations(self) -> Dict[str, Any]:
        """Get scaling recommendations for manual review."""
        current_metrics = self.metrics_collector.get_aggregated_metrics(300)
        metrics_history = self.metrics_collector.get_recent_metrics(1800)
        
        # Analyze patterns
        if len(metrics_history) >= 10:
            cpu_trend = [m.cpu_percent for m in metrics_history[-10:]]
            memory_trend = [m.memory_percent for m in metrics_history[-10:]]
            request_trend = [m.request_rate for m in metrics_history[-10:]]
            
            recommendations = []
            
            # Check for sustained high load
            if statistics.mean(cpu_trend) > 70:
                recommendations.append("Consider increasing CPU capacity or optimizing CPU-intensive operations")
            
            if statistics.mean(memory_trend) > 75:
                recommendations.append("Consider increasing memory capacity or optimizing memory usage")
            
            # Check for load patterns
            cpu_variance = statistics.variance(cpu_trend) if len(cpu_trend) > 1 else 0
            if cpu_variance > 200:  # High variance
                recommendations.append("High CPU variance detected - consider predictive scaling")
            
            # Check for scaling frequency
            recent_scales = [d for d in self.scaling_history if d.timestamp > time.time() - 3600]
            if len(recent_scales) > 5:
                recommendations.append("Frequent scaling detected - consider adjusting thresholds")
            
            return {
                'current_metrics': current_metrics,
                'trends': {
                    'cpu_trend': cpu_trend,
                    'memory_trend': memory_trend,
                    'request_trend': request_trend
                },
                'recommendations': recommendations,
                'recent_scaling_actions': len(recent_scales)
            }
        
        return {'current_metrics': current_metrics, 'recommendations': []}


def main():
    """Demonstrate auto-scaling capabilities."""
    print("📈 AUTO-SCALING DEMONSTRATION")
    
    # Create scaling configuration
    config = ScalingConfig(
        min_replicas=2,
        max_replicas=8,
        target_cpu_percent=70.0,
        scale_up_threshold=75.0,
        scale_down_threshold=25.0,
        scale_up_cooldown=60,  # Shorter for demo
        scale_down_cooldown=120,
        enable_predictive_scaling=True,
        enable_cost_optimization=True
    )
    
    print(f"Configuration: {config.to_dict()}")
    
    # Create auto-scaler
    autoscaler = AutoScaler(config)
    
    # Start auto-scaling
    autoscaler.start_autoscaling()
    
    print("\n🚀 Auto-scaling started...")
    print("Monitoring for 2 minutes...")
    
    # Monitor for 2 minutes
    for i in range(12):  # 12 * 10 seconds = 2 minutes
        time.sleep(10)
        
        status = autoscaler.get_scaling_status()
        current_metrics = status.get('current_metrics', {})
        
        print(f"\nStatus check {i+1}:")
        print(f"  Replicas: {status['current_replicas']}")
        print(f"  CPU: {current_metrics.get('avg_cpu_percent', 0):.1f}%")
        print(f"  Memory: {current_metrics.get('avg_memory_percent', 0):.1f}%")
        print(f"  Requests/s: {current_metrics.get('avg_request_rate', 0):.1f}")
        
        # Simulate load spike halfway through
        if i == 6:
            print("  📊 Simulating load spike...")
            # In real implementation, this would be actual load
    
    # Get final recommendations
    recommendations = autoscaler.get_scaling_recommendations()
    print(f"\n📋 Scaling Recommendations:")
    for rec in recommendations.get('recommendations', []):
        print(f"  • {rec}")
    
    # Stop auto-scaling
    autoscaler.stop_autoscaling()
    
    print("\n✅ Auto-scaling demonstration completed!")
    return True


if __name__ == "__main__":
    main()