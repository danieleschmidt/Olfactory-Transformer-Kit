"""
Intelligent Auto-Scaling System for Generation 3.

Implements predictive auto-scaling, load balancing, and resource optimization
with machine learning-driven capacity planning.
"""

import asyncio
import logging
import time
import threading
from typing import Dict, List, Optional, Any, Callable, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import json
import statistics
import math
from enum import Enum
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, Future
import queue
import random

logger = logging.getLogger(__name__)

class ScalingDirection(Enum):
    """Scaling direction options."""
    UP = "up"
    DOWN = "down"
    MAINTAIN = "maintain"

class ResourceType(Enum):
    """Types of resources to scale."""
    CPU = "cpu"
    MEMORY = "memory"
    THREADS = "threads"
    PROCESSES = "processes"
    CONNECTIONS = "connections"
    CACHE = "cache"

@dataclass
class ScalingMetric:
    """Metrics for scaling decisions."""
    resource_type: ResourceType
    current_usage: float
    target_usage: float
    predicted_usage: float
    confidence: float
    timestamp: datetime
    scaling_pressure: float = 0.0

@dataclass
class ScalingEvent:
    """Record of scaling events."""
    timestamp: datetime
    resource_type: ResourceType
    direction: ScalingDirection
    from_capacity: int
    to_capacity: int
    trigger_metric: float
    success: bool
    duration: float
    cost: float = 0.0

class LoadPredictor:
    """ML-inspired load prediction system."""
    
    def __init__(self, history_size: int = 1000):
        self.history_size = history_size
        self.load_history: deque = deque(maxlen=history_size)
        self.pattern_cache: Dict[str, List[float]] = {}
        self.seasonal_patterns: Dict[str, List[float]] = defaultdict(list)
        
    def record_load(self, load: float, timestamp: Optional[datetime] = None):
        """Record current load measurement."""
        if timestamp is None:
            timestamp = datetime.now()
            
        self.load_history.append((timestamp, load))
        
        # Update seasonal patterns
        hour = timestamp.hour
        day_of_week = timestamp.weekday()
        
        self.seasonal_patterns[f"hour_{hour}"].append(load)
        self.seasonal_patterns[f"weekday_{day_of_week}"].append(load)
        
        # Trim seasonal data
        for pattern_key in self.seasonal_patterns:
            if len(self.seasonal_patterns[pattern_key]) > 100:
                self.seasonal_patterns[pattern_key] = self.seasonal_patterns[pattern_key][-50:]
                
    def predict_load(self, minutes_ahead: int = 5) -> Tuple[float, float]:
        """Predict load N minutes ahead with confidence."""
        
        if len(self.load_history) < 3:
            current_load = self.load_history[-1][1] if self.load_history else 50.0
            return current_load, 0.5
            
        # Multiple prediction methods
        trend_prediction = self._trend_prediction(minutes_ahead)
        seasonal_prediction = self._seasonal_prediction(minutes_ahead)
        pattern_prediction = self._pattern_prediction(minutes_ahead)
        
        # Ensemble prediction with weights
        weights = [0.4, 0.3, 0.3]  # trend, seasonal, pattern
        predictions = [trend_prediction, seasonal_prediction, pattern_prediction]
        
        # Remove None predictions and adjust weights
        valid_predictions = [(p, w) for p, w in zip(predictions, weights) if p is not None]
        
        if not valid_predictions:
            current_load = self.load_history[-1][1]
            return current_load, 0.3
            
        # Weighted average
        total_weight = sum(w for _, w in valid_predictions)
        weighted_sum = sum(p * w for p, w in valid_predictions)
        ensemble_prediction = weighted_sum / total_weight
        
        # Calculate confidence based on prediction consistency
        prediction_variance = statistics.variance([p for p, _ in valid_predictions]) if len(valid_predictions) > 1 else 0
        confidence = max(0.1, 1.0 - prediction_variance / 100.0)
        
        return ensemble_prediction, confidence
        
    def _trend_prediction(self, minutes_ahead: int) -> Optional[float]:
        """Linear trend-based prediction."""
        if len(self.load_history) < 5:
            return None
            
        # Use last 10 points for trend
        recent_points = list(self.load_history)[-10:]
        
        # Simple linear regression
        x_values = list(range(len(recent_points)))
        y_values = [point[1] for point in recent_points]
        
        if len(x_values) < 2:
            return None
            
        # Calculate slope and intercept
        n = len(x_values)
        sum_x = sum(x_values)
        sum_y = sum(y_values)
        sum_xy = sum(x * y for x, y in zip(x_values, y_values))
        sum_x2 = sum(x * x for x in x_values)
        
        denominator = n * sum_x2 - sum_x * sum_x
        if denominator == 0:
            return y_values[-1]
            
        slope = (n * sum_xy - sum_x * sum_y) / denominator
        intercept = (sum_y - slope * sum_x) / n
        
        # Predict future point
        future_x = len(x_values) + (minutes_ahead / 5)  # Assuming 5-minute intervals
        prediction = slope * future_x + intercept
        
        return max(0, prediction)
        
    def _seasonal_prediction(self, minutes_ahead: int) -> Optional[float]:
        """Seasonal pattern-based prediction."""
        target_time = datetime.now() + timedelta(minutes=minutes_ahead)
        hour = target_time.hour
        day_of_week = target_time.weekday()
        
        # Get seasonal patterns
        hourly_pattern = self.seasonal_patterns.get(f"hour_{hour}", [])
        daily_pattern = self.seasonal_patterns.get(f"weekday_{day_of_week}", [])
        
        predictions = []
        if hourly_pattern:
            predictions.append(statistics.mean(hourly_pattern))
        if daily_pattern:
            predictions.append(statistics.mean(daily_pattern))
            
        if predictions:
            return statistics.mean(predictions)
        return None
        
    def _pattern_prediction(self, minutes_ahead: int) -> Optional[float]:
        """Pattern matching-based prediction."""
        if len(self.load_history) < 20:
            return None
            
        # Extract recent pattern
        recent_pattern = [point[1] for point in list(self.load_history)[-10:]]
        
        # Find similar patterns in history
        pattern_length = len(recent_pattern)
        history_values = [point[1] for point in self.load_history]
        
        similar_patterns = []
        for i in range(len(history_values) - pattern_length - 5):
            historical_pattern = history_values[i:i + pattern_length]
            
            # Calculate pattern similarity (inverse of mean squared error)
            mse = sum((a - b) ** 2 for a, b in zip(recent_pattern, historical_pattern)) / pattern_length
            similarity = 1.0 / (mse + 1.0)
            
            if similarity > 0.1:  # Threshold for similar patterns
                # Get the next values after this pattern
                future_start = i + pattern_length
                future_end = min(future_start + 5, len(history_values))
                if future_end > future_start:
                    future_values = history_values[future_start:future_end]
                    similar_patterns.append((similarity, future_values))
                    
        if similar_patterns:
            # Weight patterns by similarity
            weighted_predictions = []
            total_weight = 0
            
            for similarity, future_values in similar_patterns:
                if future_values:
                    predicted_value = future_values[0]  # Next immediate value
                    weighted_predictions.append(predicted_value * similarity)
                    total_weight += similarity
                    
            if total_weight > 0:
                return sum(weighted_predictions) / total_weight
                
        return None

class IntelligentScaler:
    """Main intelligent scaling system."""
    
    def __init__(self):
        self.load_predictor = LoadPredictor()
        self.scaling_history: List[ScalingEvent] = []
        self.resource_limits: Dict[ResourceType, Dict[str, int]] = self._initialize_resource_limits()
        self.current_capacity: Dict[ResourceType, int] = self._initialize_current_capacity()
        self.scaling_policies: Dict[ResourceType, Dict[str, float]] = self._initialize_scaling_policies()
        self.cool_down_periods: Dict[ResourceType, datetime] = {}
        self._lock = threading.RLock()
        
        # Scaling executors
        self.thread_pool: Optional[ThreadPoolExecutor] = None
        self.process_pool: Optional[ProcessPoolExecutor] = None
        
        # Start monitoring
        self._start_monitoring()
        
    def _initialize_resource_limits(self) -> Dict[ResourceType, Dict[str, int]]:
        """Initialize resource limits."""
        cpu_count = mp.cpu_count()
        
        return {
            ResourceType.THREADS: {"min": cpu_count, "max": cpu_count * 8},
            ResourceType.PROCESSES: {"min": 2, "max": cpu_count * 2},
            ResourceType.CONNECTIONS: {"min": 10, "max": 1000},
            ResourceType.CACHE: {"min": 64, "max": 1024},  # MB
            ResourceType.MEMORY: {"min": 512, "max": 8192},  # MB
        }
        
    def _initialize_current_capacity(self) -> Dict[ResourceType, int]:
        """Initialize current capacity."""
        cpu_count = mp.cpu_count()
        
        return {
            ResourceType.THREADS: cpu_count * 2,
            ResourceType.PROCESSES: max(2, cpu_count // 2),
            ResourceType.CONNECTIONS: 50,
            ResourceType.CACHE: 256,  # MB
            ResourceType.MEMORY: 1024,  # MB
        }
        
    def _initialize_scaling_policies(self) -> Dict[ResourceType, Dict[str, float]]:
        """Initialize scaling policies."""
        return {
            ResourceType.THREADS: {
                "scale_up_threshold": 0.8,
                "scale_down_threshold": 0.3,
                "scale_factor": 1.5,
                "cool_down_minutes": 2
            },
            ResourceType.PROCESSES: {
                "scale_up_threshold": 0.85,
                "scale_down_threshold": 0.25,
                "scale_factor": 1.3,
                "cool_down_minutes": 5
            },
            ResourceType.CONNECTIONS: {
                "scale_up_threshold": 0.9,
                "scale_down_threshold": 0.4,
                "scale_factor": 1.2,
                "cool_down_minutes": 1
            },
            ResourceType.CACHE: {
                "scale_up_threshold": 0.85,
                "scale_down_threshold": 0.5,
                "scale_factor": 1.4,
                "cool_down_minutes": 10
            },
            ResourceType.MEMORY: {
                "scale_up_threshold": 0.8,
                "scale_down_threshold": 0.4,
                "scale_factor": 1.3,
                "cool_down_minutes": 5
            },
            ResourceType.CPU: {
                "scale_up_threshold": 0.75,
                "scale_down_threshold": 0.3,
                "scale_factor": 1.2,
                "cool_down_minutes": 3
            }
        }
        
    def record_resource_usage(self, resource_type: ResourceType, usage_percent: float):
        """Record resource usage for scaling decisions."""
        with self._lock:
            # Record for load prediction
            if resource_type == ResourceType.CPU:
                self.load_predictor.record_load(usage_percent)
                
            # Check if scaling is needed
            self._evaluate_scaling_need(resource_type, usage_percent)
            
    def _evaluate_scaling_need(self, resource_type: ResourceType, current_usage: float):
        """Evaluate if scaling is needed for a resource."""
        
        # Check cool-down period
        if resource_type in self.cool_down_periods:
            cool_down_end = self.cool_down_periods[resource_type]
            if datetime.now() < cool_down_end:
                return  # Still in cool-down period
                
        policy = self.scaling_policies.get(resource_type, {})
        scale_up_threshold = policy.get("scale_up_threshold", 0.8)
        scale_down_threshold = policy.get("scale_down_threshold", 0.3)
        
        # Predict future load for proactive scaling
        predicted_usage, confidence = self.load_predictor.predict_load(5)
        
        # Create scaling metric
        metric = ScalingMetric(
            resource_type=resource_type,
            current_usage=current_usage,
            target_usage=70.0,  # Target 70% utilization
            predicted_usage=predicted_usage,
            confidence=confidence,
            timestamp=datetime.now(),
            scaling_pressure=self._calculate_scaling_pressure(current_usage, predicted_usage)
        )
        
        # Determine scaling direction
        scaling_direction = None
        
        if current_usage > scale_up_threshold or (predicted_usage > scale_up_threshold and confidence > 0.7):
            scaling_direction = ScalingDirection.UP
        elif current_usage < scale_down_threshold and predicted_usage < scale_down_threshold:
            scaling_direction = ScalingDirection.DOWN
            
        if scaling_direction:
            self._execute_scaling(resource_type, scaling_direction, metric)
            
    def _calculate_scaling_pressure(self, current_usage: float, predicted_usage: float) -> float:
        """Calculate scaling pressure (urgency of scaling need)."""
        
        # Higher pressure for higher utilization
        usage_pressure = (current_usage / 100.0) ** 2
        
        # Higher pressure if prediction shows increasing load
        trend_pressure = max(0, predicted_usage - current_usage) / 100.0
        
        # Combine pressures
        total_pressure = usage_pressure + trend_pressure * 0.5
        
        return min(total_pressure, 1.0)
        
    def _execute_scaling(self, resource_type: ResourceType, direction: ScalingDirection, metric: ScalingMetric):
        """Execute scaling operation."""
        
        start_time = time.time()
        current_capacity = self.current_capacity[resource_type]
        limits = self.resource_limits[resource_type]
        policy = self.scaling_policies[resource_type]
        
        # Calculate new capacity
        if direction == ScalingDirection.UP:
            scale_factor = policy.get("scale_factor", 1.5)
            new_capacity = min(int(current_capacity * scale_factor), limits["max"])
        else:  # ScalingDirection.DOWN
            scale_factor = policy.get("scale_factor", 1.5)
            new_capacity = max(int(current_capacity / scale_factor), limits["min"])
            
        if new_capacity == current_capacity:
            return  # No change needed
            
        # Execute the scaling
        success = self._apply_scaling(resource_type, new_capacity)
        
        if success:
            self.current_capacity[resource_type] = new_capacity
            
            # Set cool-down period
            cool_down_minutes = policy.get("cool_down_minutes", 5)
            self.cool_down_periods[resource_type] = datetime.now() + timedelta(minutes=cool_down_minutes)
            
        # Record scaling event
        duration = time.time() - start_time
        event = ScalingEvent(
            timestamp=datetime.now(),
            resource_type=resource_type,
            direction=direction,
            from_capacity=current_capacity,
            to_capacity=new_capacity if success else current_capacity,
            trigger_metric=metric.current_usage,
            success=success,
            duration=duration
        )
        
        self.scaling_history.append(event)
        
        # Trim scaling history
        if len(self.scaling_history) > 1000:
            self.scaling_history = self.scaling_history[-500:]
            
        logger.info(f"Scaling {resource_type.value} {direction.value} from {current_capacity} to {new_capacity}, success: {success}")
        
    def _apply_scaling(self, resource_type: ResourceType, new_capacity: int) -> bool:
        """Apply the scaling change to actual resources."""
        
        try:
            if resource_type == ResourceType.THREADS:
                if self.thread_pool:
                    self.thread_pool.shutdown(wait=False)
                self.thread_pool = ThreadPoolExecutor(max_workers=new_capacity)
                return True
                
            elif resource_type == ResourceType.PROCESSES:
                if self.process_pool:
                    self.process_pool.shutdown(wait=False)
                self.process_pool = ProcessPoolExecutor(max_workers=new_capacity)
                return True
                
            # For other resource types, this would integrate with actual resource management
            # For now, we just update the capacity tracking
            return True
            
        except Exception as e:
            logger.error(f"Failed to apply scaling for {resource_type.value}: {e}")
            return False
            
    def get_scaling_recommendations(self) -> List[Dict[str, Any]]:
        """Get scaling recommendations based on current analysis."""
        
        recommendations = []
        
        for resource_type in ResourceType:
            if resource_type in self.current_capacity:
                current_capacity = self.current_capacity[resource_type]
                limits = self.resource_limits[resource_type]
                
                # Get recent scaling events for this resource
                recent_events = [
                    event for event in self.scaling_history[-10:]
                    if event.resource_type == resource_type and
                    event.timestamp > datetime.now() - timedelta(hours=1)
                ]
                
                # Predict optimal capacity
                predicted_usage, confidence = self.load_predictor.predict_load(15)
                optimal_capacity = self._calculate_optimal_capacity(resource_type, predicted_usage)
                
                recommendation = {
                    "resource_type": resource_type.value,
                    "current_capacity": current_capacity,
                    "optimal_capacity": optimal_capacity,
                    "min_capacity": limits["min"],
                    "max_capacity": limits["max"],
                    "predicted_usage": predicted_usage,
                    "confidence": confidence,
                    "recent_scaling_events": len(recent_events),
                    "recommendation": self._generate_recommendation_text(resource_type, current_capacity, optimal_capacity)
                }
                
                recommendations.append(recommendation)
                
        return recommendations
        
    def _calculate_optimal_capacity(self, resource_type: ResourceType, predicted_usage: float) -> int:
        """Calculate optimal capacity for predicted usage."""
        
        target_utilization = 0.7  # Target 70% utilization
        current_capacity = self.current_capacity[resource_type]
        
        if predicted_usage > 0:
            # Calculate capacity needed to achieve target utilization
            optimal_capacity = int((predicted_usage / target_utilization) * current_capacity / 100.0)
        else:
            optimal_capacity = current_capacity
            
        # Respect limits
        limits = self.resource_limits[resource_type]
        optimal_capacity = max(limits["min"], min(optimal_capacity, limits["max"]))
        
        return optimal_capacity
        
    def _generate_recommendation_text(self, resource_type: ResourceType, current: int, optimal: int) -> str:
        """Generate human-readable recommendation text."""
        
        if optimal > current * 1.2:
            return f"Consider scaling up {resource_type.value} to handle increased load"
        elif optimal < current * 0.8:
            return f"Consider scaling down {resource_type.value} to save resources"
        else:
            return f"Current {resource_type.value} capacity is optimal"
            
    def _start_monitoring(self):
        """Start background monitoring thread."""
        
        def monitoring_loop():
            while True:
                try:
                    time.sleep(30)  # Monitor every 30 seconds
                    self._background_monitoring()
                except Exception as e:
                    logger.error(f"Monitoring error: {e}")
                    
        monitoring_thread = threading.Thread(target=monitoring_loop, daemon=True)
        monitoring_thread.start()
        
    def _background_monitoring(self):
        """Background monitoring and optimization."""
        
        # Simulate current resource usage for demonstration
        import random
        
        # Simulate varying load patterns
        current_hour = datetime.now().hour
        base_usage = 30 + 20 * math.sin(current_hour * math.pi / 12)  # Daily pattern
        noise = random.gauss(0, 10)
        simulated_usage = max(10, min(95, base_usage + noise))
        
        # Record usage for different resource types
        self.record_resource_usage(ResourceType.CPU, simulated_usage)
        self.record_resource_usage(ResourceType.THREADS, simulated_usage * 0.8)
        self.record_resource_usage(ResourceType.MEMORY, simulated_usage * 1.1)
        
    def get_scaling_status(self) -> Dict[str, Any]:
        """Get comprehensive scaling system status."""
        
        recent_events = [
            event for event in self.scaling_history
            if event.timestamp > datetime.now() - timedelta(hours=1)
        ]
        
        successful_events = [event for event in recent_events if event.success]
        
        return {
            "current_capacity": {rt.value: cap for rt, cap in self.current_capacity.items()},
            "resource_limits": {
                rt.value: limits for rt, limits in self.resource_limits.items()
            },
            "recent_scaling_events": len(recent_events),
            "successful_scaling_events": len(successful_events),
            "success_rate": len(successful_events) / max(len(recent_events), 1),
            "active_cool_downs": len([
                rt for rt, cooldown in self.cool_down_periods.items()
                if datetime.now() < cooldown
            ]),
            "load_prediction_confidence": self.load_predictor.predict_load(5)[1],
            "total_scaling_history": len(self.scaling_history)
        }

# Global scaler instance
intelligent_scaler = IntelligentScaler()

def auto_scale_resource(resource_type: ResourceType):
    """Decorator to automatically scale resources for a function."""
    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                # Record performance as resource usage indicator
                usage_percent = min(95, duration * 100)  # Convert to percentage
                intelligent_scaler.record_resource_usage(resource_type, usage_percent)
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                # High usage for errors to trigger scaling
                intelligent_scaler.record_resource_usage(resource_type, 90.0)
                raise e
                
        wrapper.__name__ = f"auto_scaled_{func.__name__}"
        return wrapper
    return decorator

if __name__ == "__main__":
    # Example usage and testing
    
    @auto_scale_resource(ResourceType.THREADS)
    def example_workload(workload_size: int):
        """Example workload function."""
        time.sleep(workload_size * 0.01)  # Simulate work
        return f"Processed workload of size {workload_size}"
        
    # Test scaling system
    print("Testing Intelligent Scaling System...")
    
    # Simulate varying workloads
    for i in range(20):
        workload_size = random.randint(1, 100)
        result = example_workload(workload_size)
        time.sleep(0.1)
        
    # Get recommendations
    recommendations = intelligent_scaler.get_scaling_recommendations()
    print(f"Scaling Recommendations: {json.dumps(recommendations, indent=2)}")
    
    # Get status
    status = intelligent_scaler.get_scaling_status()
    print(f"Scaling Status: {json.dumps(status, indent=2)}")
    
    time.sleep(2)  # Let background monitoring run