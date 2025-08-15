"""
Autonomous Validation Module for Olfactory AI Research.

Implements automated validation pipelines including:
- Real-time model monitoring
- Automated A/B testing
- Continuous integration for research
- Self-healing experiment pipelines
"""

import numpy as np
import logging
import json
import asyncio
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import threading
import queue
import time

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@dataclass
class ValidationConfig:
    """Configuration for autonomous validation."""
    validation_interval: int = 3600  # seconds
    drift_threshold: float = 0.1
    performance_threshold: float = 0.05
    min_samples_for_validation: int = 100
    enable_auto_retrain: bool = True
    enable_a_b_testing: bool = True
    confidence_level: float = 0.95
    alert_channels: List[str] = None


@dataclass
class ValidationResult:
    """Result from validation check."""
    timestamp: str
    model_version: str
    metrics: Dict[str, float]
    drift_detected: bool
    performance_degradation: bool
    recommendations: List[str]
    alert_level: str  # 'info', 'warning', 'critical'
    raw_data: Dict[str, Any]


class ModelMonitor:
    """Real-time model performance monitoring."""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        self.metrics_history = []
        self.baseline_metrics = None
        self.is_monitoring = False
        self.monitor_thread = None
        
    def start_monitoring(self, model: Any, validation_data: Dict[str, np.ndarray]) -> None:
        """Start autonomous monitoring."""
        if self.is_monitoring:
            logging.warning("Monitoring already active")
            return
            
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(model, validation_data),
            daemon=True
        )
        self.monitor_thread.start()
        logging.info("Started autonomous model monitoring")
    
    def stop_monitoring(self) -> None:
        """Stop monitoring."""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logging.info("Stopped model monitoring")
    
    def _monitoring_loop(self, model: Any, validation_data: Dict[str, np.ndarray]) -> None:
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                # Perform validation check
                result = self._validate_model(model, validation_data)
                
                # Store metrics
                self.metrics_history.append(result)
                
                # Check for alerts
                if result.alert_level in ['warning', 'critical']:
                    self._send_alert(result)
                
                # Auto-remediation
                if result.performance_degradation and self.config.enable_auto_retrain:
                    self._trigger_auto_retrain(model, validation_data, result)
                
                # Sleep until next check
                time.sleep(self.config.validation_interval)
                
            except Exception as e:
                logging.error(f"Monitoring error: {e}")
                time.sleep(60)  # Wait 1 minute before retry
    
    def _validate_model(self, model: Any, validation_data: Dict[str, np.ndarray]) -> ValidationResult:
        """Validate model performance."""
        timestamp = datetime.now().isoformat()
        
        # Generate predictions
        try:
            if hasattr(model, 'predict'):
                predictions = model.predict(validation_data['X'])
            else:
                predictions = np.random.random(len(validation_data['y']))
        except Exception as e:
            logging.error(f"Prediction failed during validation: {e}")
            predictions = np.zeros(len(validation_data['y']))
        
        # Compute current metrics
        current_metrics = self._compute_validation_metrics(validation_data['y'], predictions)
        
        # Initialize baseline if not set
        if self.baseline_metrics is None:
            self.baseline_metrics = current_metrics.copy()
        
        # Check for drift and degradation
        drift_detected = self._detect_drift(current_metrics, self.baseline_metrics)
        performance_degradation = self._detect_performance_degradation(current_metrics, self.baseline_metrics)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(drift_detected, performance_degradation, current_metrics)
        
        # Determine alert level
        if performance_degradation:
            alert_level = 'critical'
        elif drift_detected:
            alert_level = 'warning'
        else:
            alert_level = 'info'
        
        return ValidationResult(
            timestamp=timestamp,
            model_version="current",
            metrics=current_metrics,
            drift_detected=drift_detected,
            performance_degradation=performance_degradation,
            recommendations=recommendations,
            alert_level=alert_level,
            raw_data={'predictions': predictions.tolist()[:100]}  # Sample of predictions
        )
    
    def _compute_validation_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Compute validation metrics."""
        # Handle invalid predictions
        valid_mask = ~(np.isnan(y_pred) | np.isinf(y_pred))
        if np.sum(valid_mask) == 0:
            return {'mse': float('inf'), 'mae': float('inf'), 'r2': -float('inf')}
        
        y_true_valid = y_true[valid_mask]
        y_pred_valid = y_pred[valid_mask]
        
        mse = np.mean((y_true_valid - y_pred_valid) ** 2)
        mae = np.mean(np.abs(y_true_valid - y_pred_valid))
        
        # R² score
        ss_res = np.sum((y_true_valid - y_pred_valid) ** 2)
        ss_tot = np.sum((y_true_valid - np.mean(y_true_valid)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-10))
        
        return {'mse': mse, 'mae': mae, 'r2': r2}
    
    def _detect_drift(self, current: Dict[str, float], baseline: Dict[str, float]) -> bool:
        """Detect model drift."""
        for metric in ['mse', 'mae']:
            if metric in current and metric in baseline:
                relative_change = abs(current[metric] - baseline[metric]) / (baseline[metric] + 1e-10)
                if relative_change > self.config.drift_threshold:
                    return True
        return False
    
    def _detect_performance_degradation(self, current: Dict[str, float], baseline: Dict[str, float]) -> bool:
        """Detect performance degradation."""
        if 'r2' in current and 'r2' in baseline:
            degradation = baseline['r2'] - current['r2']
            return degradation > self.config.performance_threshold
        return False
    
    def _generate_recommendations(self, drift_detected: bool, performance_degradation: bool, 
                                current_metrics: Dict[str, float]) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        if performance_degradation:
            recommendations.append("Consider retraining the model with recent data")
            recommendations.append("Review data quality and preprocessing pipeline")
        
        if drift_detected:
            recommendations.append("Investigate potential data distribution changes")
            recommendations.append("Update feature engineering pipeline")
        
        if current_metrics.get('mse', 0) > 1.0:
            recommendations.append("High prediction error detected - review model architecture")
        
        if not recommendations:
            recommendations.append("Model performance is stable")
        
        return recommendations
    
    def _send_alert(self, result: ValidationResult) -> None:
        """Send alert for validation issues."""
        alert_message = f"""
        Model Validation Alert - {result.alert_level.upper()}
        
        Timestamp: {result.timestamp}
        Drift Detected: {result.drift_detected}
        Performance Degradation: {result.performance_degradation}
        
        Current Metrics:
        - MSE: {result.metrics.get('mse', 'N/A'):.4f}
        - MAE: {result.metrics.get('mae', 'N/A'):.4f}
        - R²: {result.metrics.get('r2', 'N/A'):.4f}
        
        Recommendations:
        {chr(10).join(f'- {rec}' for rec in result.recommendations)}
        """
        
        logging.warning(alert_message)
        
        # In production, would send to actual alert channels
        # (email, Slack, PagerDuty, etc.)
    
    def _trigger_auto_retrain(self, model: Any, validation_data: Dict[str, np.ndarray], 
                            result: ValidationResult) -> None:
        """Trigger automatic model retraining."""
        logging.info("Triggering automatic model retraining due to performance degradation")
        
        try:
            if hasattr(model, 'train'):
                # Simple retraining with validation data
                model.train(validation_data['X'], validation_data['y'])
                logging.info("Automatic retraining completed")
                
                # Update baseline metrics after retraining
                new_predictions = model.predict(validation_data['X'])
                self.baseline_metrics = self._compute_validation_metrics(validation_data['y'], new_predictions)
                
            else:
                logging.warning("Model does not support retraining")
                
        except Exception as e:
            logging.error(f"Automatic retraining failed: {e}")


class ABTestingFramework:
    """Automated A/B testing for model improvements."""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        self.experiments = {}
        self.traffic_split = 0.5  # 50/50 split by default
        
    def create_experiment(self, experiment_id: str, control_model: Any, 
                         variant_model: Any, traffic_split: float = 0.5) -> None:
        """Create A/B test experiment."""
        self.experiments[experiment_id] = {
            'control': control_model,
            'variant': variant_model,
            'traffic_split': traffic_split,
            'control_metrics': [],
            'variant_metrics': [],
            'start_time': datetime.now(),
            'status': 'running'
        }
        logging.info(f"Created A/B test experiment: {experiment_id}")
    
    def route_prediction(self, experiment_id: str, input_data: np.ndarray, 
                        user_id: str = None) -> Tuple[Any, str]:
        """Route prediction request to control or variant."""
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        experiment = self.experiments[experiment_id]
        
        # Determine routing (simplified hash-based routing)
        if user_id:
            route_hash = hash(user_id) % 100
        else:
            route_hash = np.random.randint(0, 100)
        
        if route_hash < experiment['traffic_split'] * 100:
            model = experiment['variant']
            variant = 'variant'
        else:
            model = experiment['control']
            variant = 'control'
        
        # Make prediction
        prediction = model.predict(input_data) if hasattr(model, 'predict') else np.random.random(len(input_data))
        
        return prediction, variant
    
    def record_outcome(self, experiment_id: str, variant: str, 
                      prediction: np.ndarray, actual: np.ndarray) -> None:
        """Record experiment outcome."""
        if experiment_id not in self.experiments:
            return
        
        experiment = self.experiments[experiment_id]
        
        # Compute metrics
        metrics = self._compute_ab_metrics(actual, prediction)
        
        # Store in appropriate variant
        if variant == 'control':
            experiment['control_metrics'].append(metrics)
        else:
            experiment['variant_metrics'].append(metrics)
    
    def analyze_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """Analyze A/B test results."""
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        experiment = self.experiments[experiment_id]
        
        control_metrics = experiment['control_metrics']
        variant_metrics = experiment['variant_metrics']
        
        if len(control_metrics) < 10 or len(variant_metrics) < 10:
            return {'status': 'insufficient_data', 'message': 'Need more data for analysis'}
        
        # Aggregate metrics
        control_agg = self._aggregate_metrics(control_metrics)
        variant_agg = self._aggregate_metrics(variant_metrics)
        
        # Statistical significance test (simplified)
        significance_result = self._test_significance(control_metrics, variant_metrics)
        
        # Determine winner
        winner = self._determine_winner(control_agg, variant_agg, significance_result)
        
        return {
            'experiment_id': experiment_id,
            'status': 'completed',
            'control_metrics': control_agg,
            'variant_metrics': variant_agg,
            'statistical_significance': significance_result,
            'winner': winner,
            'recommendation': self._generate_ab_recommendation(winner, significance_result)
        }
    
    def _compute_ab_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Compute A/B test metrics."""
        valid_mask = ~(np.isnan(y_pred) | np.isinf(y_pred))
        if np.sum(valid_mask) == 0:
            return {'mse': float('inf'), 'mae': float('inf')}
        
        y_true_valid = y_true[valid_mask]
        y_pred_valid = y_pred[valid_mask]
        
        return {
            'mse': np.mean((y_true_valid - y_pred_valid) ** 2),
            'mae': np.mean(np.abs(y_true_valid - y_pred_valid))
        }
    
    def _aggregate_metrics(self, metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
        """Aggregate metrics across samples."""
        if not metrics_list:
            return {}
        
        aggregated = {}
        for key in metrics_list[0].keys():
            values = [m[key] for m in metrics_list if key in m]
            aggregated[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'count': len(values)
            }
        return aggregated
    
    def _test_significance(self, control_metrics: List[Dict[str, float]], 
                         variant_metrics: List[Dict[str, float]]) -> Dict[str, float]:
        """Test statistical significance between variants."""
        if not control_metrics or not variant_metrics:
            return {'significant': False, 'p_value': 1.0}
        
        # Use MSE for significance testing
        control_mse = [m['mse'] for m in control_metrics if 'mse' in m]
        variant_mse = [m['mse'] for m in variant_metrics if 'mse' in m]
        
        if len(control_mse) < 5 or len(variant_mse) < 5:
            return {'significant': False, 'p_value': 1.0}
        
        # Simplified t-test
        control_mean = np.mean(control_mse)
        variant_mean = np.mean(variant_mse)
        control_std = np.std(control_mse)
        variant_std = np.std(variant_mse)
        
        # Pooled standard error
        n1, n2 = len(control_mse), len(variant_mse)
        pooled_se = np.sqrt((control_std**2)/n1 + (variant_std**2)/n2)
        
        # t-statistic
        t_stat = abs(control_mean - variant_mean) / (pooled_se + 1e-10)
        
        # Approximate p-value (simplified)
        p_value = 2 * (1 - min(0.95, t_stat / 4))  # Very rough approximation
        
        return {
            'significant': p_value < 0.05,
            'p_value': p_value,
            't_statistic': t_stat
        }
    
    def _determine_winner(self, control_agg: Dict[str, Any], variant_agg: Dict[str, Any], 
                         significance: Dict[str, float]) -> str:
        """Determine experiment winner."""
        if not significance['significant']:
            return 'no_winner'
        
        control_mse = control_agg.get('mse', {}).get('mean', float('inf'))
        variant_mse = variant_agg.get('mse', {}).get('mean', float('inf'))
        
        return 'variant' if variant_mse < control_mse else 'control'
    
    def _generate_ab_recommendation(self, winner: str, significance: Dict[str, float]) -> str:
        """Generate recommendation based on A/B test results."""
        if winner == 'no_winner':
            return "No significant difference detected. Continue with current model."
        elif winner == 'variant':
            return "Variant model shows significant improvement. Consider full rollout."
        else:
            return "Control model performs better. Keep current implementation."


class ContinuousIntegration:
    """Continuous integration pipeline for research experiments."""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        self.pipeline_queue = queue.Queue()
        self.worker_thread = None
        self.is_running = False
    
    def start_pipeline(self) -> None:
        """Start CI pipeline."""
        if self.is_running:
            return
            
        self.is_running = True
        self.worker_thread = threading.Thread(target=self._pipeline_worker, daemon=True)
        self.worker_thread.start()
        logging.info("Started continuous integration pipeline")
    
    def stop_pipeline(self) -> None:
        """Stop CI pipeline."""
        self.is_running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5)
        logging.info("Stopped continuous integration pipeline")
    
    def submit_experiment(self, experiment_config: Dict[str, Any]) -> str:
        """Submit experiment to CI pipeline."""
        experiment_id = f"ci_exp_{int(time.time())}"
        self.pipeline_queue.put({
            'id': experiment_id,
            'config': experiment_config,
            'submitted_at': datetime.now()
        })
        logging.info(f"Submitted experiment to CI pipeline: {experiment_id}")
        return experiment_id
    
    def _pipeline_worker(self) -> None:
        """CI pipeline worker."""
        while self.is_running:
            try:
                # Get next experiment (with timeout)
                experiment = self.pipeline_queue.get(timeout=1)
                
                # Run experiment pipeline
                result = self._run_experiment_pipeline(experiment)
                
                # Log result
                logging.info(f"Completed CI experiment {experiment['id']}: {result['status']}")
                
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"CI pipeline error: {e}")
    
    def _run_experiment_pipeline(self, experiment: Dict[str, Any]) -> Dict[str, Any]:
        """Run complete experiment pipeline."""
        experiment_id = experiment['id']
        config = experiment['config']
        
        try:
            # Stage 1: Data validation
            data_validation = self._validate_data(config.get('data', {}))
            if not data_validation['valid']:
                return {'status': 'failed', 'stage': 'data_validation', 'error': data_validation['error']}
            
            # Stage 2: Model training
            model_result = self._train_model(config.get('model', {}))
            if not model_result['success']:
                return {'status': 'failed', 'stage': 'model_training', 'error': model_result['error']}
            
            # Stage 3: Model validation
            validation_result = self._validate_model_ci(model_result['model'], config.get('validation', {}))
            if not validation_result['passed']:
                return {'status': 'failed', 'stage': 'model_validation', 'error': validation_result['error']}
            
            # Stage 4: Performance benchmarking
            benchmark_result = self._benchmark_model(model_result['model'], config.get('benchmark', {}))
            
            return {
                'status': 'success',
                'experiment_id': experiment_id,
                'model_metrics': validation_result['metrics'],
                'benchmark_score': benchmark_result['score']
            }
            
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    def _validate_data(self, data_config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate experiment data."""
        # Simplified data validation
        required_fields = ['train_size', 'test_size', 'features']
        
        for field in required_fields:
            if field not in data_config:
                return {'valid': False, 'error': f'Missing required field: {field}'}
        
        if data_config['train_size'] < 100:
            return {'valid': False, 'error': 'Training data too small'}
        
        return {'valid': True}
    
    def _train_model(self, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """Train model in CI pipeline."""
        try:
            # Create dummy model for demonstration
            class DummyModel:
                def __init__(self):
                    self.trained = False
                
                def train(self, X, y):
                    self.trained = True
                    return self
                
                def predict(self, X):
                    return np.random.random(len(X))
            
            model = DummyModel()
            
            # Simulate training
            train_X = np.random.random((100, 10))
            train_y = np.random.random(100)
            model.train(train_X, train_y)
            
            return {'success': True, 'model': model}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _validate_model_ci(self, model: Any, validation_config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate model in CI pipeline."""
        try:
            # Generate test data
            test_X = np.random.random((50, 10))
            test_y = np.random.random(50)
            
            # Make predictions
            predictions = model.predict(test_X)
            
            # Compute metrics
            mse = np.mean((test_y - predictions) ** 2)
            mae = np.mean(np.abs(test_y - predictions))
            
            # Check thresholds
            max_mse = validation_config.get('max_mse', 1.0)
            max_mae = validation_config.get('max_mae', 0.8)
            
            if mse > max_mse or mae > max_mae:
                return {'passed': False, 'error': f'Model performance below threshold: MSE={mse:.3f}, MAE={mae:.3f}'}
            
            return {'passed': True, 'metrics': {'mse': mse, 'mae': mae}}
            
        except Exception as e:
            return {'passed': False, 'error': str(e)}
    
    def _benchmark_model(self, model: Any, benchmark_config: Dict[str, Any]) -> Dict[str, Any]:
        """Benchmark model performance."""
        try:
            # Simulate benchmarking
            benchmark_X = np.random.random((1000, 10))
            
            start_time = time.time()
            predictions = model.predict(benchmark_X)
            inference_time = time.time() - start_time
            
            # Compute benchmark score
            throughput = len(benchmark_X) / inference_time
            score = min(100, throughput / 10)  # Normalize to 0-100
            
            return {'score': score, 'throughput': throughput, 'inference_time': inference_time}
            
        except Exception as e:
            return {'score': 0, 'error': str(e)}


class AutonomousValidator:
    """Main autonomous validation orchestrator."""
    
    def __init__(self, config: ValidationConfig = None):
        self.config = config or ValidationConfig()
        self.monitor = ModelMonitor(self.config)
        self.ab_testing = ABTestingFramework(self.config)
        self.ci_pipeline = ContinuousIntegration(self.config)
        
    def start_autonomous_validation(self, model: Any, validation_data: Dict[str, np.ndarray]) -> None:
        """Start all autonomous validation components."""
        logging.info("Starting autonomous validation system")
        
        # Start monitoring
        self.monitor.start_monitoring(model, validation_data)
        
        # Start CI pipeline
        self.ci_pipeline.start_pipeline()
        
        logging.info("Autonomous validation system active")
    
    def stop_autonomous_validation(self) -> None:
        """Stop all validation components."""
        logging.info("Stopping autonomous validation system")
        
        self.monitor.stop_monitoring()
        self.ci_pipeline.stop_pipeline()
        
        logging.info("Autonomous validation system stopped")
    
    def create_ab_experiment(self, experiment_id: str, control_model: Any, 
                           variant_model: Any) -> None:
        """Create A/B test experiment."""
        self.ab_testing.create_experiment(experiment_id, control_model, variant_model)
    
    def submit_ci_experiment(self, experiment_config: Dict[str, Any]) -> str:
        """Submit experiment to CI pipeline."""
        return self.ci_pipeline.submit_experiment(experiment_config)
    
    def get_validation_status(self) -> Dict[str, Any]:
        """Get current validation system status."""
        return {
            'monitoring_active': self.monitor.is_monitoring,
            'ci_pipeline_active': self.ci_pipeline.is_running,
            'recent_validations': len(self.monitor.metrics_history),
            'active_ab_tests': len(self.ab_testing.experiments)
        }


def main():
    """Demonstrate autonomous validation system."""
    # Create configuration
    config = ValidationConfig(
        validation_interval=10,  # 10 seconds for demo
        drift_threshold=0.15,
        performance_threshold=0.1,
        enable_auto_retrain=True,
        enable_a_b_testing=True
    )
    
    # Create autonomous validator
    validator = AutonomousValidator(config)
    
    # Create dummy model and data
    class DemoModel:
        def __init__(self, noise_level=0.1):
            self.noise_level = noise_level
        
        def train(self, X, y):
            return self
        
        def predict(self, X):
            return np.random.normal(0.5, self.noise_level, len(X))
    
    model = DemoModel()
    validation_data = {
        'X': np.random.random((200, 10)),
        'y': np.random.random(200)
    }
    
    # Start autonomous validation
    validator.start_autonomous_validation(model, validation_data)
    
    # Let it run for a short demo period
    print("Running autonomous validation for 30 seconds...")
    time.sleep(30)
    
    # Check status
    status = validator.get_validation_status()
    print(f"Validation status: {status}")
    
    # Create A/B test
    control_model = DemoModel(noise_level=0.1)
    variant_model = DemoModel(noise_level=0.05)  # Better model
    validator.create_ab_experiment("demo_test", control_model, variant_model)
    
    # Submit CI experiment
    ci_config = {
        'data': {'train_size': 1000, 'test_size': 200, 'features': 10},
        'model': {'type': 'demo'},
        'validation': {'max_mse': 0.5, 'max_mae': 0.4},
        'benchmark': {'min_throughput': 100}
    }
    
    experiment_id = validator.submit_ci_experiment(ci_config)
    print(f"Submitted CI experiment: {experiment_id}")
    
    # Wait a bit more for CI to complete
    time.sleep(10)
    
    # Stop validation
    validator.stop_autonomous_validation()
    
    print("Autonomous validation demonstration completed")
    return validator


if __name__ == "__main__":
    main()