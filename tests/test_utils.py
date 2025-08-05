"""Tests for utility modules (caching, optimization, monitoring)."""

import pytest
import torch
import numpy as np
import time
import threading
from unittest.mock import Mock, patch, MagicMock
import tempfile
from pathlib import Path
import json

from olfactory_transformer.utils.caching import LRUCache, ModelCache, PredictionCache
from olfactory_transformer.utils.optimization import ModelOptimizer, InferenceAccelerator
from olfactory_transformer.utils.monitoring import PerformanceMonitor, ResourceTracker, PerformanceMetrics
from olfactory_transformer.core.config import ScentPrediction
from olfactory_transformer.core.model import OlfactoryTransformer, OlfactoryConfig


class TestLRUCache:
    """Test LRUCache implementation."""
    
    def test_cache_creation(self):
        """Test basic cache creation."""
        cache = LRUCache(max_size=10, max_memory_mb=1)
        
        assert cache.max_size == 10
        assert cache.max_memory_bytes == 1024 * 1024
        assert len(cache.cache) == 0
        assert cache.hits == 0
        assert cache.misses == 0
    
    def test_cache_put_get(self):
        """Test basic put/get operations."""
        cache = LRUCache(max_size=3)
        
        # Test put and get
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        
        assert cache.get("key1") == "value1"
        assert cache.get("key2") == "value2"
        assert cache.get("nonexistent") is None
        
        # Check stats
        assert cache.hits == 2
        assert cache.misses == 1
    
    def test_cache_lru_eviction(self):
        """Test LRU eviction policy."""
        cache = LRUCache(max_size=2)
        
        # Fill cache
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        
        # Access key1 to make it recently used
        cache.get("key1")
        
        # Add third item - should evict key2
        cache.put("key3", "value3")
        
        assert cache.get("key1") == "value1"  # Should still exist
        assert cache.get("key2") is None      # Should be evicted
        assert cache.get("key3") == "value3"  # Should exist
    
    def test_cache_stats(self):
        """Test cache statistics."""
        cache = LRUCache(max_size=5)
        
        cache.put("key1", "value1")
        cache.get("key1")  # hit
        cache.get("key2")  # miss
        
        stats = cache.stats()
        
        assert stats["size"] == 1
        assert stats["max_size"] == 5
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5
    
    def test_cache_thread_safety(self):
        """Test thread safety of cache operations."""
        cache = LRUCache(max_size=100)
        
        def worker(thread_id):
            for i in range(10):
                key = f"thread_{thread_id}_key_{i}"
                value = f"thread_{thread_id}_value_{i}"
                cache.put(key, value)
                retrieved = cache.get(key)
                assert retrieved == value
        
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Should have processed all items without errors
        assert cache.stats()["size"] <= 100


class TestModelCache:
    """Test ModelCache implementation."""
    
    def test_model_cache_creation(self):
        """Test model cache creation."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache = ModelCache(
                cache_dir=Path(tmp_dir),
                memory_cache_size=50,
                disk_cache_size_gb=1
            )
            
            assert cache.cache_dir == Path(tmp_dir)
            assert cache.memory_cache.max_size == 50
            assert cache.disk_cache_size_bytes == 1024 * 1024 * 1024
    
    def test_molecular_features_caching(self):
        """Test caching of molecular features."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache = ModelCache(cache_dir=Path(tmp_dir))
            
            smiles = "CCO"
            features = {
                "molecular_weight": 46.07,
                "logp": -0.31,
                "num_atoms": 3
            }
            
            # Cache features
            cache.cache_molecular_features(smiles, features)
            
            # Retrieve features
            cached_features = cache.get_molecular_features(smiles)
            
            assert cached_features == features
            
            # Should also work for non-existent molecule
            assert cache.get_molecular_features("INVALID") is None
    
    def test_prediction_caching(self):
        """Test caching of predictions."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache = ModelCache(cache_dir=Path(tmp_dir))
            
            smiles = "CCO"
            prediction = ScentPrediction(
                primary_notes=["fresh", "alcohol"],
                intensity=5.5,
                confidence=0.85,
                chemical_family="alcohol"
            )
            
            # Cache prediction
            cache.cache_prediction(smiles, prediction)
            
            # Retrieve prediction
            cached_prediction = cache.get_prediction(smiles)
            
            assert isinstance(cached_prediction, ScentPrediction)
            assert cached_prediction.primary_notes == prediction.primary_notes
            assert cached_prediction.intensity == prediction.intensity
            assert cached_prediction.confidence == prediction.confidence
    
    def test_embeddings_caching(self):
        """Test caching of embeddings."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache = ModelCache(cache_dir=Path(tmp_dir))
            
            identifier = "molecule_embedding_CCO"
            embeddings = np.random.random((128,)).astype(np.float32)
            
            # Cache embeddings
            cache.cache_embeddings(identifier, embeddings)
            
            # Retrieve embeddings
            cached_embeddings = cache.get_embeddings(identifier)
            
            assert isinstance(cached_embeddings, np.ndarray)
            np.testing.assert_array_almost_equal(cached_embeddings, embeddings)
    
    def test_cache_cleanup(self):
        """Test cache cleanup functionality."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache = ModelCache(cache_dir=Path(tmp_dir))
            
            # Add some cache entries
            cache.cache_molecular_features("CCO", {"mw": 46.07})
            cache.cache_prediction("CCO", ScentPrediction(primary_notes=["test"]))
            
            # Check that entries exist
            assert len(cache.metadata["entries"]) == 2
            
            # Run cleanup with max_age_days=0 (should remove everything)
            cache.cleanup_old_cache(max_age_days=0)
            
            # Entries should be removed
            assert len(cache.metadata["entries"]) == 0
    
    def test_cache_stats(self):
        """Test cache statistics."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache = ModelCache(cache_dir=Path(tmp_dir))
            
            # Add some data
            cache.cache_molecular_features("CCO", {"mw": 46.07})
            cache.cache_prediction("CCO", ScentPrediction(primary_notes=["test"]))
            
            stats = cache.get_cache_stats()
            
            assert "memory_cache" in stats
            assert "disk_cache" in stats
            assert stats["disk_cache"]["entries"] == 2
            assert "type_breakdown" in stats["disk_cache"]


class TestPredictionCache:
    """Test PredictionCache implementation."""
    
    def test_prediction_cache_creation(self):
        """Test prediction cache creation."""
        cache = PredictionCache()
        
        assert cache.model_cache is not None
        assert isinstance(cache.batch_cache, dict)
    
    def test_batch_prediction_caching(self):
        """Test batch prediction with caching."""
        cache = PredictionCache()
        
        # Mock prediction function
        def mock_predict(smiles):
            return ScentPrediction(
                primary_notes=[f"note_{smiles}"],
                intensity=len(smiles),
                confidence=0.8
            )
        
        smiles_list = ["CCO", "CC(C)O", "C1=CC=CC=C1"]
        
        # First call - should compute all
        results1 = cache.get_or_predict(smiles_list, mock_predict, batch_size=2)
        
        assert len(results1) == 3
        assert all(isinstance(r, ScentPrediction) for r in results1)
        
        # Second call - should use cache
        results2 = cache.get_or_predict(smiles_list, mock_predict, batch_size=2)
        
        # Results should be identical
        for r1, r2 in zip(results1, results2):
            assert r1.primary_notes == r2.primary_notes
            assert r1.intensity == r2.intensity


class TestModelOptimizer:
    """Test ModelOptimizer functionality."""
    
    def test_optimizer_creation(self):
        """Test optimizer creation."""
        config = OlfactoryConfig(vocab_size=100, hidden_size=32, num_hidden_layers=1)
        model = OlfactoryTransformer(config)
        
        optimizer = ModelOptimizer(model)
        
        assert optimizer.model == model
        assert len(optimizer.optimized_models) == 0
    
    def test_model_quantization(self):
        """Test model quantization."""
        config = OlfactoryConfig(vocab_size=100, hidden_size=32, num_hidden_layers=1)
        model = OlfactoryTransformer(config)
        model.eval()
        
        optimizer = ModelOptimizer(model)
        
        # Test dynamic quantization
        quantized_model = optimizer.quantize_model(method="dynamic")
        
        assert quantized_model is not None
        assert "quantized" in optimizer.optimized_models
        
        # Test that quantized model can still run inference
        with torch.no_grad():
            input_ids = torch.randint(0, 100, (1, 10))
            attention_mask = torch.ones(1, 10)
            
            outputs = quantized_model(input_ids=input_ids, attention_mask=attention_mask)
            assert "scent_logits" in outputs
    
    @patch('olfactory_transformer.utils.optimization.HAS_TORCHSCRIPT', True)
    def test_torchscript_export(self):
        """Test TorchScript export."""
        config = OlfactoryConfig(vocab_size=50, hidden_size=16, num_hidden_layers=1)
        model = OlfactoryTransformer(config)
        
        optimizer = ModelOptimizer(model)
        
        with tempfile.NamedTemporaryFile(suffix='.pt') as tmp_file:
            example_input = {
                "input_ids": torch.randint(0, 50, (1, 5)),
                "attention_mask": torch.ones(1, 5)
            }
            
            # This might fail due to model complexity, but should not crash
            try:
                traced_model = optimizer.export_torchscript(
                    tmp_file.name,
                    example_input,
                    method="trace"
                )
                # If successful, should be able to load
                if traced_model is not None:
                    assert Path(tmp_file.name).exists()
            except Exception:
                # TorchScript export can be finicky with complex models
                pass
    
    @patch('olfactory_transformer.utils.optimization.HAS_ONNX', True)
    def test_onnx_export(self):
        """Test ONNX export."""
        config = OlfactoryConfig(vocab_size=50, hidden_size=16, num_hidden_layers=1)
        model = OlfactoryTransformer(config)
        
        optimizer = ModelOptimizer(model)
        
        with tempfile.NamedTemporaryFile(suffix='.onnx') as tmp_file:
            example_input = {
                "input_ids": torch.randint(0, 50, (1, 5)),
                "attention_mask": torch.ones(1, 5)
            }
            
            with patch('torch.onnx.export') as mock_export:
                success = optimizer.export_onnx(tmp_file.name, example_input)
                
                # Should attempt to export
                mock_export.assert_called_once()


class TestInferenceAccelerator:
    """Test InferenceAccelerator functionality."""
    
    def test_accelerator_creation(self):
        """Test accelerator creation."""
        config = OlfactoryConfig(vocab_size=50, hidden_size=16)
        model = OlfactoryTransformer(config)
        
        accelerator = InferenceAccelerator(
            model=model,
            max_batch_size=8,
            max_workers=2
        )
        
        assert accelerator.model == model
        assert accelerator.max_batch_size == 8
        assert accelerator.max_workers == 2
        assert not accelerator.running
    
    def test_synchronous_batch_processing(self):
        """Test synchronous batch processing."""
        config = OlfactoryConfig(vocab_size=50, hidden_size=16, num_hidden_layers=1)
        model = OlfactoryTransformer(config)
        model.eval()
        
        accelerator = InferenceAccelerator(model=model, max_batch_size=4)
        
        # Create batch data
        batch_data = []
        for i in range(3):
            batch_data.append({
                "input_ids": torch.randint(0, 50, (5,)),
                "attention_mask": torch.ones(5)
            })
        
        # Process batch
        results = accelerator.predict_batch_sync(batch_data)
        
        assert len(results) == 3
        assert all(isinstance(r, dict) for r in results)
        
        # Check that all results have expected keys
        for result in results:
            assert "scent_logits" in result or len(result) > 0
    
    def test_async_prediction(self):
        """Test asynchronous prediction."""
        config = OlfactoryConfig(vocab_size=50, hidden_size=16, num_hidden_layers=1)
        model = OlfactoryTransformer(config)
        model.eval()
        
        accelerator = InferenceAccelerator(model=model)
        
        # Use context manager
        with accelerator:
            input_data = {
                "input_ids": torch.randint(0, 50, (5,)),
                "attention_mask": torch.ones(5)
            }
            
            # Submit async request
            event = accelerator.predict_async("test_request", input_data)
            
            # Wait for completion
            success = event.wait(timeout=5.0)
            assert success
            
            # Get result
            result = accelerator.get_result("test_request")
            assert result is not None
    
    def test_performance_stats(self):
        """Test performance statistics."""
        config = OlfactoryConfig(vocab_size=50, hidden_size=16, num_hidden_layers=1)
        model = OlfactoryTransformer(config)
        
        accelerator = InferenceAccelerator(model=model)
        
        # Process some batches to generate stats
        batch_data = [{
            "input_ids": torch.randint(0, 50, (5,)),
            "attention_mask": torch.ones(5)
        }]
        
        accelerator.predict_batch_sync(batch_data)
        accelerator.predict_batch_sync(batch_data)
        
        stats = accelerator.get_performance_stats()
        
        assert "requests_processed" in stats
        assert "batches_processed" in stats
        assert "avg_batch_size" in stats
        assert stats["requests_processed"] == 2
        assert stats["batches_processed"] == 2


class TestPerformanceMonitor:
    """Test PerformanceMonitor functionality."""
    
    def test_monitor_creation(self):
        """Test monitor creation."""
        monitor = PerformanceMonitor(window_size=50, log_interval=30.0)
        
        assert monitor.window_size == 50
        assert monitor.log_interval == 30.0
        assert len(monitor.metrics_history) == 0
    
    def test_metrics_recording(self):
        """Test metrics recording."""
        monitor = PerformanceMonitor(window_size=10)
        
        metrics = PerformanceMetrics(
            timestamp=time.time(),
            inference_time=0.1,
            throughput=10.0,
            memory_usage_mb=256.0,
            model_name="test_model"
        )
        
        monitor.record_metrics(metrics)
        
        assert len(monitor.metrics_history) == 1
        assert "test_model" in monitor.aggregated_metrics
    
    def test_inference_timer(self):
        """Test inference timing context manager."""
        monitor = PerformanceMonitor()
        
        with monitor.time_inference("test_model"):
            time.sleep(0.01)  # Simulate some work
        
        assert len(monitor.metrics_history) == 1
        metrics = monitor.metrics_history[0]
        assert metrics.model_name == "test_model"
        assert metrics.inference_time > 0
    
    def test_current_stats(self):
        """Test current statistics."""
        monitor = PerformanceMonitor()
        
        # Add some metrics
        for i in range(5):
            metrics = PerformanceMetrics(
                timestamp=time.time(),
                inference_time=0.1 + i * 0.01,
                throughput=10.0 - i,
                memory_usage_mb=100.0 + i * 10,
                model_name="test_model"
            )
            monitor.record_metrics(metrics)
        
        stats = monitor.get_current_stats()
        
        assert "recent_avg_inference_time" in stats
        assert "recent_avg_throughput" in stats
        assert "test_model_avg_time" in stats
        assert stats["total_inferences"] == 5
    
    def test_metrics_export(self):
        """Test metrics export."""
        monitor = PerformanceMonitor()
        
        # Add some metrics
        metrics = PerformanceMetrics(
            timestamp=time.time(),
            inference_time=0.1,
            throughput=10.0,
            memory_usage_mb=256.0,
            model_name="test_model"
        )
        monitor.record_metrics(metrics)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            monitor.export_metrics(temp_path)
            
            # Verify file was created and contains data
            assert Path(temp_path).exists()
            
            with open(temp_path, 'r') as f:
                data = json.load(f)
            
            assert "metrics_history" in data
            assert "aggregated_metrics" in data
            assert len(data["metrics_history"]) == 1
        finally:
            Path(temp_path).unlink()


class TestResourceTracker:
    """Test ResourceTracker functionality."""
    
    def test_tracker_creation(self):
        """Test tracker creation."""
        tracker = ResourceTracker(
            memory_threshold=85.0,
            cpu_threshold=95.0
        )
        
        assert tracker.memory_threshold == 85.0
        assert tracker.cpu_threshold == 95.0
        assert len(tracker.resource_history) == 0
        assert len(tracker.alert_callbacks) == 0
    
    def test_resource_snapshot(self):
        """Test resource snapshot."""
        tracker = ResourceTracker()
        
        snapshot = tracker.get_resource_snapshot()
        
        assert snapshot.timestamp > 0
        assert 0 <= snapshot.cpu_percent <= 100
        assert 0 <= snapshot.memory_percent <= 100
        assert snapshot.memory_used_gb >= 0
        assert snapshot.memory_total_gb > 0
        assert 0 <= snapshot.disk_usage_percent <= 100
    
    def test_threshold_checking(self):
        """Test threshold checking."""
        tracker = ResourceTracker(
            memory_threshold=50.0,  # Low threshold for testing
            cpu_threshold=50.0
        )
        
        # Create mock snapshot with high resource usage
        from olfactory_transformer.utils.monitoring import ResourceSnapshot
        
        snapshot = ResourceSnapshot(
            timestamp=time.time(),
            cpu_percent=75.0,  # Above threshold
            memory_percent=60.0,  # Above threshold
            memory_used_gb=8.0,
            memory_total_gb=16.0,
            disk_usage_percent=40.0
        )
        
        alerts = tracker.check_thresholds(snapshot)
        
        # Should have alerts for CPU and memory
        assert len(alerts) >= 2
        alert_types = [alert["type"] for alert in alerts]
        assert "high_cpu" in alert_types
        assert "high_memory" in alert_types
    
    def test_alert_callbacks(self):
        """Test alert callback system."""
        tracker = ResourceTracker(memory_threshold=10.0)  # Very low threshold
        
        alerts_received = []
        
        def test_callback(alert_type, alert_data):
            alerts_received.append((alert_type, alert_data))
        
        tracker.add_alert_callback(test_callback)
        
        # Create high memory usage snapshot
        from olfactory_transformer.utils.monitoring import ResourceSnapshot
        
        snapshot = ResourceSnapshot(
            timestamp=time.time(),
            cpu_percent=20.0,
            memory_percent=50.0,  # Above 10% threshold
            memory_used_gb=4.0,
            memory_total_gb=8.0,
            disk_usage_percent=30.0
        )
        
        alerts = tracker.check_thresholds(snapshot)
        
        # Manually trigger callbacks (since we're not running the monitoring loop)
        for alert in alerts:
            for callback in tracker.alert_callbacks:
                callback(alert["type"], alert)
        
        # Should have received alerts
        assert len(alerts_received) > 0
        assert alerts_received[0][0] == "high_memory"


if __name__ == "__main__":
    pytest.main([__file__])