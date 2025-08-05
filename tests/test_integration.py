"""Integration tests for the complete olfactory transformer system."""

import pytest
import torch
import numpy as np
import tempfile
from pathlib import Path
import time
from unittest.mock import Mock, patch

from olfactory_transformer import (
    OlfactoryTransformer,
    MoleculeTokenizer,
    ENoseInterface,
    ScentDesigner,
    PerceptualEvaluator,
    OlfactoryTrainer
)
from olfactory_transformer.core.config import OlfactoryConfig, ScentPrediction
from olfactory_transformer.training.trainer import OlfactoryDataset, TrainingArguments
from olfactory_transformer.utils.caching import ModelCache
from olfactory_transformer.utils.optimization import InferenceAccelerator
from olfactory_transformer.utils.monitoring import PerformanceMonitor


class TestEndToEndPrediction:
    """Test complete prediction pipeline from SMILES to scent prediction."""
    
    def test_complete_prediction_pipeline(self):
        """Test the complete prediction pipeline."""
        # Create small model for testing
        config = OlfactoryConfig(
            vocab_size=200,
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            max_position_embeddings=50
        )
        
        # Create and build tokenizer
        tokenizer = MoleculeTokenizer(vocab_size=200)
        smiles_list = [
            "CCO",           # Ethanol
            "CC(C)O",        # Isopropanol
            "C1=CC=CC=C1",   # Benzene
            "CC(=O)OCC",     # Ethyl acetate
            "C1=CC=C(C=C1)C=O"  # Benzaldehyde
        ]
        tokenizer.build_vocab_from_smiles(smiles_list)
        
        # Create model
        model = OlfactoryTransformer(config)
        model.eval()
        
        # Test prediction
        test_smiles = "CCO"
        
        with torch.no_grad():
            prediction = model.predict_scent(test_smiles, tokenizer)
        
        # Verify prediction structure
        assert isinstance(prediction, ScentPrediction)
        assert len(prediction.primary_notes) > 0
        assert 0 <= prediction.intensity <= 10
        assert 0 <= prediction.confidence <= 1
        assert prediction.chemical_family in model.chemical_families
        assert prediction.ifra_category is not None
        
        # Test multiple molecules
        test_molecules = ["CCO", "CC(C)O", "C1=CC=CC=C1"]
        predictions = []
        
        for smiles in test_molecules:
            with torch.no_data():
                pred = model.predict_scent(smiles, tokenizer)
                predictions.append(pred)
        
        assert len(predictions) == 3
        assert all(isinstance(p, ScentPrediction) for p in predictions)
    
    def test_batch_prediction_efficiency(self):
        """Test batch prediction for efficiency."""
        config = OlfactoryConfig(
            vocab_size=100,
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=2
        )
        
        model = OlfactoryTransformer(config)
        accelerator = InferenceAccelerator(model, max_batch_size=4)
        
        # Create batch input
        batch_data = []
        for i in range(6):  # More than batch size
            batch_data.append({
                "input_ids": torch.randint(0, 100, (10,)),
                "attention_mask": torch.ones(10)
            })
        
        # Process batch
        start_time = time.time()
        results = accelerator.predict_batch_sync(batch_data)
        end_time = time.time()
        
        assert len(results) == 6
        assert all(isinstance(r, dict) for r in results)
        
        # Check performance stats
        stats = accelerator.get_performance_stats()
        assert stats["requests_processed"] == 6
        assert stats["batches_processed"] >= 1  # Should batch efficiently
        
        print(f"Batch processing took {end_time - start_time:.3f}s for {len(batch_data)} samples")


class TestSensorIntegration:
    """Test sensor integration with model predictions."""
    
    def test_sensor_to_prediction_pipeline(self):
        """Test complete sensor-to-prediction pipeline."""
        # Create sensor interface
        enose = ENoseInterface(sensors=["TGS2600", "TGS2602", "TGS2610"])
        
        # Create model
        config = OlfactoryConfig(
            vocab_size=50,
            hidden_size=32,
            num_hidden_layers=1,
            sensor_channels=64
        )
        model = OlfactoryTransformer(config)
        model.eval()
        
        # Take sensor reading
        reading = enose.read_single()
        
        # Make prediction from sensor data
        with torch.no_grad():
            prediction = model.predict_from_sensors(reading)
        
        assert isinstance(prediction, ScentPrediction)
        assert len(prediction.primary_notes) > 0
        assert 0 <= prediction.intensity <= 10
        assert 0 <= prediction.confidence <= 1
    
    def test_real_time_sensor_streaming(self):
        """Test real-time sensor streaming with predictions."""
        enose = ENoseInterface(sensors=["TGS2600"], sampling_rate=5.0)
        
        config = OlfactoryConfig(vocab_size=50, hidden_size=16, num_hidden_layers=1)
        model = OlfactoryTransformer(config)
        model.eval()
        
        predictions = []
        
        # Stream for a short duration
        with enose.stream(duration=0.5) as sensor_stream:
            for reading in sensor_stream:
                with torch.no_grad():
                    prediction = model.predict_from_sensors(reading)
                predictions.append(prediction)
                
                # Break after a few predictions to avoid long test
                if len(predictions) >= 2:
                    break
        
        assert len(predictions) >= 1
        assert all(isinstance(p, ScentPrediction) for p in predictions)


class TestMolecularDesign:
    """Test molecular design integration."""
    
    def test_design_to_evaluation_pipeline(self):
        """Test complete molecular design and evaluation pipeline."""
        # Create model
        config = OlfactoryConfig(vocab_size=100, hidden_size=32, num_hidden_layers=1)
        model = OlfactoryTransformer(config)
        
        # Create designer
        designer = ScentDesigner(model)
        
        # Design molecules
        target_profile = {
            "notes": ["floral", "fresh"],
            "intensity": 7.0,
            "longevity": "high",
            "character": "elegant"
        }
        
        candidates = designer.design_molecules(
            target_profile,
            n_candidates=5,
            molecular_weight=(150, 300),
            logp=(1, 4)
        )
        
        assert len(candidates) <= 5  # May be fewer if constraints are strict
        assert all(hasattr(c, 'smiles') for c in candidates)
        assert all(hasattr(c, 'profile_match') for c in candidates)
        assert all(0 <= c.profile_match <= 1 for c in candidates)
        
        # Test formulation optimization
        if candidates:
            formulation = designer.optimize_formulation(candidates, target_profile)
            
            assert isinstance(formulation, dict)
            assert len(formulation) <= len(candidates)
            assert all(isinstance(k, str) for k in formulation.keys())  # SMILES keys
            assert all(isinstance(v, float) for v in formulation.values())  # Weight fractions
    
    def test_ai_perfumer_workflow(self):
        """Test AI perfumer workflow."""
        from olfactory_transformer.design.inverse import AIPerfumer
        
        config = OlfactoryConfig(vocab_size=50, hidden_size=16, num_hidden_layers=1)
        model = OlfactoryTransformer(config)
        
        perfumer = AIPerfumer(model)
        
        brief = """
        Create a unisex summer fragrance with:
        - Top notes: Fresh citrus with mint
        - Heart: Light floral, jasmine
        - Base: Clean woods, white musk
        - Character: Modern, minimalist, long-lasting
        """
        
        formula = perfumer.create_fragrance(
            brief=brief,
            concentration="eau_de_parfum",
            price_range="premium"
        )
        
        assert formula is not None
        assert hasattr(formula, 'name')
        assert hasattr(formula, 'formulation')
        assert hasattr(formula, 'molecules')
        assert isinstance(formula.formulation, dict)
        
        # Test export
        export_text = formula.export_to_perfumers_workbench()
        assert isinstance(export_text, str)
        assert "Fragrance:" in export_text
        assert "Formulation:" in export_text


class TestTrainingPipeline:
    """Test complete training pipeline."""
    
    def test_training_workflow(self):
        """Test complete training workflow."""
        # Create small dataset
        molecules = ["CCO", "CC(C)O", "C1=CC=CC=C1"] * 10  # Repeat for more samples
        descriptions = []
        
        for smiles in molecules:
            desc = {
                "primary_notes": ["fresh", "clean"] if "C" in smiles else ["aromatic"],
                "intensity": np.random.uniform(3, 8),
                "chemical_family": "alcohol" if "O" in smiles else "aromatic"
            }
            descriptions.append(desc)
        
        # Create tokenizer
        tokenizer = MoleculeTokenizer(vocab_size=100)
        tokenizer.build_vocab_from_smiles(molecules)
        
        # Create dataset
        dataset = OlfactoryDataset(
            molecules=molecules,
            descriptions=descriptions,
            tokenizer=tokenizer,
            max_length=20
        )
        
        train_dataset, eval_dataset = dataset.split(0.2)
        
        # Create small model for testing
        config = OlfactoryConfig(
            vocab_size=100,
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=2
        )
        model = OlfactoryTransformer(config)
        
        # Setup training
        with tempfile.TemporaryDirectory() as tmp_dir:
            args = TrainingArguments(
                output_dir=tmp_dir,
                num_epochs=1,  # Just one epoch for testing
                batch_size=4,
                learning_rate=1e-3,
                eval_steps=5,
                logging_steps=2
            )
            
            trainer = OlfactoryTrainer(
                model=model,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                args=args
            )
            
            # Run training
            train_results = trainer.train()
            
            assert "train_loss" in train_results
            assert train_results["train_loss"] >= 0
            
            # Test prediction after training
            predictions = trainer.predict(eval_dataset)
            assert len(predictions) == len(eval_dataset)
            assert all(isinstance(p, dict) for p in predictions)


class TestEvaluationPipeline:
    """Test complete evaluation pipeline."""
    
    def test_evaluation_workflow(self):
        """Test complete evaluation workflow."""
        # Create model
        config = OlfactoryConfig(vocab_size=50, hidden_size=16, num_hidden_layers=1)
        model = OlfactoryTransformer(config)
        
        # Create evaluator (will use mock panel data)
        evaluator = PerceptualEvaluator(model)
        
        # Test compounds
        test_compounds = ["CCO", "CC(C)O", "C1=CC=CC=C1"]
        
        # Run evaluation
        results = evaluator.evaluate_model(test_compounds, detailed=True)
        
        assert isinstance(results, dict)
        assert len(results) > 0
        
        # Should have standard metrics
        expected_metrics = ["primary_accord", "intensity", "character"]
        for metric in expected_metrics:
            if metric in results:
                assert hasattr(results[metric], 'score')
                assert 0 <= results[metric].score <= 1
        
        # Test correlation analysis
        correlation_reports = evaluator.compare_predictions(test_compounds)
        
        assert isinstance(correlation_reports, dict)
        
        # Generate report
        report = evaluator.generate_evaluation_report(results, correlation_reports)
        
        assert isinstance(report, str)
        assert "Evaluation Report" in report
        assert "Summary" in report
    
    def test_benchmarking(self):
        """Test model benchmarking."""
        config = OlfactoryConfig(vocab_size=50, hidden_size=16, num_hidden_layers=1)
        model = OlfactoryTransformer(config)
        
        evaluator = PerceptualEvaluator(model)
        
        # Test baseline comparison
        baselines = {"random": Mock(), "similarity": Mock()}
        baseline_scores = evaluator.benchmark_against_baselines(baselines)
        
        assert isinstance(baseline_scores, dict)
        assert "random" in baseline_scores
        assert "fingerprint_similarity" in baseline_scores


class TestCachingIntegration:
    """Test caching integration across the system."""
    
    def test_end_to_end_caching(self):
        """Test caching throughout the prediction pipeline."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create model cache
            cache = ModelCache(cache_dir=Path(tmp_dir))
            
            # Create model and tokenizer
            config = OlfactoryConfig(vocab_size=50, hidden_size=16, num_hidden_layers=1)
            model = OlfactoryTransformer(config)
            tokenizer = MoleculeTokenizer(vocab_size=50)
            tokenizer.build_vocab_from_smiles(["CCO", "CC(C)O"])
            
            test_smiles = "CCO"
            
            # First prediction - should cache results
            start_time = time.time()
            prediction1 = model.predict_scent(test_smiles, tokenizer)
            first_time = time.time() - start_time
            
            # Cache the prediction manually (in real system this would be automatic)
            cache.cache_prediction(test_smiles, prediction1)
            
            # Second prediction - should use cache
            cached_prediction = cache.get_prediction(test_smiles)
            
            assert cached_prediction is not None
            assert cached_prediction.primary_notes == prediction1.primary_notes
            assert cached_prediction.intensity == prediction1.intensity
            
            # Test molecular features caching
            features = tokenizer.extract_molecular_features(test_smiles)
            if features:
                cache.cache_molecular_features(test_smiles, features)
                cached_features = cache.get_molecular_features(test_smiles)
                assert cached_features == features


class TestPerformanceMonitoring:
    """Test performance monitoring integration."""
    
    def test_monitored_inference(self):
        """Test inference with performance monitoring."""
        # Create model
        config = OlfactoryConfig(vocab_size=50, hidden_size=16, num_hidden_layers=1)
        model = OlfactoryTransformer(config)
        model.eval()
        
        # Create monitor
        monitor = PerformanceMonitor(window_size=10)
        
        # Monitored inference
        test_input = {
            "input_ids": torch.randint(0, 50, (2, 10)),
            "attention_mask": torch.ones(2, 10)
        }
        
        with monitor.time_inference("olfactory_model"):
            with torch.no_grad():
                outputs = model(**test_input)
        
        # Check that metrics were recorded
        assert len(monitor.metrics_history) == 1
        metrics = monitor.metrics_history[0]
        
        assert metrics.model_name == "olfactory_model"
        assert metrics.inference_time > 0
        assert metrics.throughput > 0
        assert metrics.memory_usage_mb > 0
        
        # Get stats
        stats = monitor.get_current_stats()
        assert "recent_avg_inference_time" in stats
        assert "olfactory_model_avg_time" in stats


class TestSystemIntegration:
    """Test complete system integration scenarios."""
    
    def test_industrial_qc_scenario(self):
        """Test industrial quality control scenario."""
        from olfactory_transformer.sensors.enose import ENoseArray, OlfactoryQualityControl
        
        # Create sensor array
        sensor_config = {
            "gas_sensors": ["TGS2600", "TGS2602"],
            "environmental": ["BME680"]
        }
        sensor_array = ENoseArray(sensor_config)
        
        # Create model
        config = OlfactoryConfig(vocab_size=50, hidden_size=16, num_hidden_layers=1)
        model = OlfactoryTransformer(config)
        
        # Create QC system
        reference_profile = {
            "target_notes": ["lavender", "fresh"],
            "acceptable_deviation": 0.15
        }
        
        qc_system = OlfactoryQualityControl(
            model=model,
            sensor_array=sensor_array,
            reference_profile=reference_profile
        )
        
        # Mock sensor streaming
        with patch.object(sensor_array, 'acquire') as mock_acquire:
            mock_readings = [
                {"gas_sensors": {"TGS2600": 2.5, "TGS2602": 3.1}},
                {"gas_sensors": {"TGS2600": 2.7, "TGS2602": 3.0}},
            ]
            
            def mock_stream():
                for reading in mock_readings:
                    yield reading
            
            mock_acquire.return_value.__enter__ = lambda self: mock_stream()
            mock_acquire.return_value.__exit__ = lambda self, *args: None
            
            # Test QC streaming
            qc_results = list(qc_system.stream(batch_size=2))
            
            assert len(qc_results) > 0
            result = qc_results[0]
            
            assert hasattr(result, 'deviation')
            assert hasattr(result, 'recommendation')
            assert isinstance(result.deviation, float)
            assert isinstance(result.recommendation, str)
    
    def test_research_workflow(self):
        """Test research and development workflow."""
        # 1. Molecular design
        config = OlfactoryConfig(vocab_size=50, hidden_size=16, num_hidden_layers=1)
        model = OlfactoryTransformer(config)
        designer = ScentDesigner(model)
        
        target_profile = {
            "notes": ["citrus", "fresh"],
            "intensity": 6.5,
            "character": "energizing"
        }
        
        candidates = designer.design_molecules(target_profile, n_candidates=3)
        
        # 2. Evaluate candidates
        evaluator = PerceptualEvaluator(model)
        
        if candidates:
            candidate_smiles = [c.smiles for c in candidates]
            evaluation_results = evaluator.evaluate_model(candidate_smiles)
            
            assert isinstance(evaluation_results, dict)
            
            # 3. Generate research report
            report = evaluator.generate_evaluation_report(evaluation_results)
            assert isinstance(report, str)
            assert len(report) > 100  # Should be a substantive report
    
    def test_production_deployment_scenario(self):
        """Test production deployment scenario with optimization."""
        from olfactory_transformer.utils.optimization import ModelOptimizer
        from olfactory_transformer.utils.monitoring import ResourceTracker
        
        # Create and optimize model
        config = OlfactoryConfig(vocab_size=50, hidden_size=16, num_hidden_layers=1)
        model = OlfactoryTransformer(config)
        model.eval()
        
        optimizer = ModelOptimizer(model)
        
        # Quantize for production
        quantized_model = optimizer.quantize_model(method="dynamic")
        
        # Setup monitoring
        resource_tracker = ResourceTracker()
        performance_monitor = PerformanceMonitor()
        
        # Setup inference accelerator
        accelerator = InferenceAccelerator(quantized_model, max_batch_size=8)
        
        # Simulate production workload
        with accelerator:
            for batch_id in range(3):
                batch_data = []
                for i in range(4):  # 4 samples per batch
                    batch_data.append({
                        "input_ids": torch.randint(0, 50, (8,)),
                        "attention_mask": torch.ones(8)
                    })
                
                with performance_monitor.time_inference(f"batch_{batch_id}"):
                    results = accelerator.predict_batch_sync(batch_data)
                
                assert len(results) == 4
        
        # Check performance stats
        perf_stats = performance_monitor.get_current_stats()
        accel_stats = accelerator.get_performance_stats()
        
        assert perf_stats["total_inferences"] >= 3
        assert accel_stats["requests_processed"] >= 12
        
        # Get resource snapshot
        resource_snapshot = resource_tracker.get_resource_snapshot()
        assert resource_snapshot.timestamp > 0
        assert resource_snapshot.memory_percent >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])