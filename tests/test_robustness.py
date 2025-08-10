"""Advanced robustness tests for Generation 2 validation."""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch
import time

from olfactory_transformer.core.model import OlfactoryTransformer
from olfactory_transformer.core.config import OlfactoryConfig, ScentPrediction
from olfactory_transformer.core.tokenizer import MoleculeTokenizer


class TestRobustnessGen2:
    """Generation 2: Advanced robustness and reliability tests."""
    
    @pytest.fixture
    def config(self):
        return OlfactoryConfig(
            vocab_size=100,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=128,
            max_position_embeddings=128,
            sensor_channels=16,
            molecular_features=32,
        )
    
    @pytest.fixture
    def model(self, config):
        return OlfactoryTransformer(config)
    
    @pytest.fixture
    def tokenizer(self):
        tokenizer = MoleculeTokenizer(vocab_size=100)
        tokenizer.build_vocab_from_smiles(['CCO', 'CC(C)O', 'CCC', 'CCCC', 'CC'])
        return tokenizer
    
    def test_stress_testing_large_batch(self, model):
        """Test model with large batch sizes."""
        for batch_size in [1, 16, 64, 128]:
            input_ids = torch.randint(0, 50, (batch_size, 10))
            outputs = model.forward(input_ids=input_ids)
            assert outputs["scent_logits"].shape[0] == batch_size
            
    def test_memory_stress_testing(self, model):
        """Test memory efficiency under stress."""
        initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        for i in range(100):
            input_ids = torch.randint(0, 50, (8, 20))
            with torch.no_grad():
                outputs = model.forward(input_ids=input_ids)
            
            if i % 20 == 0:
                import gc
                gc.collect()
        
        final_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        memory_increase = final_memory - initial_memory
        
        # Memory shouldn't increase significantly (< 100MB)
        assert memory_increase < 100 * 1024 * 1024
    
    def test_concurrent_predictions(self, model, tokenizer):
        """Test thread safety and concurrent predictions."""
        import threading
        import concurrent.futures
        
        def predict_worker():
            return model.predict_scent('CCO', tokenizer)
        
        # Run 10 concurrent predictions
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(predict_worker) for _ in range(10)]
            results = [future.result() for future in futures]
        
        # All predictions should succeed
        assert len(results) == 10
        assert all(isinstance(r, ScentPrediction) for r in results)
    
    def test_adversarial_inputs(self, model, tokenizer):
        """Test model robustness against adversarial inputs."""
        adversarial_smiles = [
            "C" * 100,  # Very long chain
            "CCCCCC[C@H](C)C(C)C",  # Complex stereochemistry
            "c1ccc2c(c1)ccc3c2ccc4c3cccc4",  # Large aromatic system
            "C/C=C/C=C/C=C/C=C/C",  # Multiple double bonds
        ]
        
        for smiles in adversarial_smiles:
            try:
                prediction = model.predict_scent(smiles, tokenizer)
                assert isinstance(prediction, ScentPrediction)
                assert 0 <= prediction.intensity <= 10
                assert 0 <= prediction.confidence <= 1
            except Exception as e:
                # Should handle gracefully
                assert isinstance(e, (ValueError, RuntimeError))
    
    def test_input_validation_edge_cases(self, model, tokenizer):
        """Test comprehensive input validation."""
        edge_cases = [
            ("", ValueError),  # Empty string
            ("   ", ValueError),  # Whitespace only
            ("C" + "X" * 10000, ValueError),  # Extremely long
            ("CC<script>alert(1)</script>", ValueError),  # XSS attempt
            ("CC\\x00CC", ValueError),  # Null bytes
            ("CC\n\rCC", ValueError),  # Control characters
        ]
        
        for input_str, expected_error in edge_cases:
            with pytest.raises(expected_error):
                model.predict_scent(input_str, tokenizer)
    
    def test_error_recovery_mechanisms(self, model, tokenizer):
        """Test error recovery and circuit breaker patterns."""
        from olfactory_transformer.utils.reliability import reliability_manager
        
        # Test circuit breaker functionality
        original_threshold = reliability_manager.failure_threshold
        reliability_manager.failure_threshold = 2  # Low threshold for testing
        
        try:
            # Trigger failures
            for _ in range(3):
                try:
                    model.predict_scent("INVALID_SMILES_XYZ", tokenizer)
                except Exception:
                    pass
            
            # Circuit should be open now
            assert reliability_manager.circuit_states.get("model_inference", False)
            
        finally:
            reliability_manager.failure_threshold = original_threshold
            reliability_manager.reset_circuit("model_inference")
    
    def test_performance_benchmarking(self, model, tokenizer):
        """Benchmark model performance."""
        test_smiles = ['CCO', 'CC(C)O', 'CCC', 'CCCC']
        times = []
        
        # Warm up
        for _ in range(5):
            model.predict_scent('CCO', tokenizer)
        
        # Benchmark
        for smiles in test_smiles * 10:
            start = time.time()
            model.predict_scent(smiles, tokenizer)
            times.append(time.time() - start)
        
        avg_time = np.mean(times)
        p95_time = np.percentile(times, 95)
        
        print(f"Average prediction time: {avg_time:.3f}s")
        print(f"95th percentile: {p95_time:.3f}s")
        
        # Should be reasonably fast (< 1 second for CPU inference)
        assert avg_time < 1.0
        assert p95_time < 2.0
    
    def test_numerical_precision_stability(self, model):
        """Test numerical stability across different precisions."""
        input_ids = torch.randint(0, 50, (1, 10))
        
        # Test with different dtypes
        model_fp32 = model.float()
        model_fp16 = model.half() if torch.cuda.is_available() else model.float()
        
        with torch.no_grad():
            output_fp32 = model_fp32.forward(input_ids=input_ids.long())
            output_fp16 = model_fp16.forward(input_ids=input_ids.long())
        
        # Outputs should be reasonable
        for key in output_fp32.keys():
            if isinstance(output_fp32[key], torch.Tensor):
                assert not torch.any(torch.isnan(output_fp32[key]))
                assert not torch.any(torch.isinf(output_fp32[key]))
    
    def test_model_state_consistency(self, model):
        """Test model state consistency across operations."""
        # Save initial state
        initial_state = {name: param.clone() for name, param in model.named_parameters()}
        
        # Run inference (should not change parameters)
        input_ids = torch.randint(0, 50, (2, 10))
        with torch.no_grad():
            model.forward(input_ids=input_ids)
        
        # Check parameters unchanged
        for name, param in model.named_parameters():
            torch.testing.assert_close(param, initial_state[name])
    
    def test_gradient_computation_stability(self, model):
        """Test gradient computation stability."""
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        
        for epoch in range(5):
            input_ids = torch.randint(0, 50, (4, 10))
            labels = {
                "scent_labels": torch.randint(0, 21, (4,)),
                "intensity_labels": torch.randn(4),
            }
            
            optimizer.zero_grad()
            outputs = model.forward(input_ids=input_ids, labels=labels)
            loss = outputs["loss"]
            
            # Check loss is reasonable
            assert not torch.isnan(loss)
            assert not torch.isinf(loss)
            assert loss.item() > 0
            
            loss.backward()
            
            # Check gradients
            total_norm = 0
            for param in model.parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
            
            # Gradients should not explode
            assert total_norm < 100.0
            
            optimizer.step()
    
    def test_data_corruption_resilience(self, model, tokenizer):
        """Test resilience to data corruption."""
        # Test with corrupted molecular features
        input_ids = torch.randint(0, 50, (1, 10))
        
        # Corrupted molecular features (NaN values)
        corrupted_mol_features = torch.tensor([[float('nan'), 1.0, 2.0, 3.0] * 8])
        
        with pytest.raises(Exception):  # Should handle gracefully
            model.forward(input_ids=input_ids, molecular_features=corrupted_mol_features)
    
    def test_model_serialization_integrity(self, model, tmp_path):
        """Test model serialization and deserialization integrity."""
        # Save model
        save_path = tmp_path / "robust_model"
        model.save_pretrained(save_path)
        
        # Load model
        loaded_model = OlfactoryTransformer.from_pretrained(save_path)
        
        # Compare parameters
        for (name1, p1), (name2, p2) in zip(model.named_parameters(), loaded_model.named_parameters()):
            assert name1 == name2
            torch.testing.assert_close(p1, p2)
        
        # Test identical outputs
        input_ids = torch.randint(0, 50, (1, 10))
        with torch.no_grad():
            output1 = model.forward(input_ids=input_ids)
            output2 = loaded_model.forward(input_ids=input_ids)
        
        for key in output1.keys():
            if isinstance(output1[key], torch.Tensor):
                torch.testing.assert_close(output1[key], output2[key])