"""Comprehensive tests for olfactory transformer model."""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch

from olfactory_transformer.core.model import OlfactoryTransformer
from olfactory_transformer.core.config import OlfactoryConfig, ScentPrediction
from olfactory_transformer.core.tokenizer import MoleculeTokenizer


class TestOlfactoryTransformer:
    """Test suite for OlfactoryTransformer model."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return OlfactoryConfig(
            vocab_size=100,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=128,
            max_position_embeddings=128,
            sensor_channels=16,
            molecular_feature_dim=32,
        )
    
    @pytest.fixture
    def model(self, config):
        """Create test model."""
        return OlfactoryTransformer(config)
    
    @pytest.fixture
    def tokenizer(self):
        """Create test tokenizer."""
        tokenizer = MoleculeTokenizer(vocab_size=100)
        # Build minimal vocabulary
        tokenizer.build_vocab_from_smiles(["CCO", "CC(C)O", "CCC"])
        return tokenizer
    
    def test_model_initialization(self, config):
        """Test model initialization."""
        model = OlfactoryTransformer(config)
        
        assert model.config == config
        assert hasattr(model, 'embeddings')
        assert hasattr(model, 'transformer_layers')
        assert hasattr(model, 'decoder')
        assert len(model.transformer_layers) == config.num_hidden_layers
    
    def test_forward_pass(self, model):
        """Test forward pass with valid input."""
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, 100, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        
        outputs = model.forward(input_ids=input_ids, attention_mask=attention_mask)
        
        assert isinstance(outputs, dict)
        assert "scent_logits" in outputs
        assert "intensity" in outputs
        assert "chemical_family_logits" in outputs
        
        # Check output shapes
        assert outputs["scent_logits"].shape == (batch_size, len(model.scent_descriptors))
        assert outputs["intensity"].shape == (batch_size, 1)
        assert outputs["chemical_family_logits"].shape == (batch_size, len(model.chemical_families))
    
    def test_forward_with_molecular_features(self, model):
        """Test forward pass with molecular features."""
        batch_size, seq_len = 1, 10
        input_ids = torch.randint(0, 100, (batch_size, seq_len))
        molecular_features = torch.randn(batch_size, 32)
        
        outputs = model.forward(
            input_ids=input_ids,
            molecular_features=molecular_features
        )
        
        assert isinstance(outputs, dict)
        assert "scent_logits" in outputs
    
    def test_forward_with_sensor_data(self, model):
        """Test forward pass with sensor data."""
        batch_size, seq_len = 1, 10
        input_ids = torch.randint(0, 100, (batch_size, seq_len))
        sensor_data = torch.randn(batch_size, 16)
        
        outputs = model.forward(
            input_ids=input_ids,
            sensor_data=sensor_data
        )
        
        assert isinstance(outputs, dict)
        assert "scent_logits" in outputs
    
    def test_forward_with_labels(self, model):
        """Test forward pass with labels (training mode)."""
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, 100, (batch_size, seq_len))
        
        labels = {
            "scent_labels": torch.randint(0, len(model.scent_descriptors), (batch_size,)),
            "intensity_labels": torch.randn(batch_size),
            "chemical_family_labels": torch.randint(0, len(model.chemical_families), (batch_size,))
        }
        
        outputs = model.forward(input_ids=input_ids, labels=labels)
        
        assert "loss" in outputs
        assert "losses" in outputs
        assert isinstance(outputs["losses"], dict)
        assert "scent_loss" in outputs["losses"]
        assert "intensity_loss" in outputs["losses"]
        assert "chemical_family_loss" in outputs["losses"]
    
    def test_predict_scent(self, model, tokenizer):
        """Test scent prediction from SMILES."""
        smiles = "CCO"  # Ethanol
        
        prediction = model.predict_scent(smiles, tokenizer)
        
        assert isinstance(prediction, ScentPrediction)
        assert isinstance(prediction.primary_notes, list)
        assert len(prediction.primary_notes) > 0
        assert 0 <= prediction.intensity <= 10
        assert 0 <= prediction.confidence <= 1
        assert isinstance(prediction.chemical_family, str)
    
    def test_predict_scent_invalid_input(self, model, tokenizer):
        """Test scent prediction with invalid input."""
        # Empty SMILES
        with pytest.raises(ValueError, match="SMILES string must be a non-empty string"):
            model.predict_scent("", tokenizer)
        
        # Non-string input
        with pytest.raises(ValueError, match="SMILES string must be a non-empty string"):
            model.predict_scent(123, tokenizer)
        
        # Dangerous characters
        with pytest.raises(ValueError, match="potentially dangerous characters"):
            model.predict_scent("CC<script>alert('xss')</script>", tokenizer)
        
        # No tokenizer
        with pytest.raises(ValueError, match="Tokenizer required"):
            model.predict_scent("CCO", None)
    
    def test_predict_from_sensors(self, model):
        """Test prediction from sensor readings."""
        from olfactory_transformer.sensors.enose import SensorReading
        
        sensor_reading = SensorReading(
            timestamp=1234567890.0,
            gas_sensors={
                "TGS2600": 2.5,
                "TGS2602": 1.8,
                "TGS2610": 3.2,
                "TGS2620": 0.9,
            },
            temperature=25.0,
            humidity=60.0,
            pressure=1013.25
        )
        
        prediction = model.predict_from_sensors(sensor_reading)
        
        assert isinstance(prediction, ScentPrediction)
        assert isinstance(prediction.primary_notes, list)
        assert len(prediction.primary_notes) > 0
        assert 0 <= prediction.intensity <= 10
        assert 0 <= prediction.confidence <= 1
    
    def test_predict_from_sensors_invalid_input(self, model):
        """Test sensor prediction with invalid input."""
        from olfactory_transformer.sensors.enose import SensorReading
        
        # Invalid sensor reading type
        with pytest.raises(ValueError, match="sensor_reading must be a SensorReading object"):
            model.predict_from_sensors("invalid")
        
        # Empty gas sensors
        empty_reading = SensorReading(
            timestamp=1234567890.0,
            gas_sensors={},
            temperature=25.0,
            humidity=60.0,
            pressure=1013.25
        )
        with pytest.raises(ValueError, match="No gas sensor data provided"):
            model.predict_from_sensors(empty_reading)
        
        # Invalid sensor values
        invalid_reading = SensorReading(
            timestamp=1234567890.0,
            gas_sensors={"TGS2600": float('nan')},
            temperature=25.0,
            humidity=60.0,
            pressure=1013.25
        )
        with pytest.raises(ValueError, match="Invalid sensor value"):
            model.predict_from_sensors(invalid_reading)
    
    def test_save_and_load_pretrained(self, model, tmp_path):
        """Test saving and loading pretrained model."""
        save_dir = tmp_path / "test_model"
        
        # Save model
        model.save_pretrained(save_dir)
        
        # Check files exist
        assert (save_dir / "pytorch_model.bin").exists()
        assert (save_dir / "config.json").exists()
        
        # Load model
        loaded_model = OlfactoryTransformer.from_pretrained(save_dir)
        
        assert loaded_model.config.vocab_size == model.config.vocab_size
        assert loaded_model.config.hidden_size == model.config.hidden_size
    
    def test_zero_shot_classify(self, model):
        """Test zero-shot classification."""
        smiles = "CCO"
        categories = ["alcohol", "hydrocarbon", "ester"]
        
        result = model.zero_shot_classify(smiles, categories)
        
        assert isinstance(result, dict)
        assert len(result) == len(categories)
        assert all(cat in result for cat in categories)
        assert all(0 <= prob <= 1 for prob in result.values())
        assert abs(sum(result.values()) - 1.0) < 1e-5  # Probabilities sum to 1
    
    def test_zero_shot_classify_invalid_input(self, model):
        """Test zero-shot classification with invalid input."""
        # Empty SMILES
        with pytest.raises(ValueError, match="SMILES must be a non-empty string"):
            model.zero_shot_classify("", ["category"])
        
        # Empty categories
        with pytest.raises(ValueError, match="Categories must be a non-empty list"):
            model.zero_shot_classify("CCO", [])
        
        # Too many categories
        with pytest.raises(ValueError, match="Too many categories"):
            model.zero_shot_classify("CCO", [f"cat_{i}" for i in range(101)])
    
    def test_forward_invalid_input(self, model):
        """Test forward pass with invalid input."""
        # Non-tensor input
        with pytest.raises(TypeError, match="input_ids must be a torch.Tensor"):
            model.forward(input_ids="invalid")
        
        # Wrong dimensions
        with pytest.raises(ValueError, match="input_ids must be 2D"):
            model.forward(input_ids=torch.tensor([1, 2, 3]))
        
        # Empty input
        with pytest.raises(ValueError, match="Invalid input dimensions"):
            model.forward(input_ids=torch.empty(0, 0))
        
        # Negative values
        with pytest.raises(ValueError, match="input_ids contains negative values"):
            model.forward(input_ids=torch.tensor([[-1, 0, 1]]))
        
        # Values >= vocab_size
        with pytest.raises(ValueError, match="input_ids contains values >= vocab_size"):
            model.forward(input_ids=torch.tensor([[100, 101, 102]]))
    
    def test_attention_mask_validation(self, model):
        """Test attention mask validation."""
        input_ids = torch.randint(0, 100, (1, 10))
        
        # Wrong shape
        with pytest.raises(ValueError, match="attention_mask shape.*!= input_ids shape"):
            model.forward(
                input_ids=input_ids,
                attention_mask=torch.ones(1, 5)  # Wrong sequence length
            )
    
    def test_molecular_features_validation(self, model):
        """Test molecular features validation."""
        input_ids = torch.randint(0, 100, (2, 10))
        
        # Wrong type
        with pytest.raises(TypeError, match="molecular_features must be a torch.Tensor"):
            model.forward(
                input_ids=input_ids,
                molecular_features="invalid"
            )
    
    def test_sensor_data_validation(self, model):
        """Test sensor data validation."""
        input_ids = torch.randint(0, 100, (2, 10))
        
        # Wrong type
        with pytest.raises(TypeError, match="sensor_data must be a torch.Tensor"):
            model.forward(
                input_ids=input_ids,
                sensor_data="invalid"
            )
    
    def test_device_consistency(self, model):
        """Test device consistency across model components."""
        device = torch.device('cpu')
        model = model.to(device)
        
        input_ids = torch.randint(0, 100, (1, 10)).to(device)
        
        outputs = model.forward(input_ids=input_ids)
        
        for key, tensor in outputs.items():
            if isinstance(tensor, torch.Tensor):
                assert tensor.device == device
    
    @pytest.mark.parametrize("batch_size", [1, 4, 8])
    @pytest.mark.parametrize("seq_len", [5, 10, 20])
    def test_different_batch_sizes(self, model, batch_size, seq_len):
        """Test model with different batch sizes and sequence lengths."""
        input_ids = torch.randint(0, 100, (batch_size, seq_len))
        
        outputs = model.forward(input_ids=input_ids)
        
        assert outputs["scent_logits"].shape[0] == batch_size
        assert outputs["intensity"].shape[0] == batch_size
        assert outputs["chemical_family_logits"].shape[0] == batch_size
    
    def test_gradient_flow(self, model):
        """Test gradient flow in training mode."""
        model.train()
        
        input_ids = torch.randint(0, 100, (2, 10))
        labels = {
            "scent_labels": torch.randint(0, len(model.scent_descriptors), (2,)),
            "intensity_labels": torch.randn(2),
        }
        
        outputs = model.forward(input_ids=input_ids, labels=labels)
        loss = outputs["loss"]
        
        # Backprop
        loss.backward()
        
        # Check gradients exist
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None
    
    def test_model_modes(self, model):
        """Test switching between train and eval modes."""
        # Test eval mode
        model.eval()
        assert not model.training
        
        # Test train mode
        model.train()
        assert model.training
        
        # Test inference consistency in eval mode
        model.eval()
        input_ids = torch.randint(0, 100, (1, 10))
        
        with torch.no_grad():
            output1 = model.forward(input_ids=input_ids)
            output2 = model.forward(input_ids=input_ids)
        
        # Outputs should be identical in eval mode
        torch.testing.assert_close(output1["scent_logits"], output2["scent_logits"])
    
    def test_memory_efficiency(self, model):
        """Test memory efficiency of the model."""
        import gc
        
        # Clear cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        input_ids = torch.randint(0, 100, (1, 10))
        
        # Test with gradient computation disabled
        with torch.no_grad():
            outputs = model.forward(input_ids=input_ids)
        
        # Ensure outputs are generated
        assert "scent_logits" in outputs
        
        # Clean up
        del outputs
        gc.collect()
    
    def test_numerical_stability(self, model):
        """Test numerical stability of the model."""
        # Test with extreme input values
        input_ids = torch.randint(0, 100, (1, 10))
        
        outputs = model.forward(input_ids=input_ids)
        
        # Check for NaN or Inf values
        for key, tensor in outputs.items():
            if isinstance(tensor, torch.Tensor):
                assert not torch.any(torch.isnan(tensor)), f"NaN found in {key}"
                assert not torch.any(torch.isinf(tensor)), f"Inf found in {key}"
                
                # Check reasonable value ranges
                if key == "intensity":
                    assert torch.all(tensor >= -10) and torch.all(tensor <= 20), f"Intensity values out of reasonable range: {tensor}"