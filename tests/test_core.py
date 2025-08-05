"""Tests for core olfactory transformer functionality."""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch
import tempfile
from pathlib import Path

from olfactory_transformer.core.model import OlfactoryTransformer
from olfactory_transformer.core.tokenizer import MoleculeTokenizer
from olfactory_transformer.core.config import OlfactoryConfig, ScentPrediction


class TestOlfactoryConfig:
    """Test OlfactoryConfig class."""
    
    def test_config_creation(self):
        """Test basic config creation."""
        config = OlfactoryConfig()
        assert config.vocab_size == 50000
        assert config.hidden_size == 1024
        assert config.num_attention_heads == 16
    
    def test_config_to_dict(self):
        """Test config serialization."""
        config = OlfactoryConfig(vocab_size=1000, hidden_size=512)
        config_dict = config.to_dict()
        
        assert config_dict["vocab_size"] == 1000
        assert config_dict["hidden_size"] == 512
    
    def test_config_from_dict(self):
        """Test config deserialization."""
        config_dict = {"vocab_size": 2000, "hidden_size": 256}
        config = OlfactoryConfig.from_dict(config_dict)
        
        assert config.vocab_size == 2000
        assert config.hidden_size == 256
    
    def test_config_json_roundtrip(self):
        """Test JSON save/load roundtrip."""
        config = OlfactoryConfig(vocab_size=3000)
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            json_path = Path(tmp_dir) / "config.json"
            config.save_json(json_path)
            
            loaded_config = OlfactoryConfig.from_json(json_path)
            assert loaded_config.vocab_size == 3000


class TestScentPrediction:
    """Test ScentPrediction class."""
    
    def test_prediction_creation(self):
        """Test basic prediction creation."""
        prediction = ScentPrediction(
            primary_notes=["floral", "fresh"],
            intensity=7.5,
            confidence=0.85
        )
        
        assert prediction.primary_notes == ["floral", "fresh"]
        assert prediction.intensity == 7.5
        assert prediction.confidence == 0.85
    
    def test_prediction_to_dict(self):
        """Test prediction serialization."""
        prediction = ScentPrediction(
            primary_notes=["woody"],
            intensity=6.0,
            chemical_family="terpene"
        )
        
        pred_dict = prediction.to_dict()
        assert pred_dict["primary_notes"] == ["woody"]
        assert pred_dict["intensity"] == 6.0
        assert pred_dict["chemical_family"] == "terpene"


class TestMoleculeTokenizer:
    """Test MoleculeTokenizer class."""
    
    def test_tokenizer_creation(self):
        """Test basic tokenizer creation."""
        tokenizer = MoleculeTokenizer(vocab_size=1000)
        assert tokenizer.vocab_size == 1000
        assert len(tokenizer.special_tokens) == 5  # 5 special tokens
    
    def test_smiles_tokenization(self):
        """Test SMILES string tokenization."""
        tokenizer = MoleculeTokenizer()
        
        # Test basic SMILES
        tokens = tokenizer.tokenize_smiles("CCO")  # Ethanol
        assert len(tokens) > 0
        assert all(isinstance(t, str) for t in tokens)
        
        # Test empty SMILES
        tokens = tokenizer.tokenize_smiles("")
        assert tokens == []
    
    def test_encoding_decoding(self):
        """Test SMILES encoding/decoding."""
        tokenizer = MoleculeTokenizer()
        
        # Build minimal vocabulary
        smiles_list = ["CCO", "CC(C)O", "C1=CC=CC=C1"]
        tokenizer.build_vocab_from_smiles(smiles_list)
        
        # Test encoding
        smiles = "CCO"
        encoded = tokenizer.encode(smiles, max_length=20)
        
        assert "input_ids" in encoded
        assert "attention_mask" in encoded
        assert len(encoded["input_ids"]) == 20
        assert len(encoded["attention_mask"]) == 20
        
        # Test decoding
        decoded = tokenizer.decode(encoded["input_ids"])
        assert isinstance(decoded, str)
    
    @patch('olfactory_transformer.core.tokenizer.HAS_RDKIT', True)
    def test_molecular_features_with_rdkit(self):
        """Test molecular feature extraction with RDKit."""
        tokenizer = MoleculeTokenizer()
        
        with patch('olfactory_transformer.core.tokenizer.Chem') as mock_chem:
            # Mock RDKit molecule
            mock_mol = Mock()
            mock_chem.MolFromSmiles.return_value = mock_mol
            mock_mol.GetNumAtoms.return_value = 3
            mock_mol.GetNumBonds.return_value = 2
            
            # Mock descriptors
            with patch('olfactory_transformer.core.tokenizer.Descriptors') as mock_desc:
                mock_desc.MolWt.return_value = 46.07
                mock_desc.MolLogP.return_value = -0.31
                mock_desc.RingCount.return_value = 0
                mock_desc.NumAromaticRings.return_value = 0
                mock_desc.TPSA.return_value = 20.23
                mock_desc.NumHDonors.return_value = 1
                mock_desc.NumHAcceptors.return_value = 1
                mock_desc.NumRotatableBonds.return_value = 0
                
                features = tokenizer.extract_molecular_features("CCO")
                
                assert "molecular_weight" in features
                assert "logp" in features
                assert features["molecular_weight"] == 46.07
                assert features["logp"] == -0.31
    
    def test_molecular_features_without_rdkit(self):
        """Test molecular feature extraction without RDKit."""
        with patch('olfactory_transformer.core.tokenizer.HAS_RDKIT', False):
            tokenizer = MoleculeTokenizer()
            features = tokenizer.extract_molecular_features("CCO")
            assert features == {}
    
    def test_tokenizer_save_load(self):
        """Test tokenizer save/load functionality."""
        tokenizer = MoleculeTokenizer(vocab_size=100)
        smiles_list = ["CCO", "CC(C)O"]
        tokenizer.build_vocab_from_smiles(smiles_list)
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = Path(tmp_dir)
            
            # Test save
            tokenizer.save_pretrained(save_path)
            assert (save_path / "vocab.json").exists()
            assert (save_path / "tokenizer_config.json").exists()
            
            # Test load
            loaded_tokenizer = MoleculeTokenizer.from_pretrained(save_path)
            assert loaded_tokenizer.vocab_size == 100
            assert len(loaded_tokenizer.token_to_id) > 0


class TestOlfactoryTransformer:
    """Test OlfactoryTransformer model."""
    
    def test_model_creation(self):
        """Test basic model creation."""
        config = OlfactoryConfig(
            vocab_size=100,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4
        )
        model = OlfactoryTransformer(config)
        
        assert model.config.vocab_size == 100
        assert model.config.hidden_size == 64
        assert len(model.transformer_layers) == 2
    
    def test_model_forward_pass(self):
        """Test model forward pass."""
        config = OlfactoryConfig(
            vocab_size=100,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            max_position_embeddings=50
        )
        model = OlfactoryTransformer(config)
        model.eval()
        
        # Create test input
        batch_size = 2
        seq_len = 10
        input_ids = torch.randint(0, 100, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        
        # Forward pass
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Check outputs
        assert "scent_logits" in outputs
        assert "intensity" in outputs
        assert "similarity_embedding" in outputs
        assert "chemical_family_logits" in outputs
        
        # Check shapes
        assert outputs["scent_logits"].shape == (batch_size, config.num_scent_classes)
        assert outputs["intensity"].shape == (batch_size, 1)
        assert outputs["similarity_embedding"].shape == (batch_size, config.similarity_dim)
        assert outputs["chemical_family_logits"].shape == (batch_size, 20)
    
    def test_model_forward_with_labels(self):
        """Test model forward pass with labels (training mode)."""
        config = OlfactoryConfig(
            vocab_size=100,
            hidden_size=64,
            num_hidden_layers=1,
            num_attention_heads=4
        )
        model = OlfactoryTransformer(config)
        
        # Create test input with labels
        batch_size = 2
        seq_len = 10
        input_ids = torch.randint(0, 100, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        
        labels = {
            "scent_labels": torch.randint(0, config.num_scent_classes, (batch_size,)),
            "intensity_labels": torch.rand(batch_size) * 10,
            "chemical_family_labels": torch.randint(0, 20, (batch_size,))
        }
        
        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        # Check that losses are computed
        assert "losses" in outputs
        assert "loss" in outputs
        assert "scent_loss" in outputs["losses"]
        assert "intensity_loss" in outputs["losses"]
        assert "chemical_family_loss" in outputs["losses"]
        
        # Check that loss is a scalar
        assert outputs["loss"].ndim == 0
    
    def test_predict_scent(self):
        """Test scent prediction from SMILES."""
        config = OlfactoryConfig(vocab_size=100, hidden_size=32, num_hidden_layers=1)
        model = OlfactoryTransformer(config)
        
        # Mock tokenizer
        tokenizer = Mock()
        tokenizer.encode.return_value = {
            "input_ids": [1, 2, 3, 0, 0],
            "attention_mask": [1, 1, 1, 0, 0]
        }
        tokenizer.extract_molecular_features.return_value = {
            "molecular_weight": 46.07,
            "logp": -0.31
        }
        
        # Test prediction
        prediction = model.predict_scent("CCO", tokenizer)
        
        assert isinstance(prediction, ScentPrediction)
        assert len(prediction.primary_notes) > 0
        assert 0 <= prediction.intensity <= 10
        assert 0 <= prediction.confidence <= 1
        assert prediction.chemical_family in model.chemical_families
    
    def test_model_save_load(self):
        """Test model save/load functionality."""
        config = OlfactoryConfig(vocab_size=50, hidden_size=32)
        model = OlfactoryTransformer(config)
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = Path(tmp_dir)
            
            # Test save
            model.save_pretrained(save_path)
            assert (save_path / "pytorch_model.bin").exists()
            assert (save_path / "config.json").exists()
            
            # Test load
            loaded_model = OlfactoryTransformer.from_pretrained(save_path)
            assert loaded_model.config.vocab_size == 50
            assert loaded_model.config.hidden_size == 32
    
    def test_zero_shot_classify(self):
        """Test zero-shot classification."""
        config = OlfactoryConfig(vocab_size=50, hidden_size=32)
        model = OlfactoryTransformer(config)
        
        categories = ["woody", "citrus", "floral"]
        result = model.zero_shot_classify("CCO", categories)
        
        assert isinstance(result, dict)
        assert len(result) == len(categories)
        assert all(cat in result for cat in categories)
        assert all(isinstance(prob, (float, np.floating)) for prob in result.values())
        assert abs(sum(result.values()) - 1.0) < 0.1  # Probabilities should roughly sum to 1


class TestModelIntegration:
    """Integration tests for model components."""
    
    def test_tokenizer_model_integration(self):
        """Test tokenizer and model working together."""
        # Create small config for testing
        config = OlfactoryConfig(
            vocab_size=200,
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=2
        )
        
        # Create tokenizer and build vocabulary
        tokenizer = MoleculeTokenizer(vocab_size=200)
        smiles_list = ["CCO", "CC(C)O", "C1=CC=CC=C1", "CC(=O)OCC"]
        tokenizer.build_vocab_from_smiles(smiles_list)
        
        # Create model
        model = OlfactoryTransformer(config)
        model.eval()
        
        # Test prediction pipeline
        test_smiles = "CCO"
        
        with torch.no_grad():
            prediction = model.predict_scent(test_smiles, tokenizer)
        
        assert isinstance(prediction, ScentPrediction)
        assert prediction.primary_notes is not None
        assert prediction.intensity >= 0
        assert prediction.confidence >= 0
    
    def test_batch_processing(self):
        """Test batch processing capabilities."""
        config = OlfactoryConfig(
            vocab_size=100,
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=2
        )
        model = OlfactoryTransformer(config)
        model.eval()
        
        # Create batch input
        batch_size = 4
        seq_len = 15
        input_ids = torch.randint(0, 100, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        
        # Test batch processing
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Check batch dimensions
        assert outputs["scent_logits"].shape[0] == batch_size
        assert outputs["intensity"].shape[0] == batch_size
        assert outputs["similarity_embedding"].shape[0] == batch_size


if __name__ == "__main__":
    pytest.main([__file__])