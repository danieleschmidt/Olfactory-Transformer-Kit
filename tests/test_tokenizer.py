"""Tests for molecule tokenizer."""

import pytest
import tempfile
from pathlib import Path

from olfactory_transformer.core.tokenizer import MoleculeTokenizer


class TestMoleculeTokenizer:
    """Test suite for MoleculeTokenizer."""
    
    @pytest.fixture
    def tokenizer(self):
        """Create test tokenizer."""
        return MoleculeTokenizer(vocab_size=100)
    
    @pytest.fixture
    def smiles_list(self):
        """Sample SMILES strings for testing."""
        return [
            "CCO",                    # Ethanol
            "CC(C)O",                 # Isopropanol
            "C1=CC=CC=C1",           # Benzene
            "CC(=O)OCC",             # Ethyl acetate
            "C1=CC=C(C=C1)C=O",      # Benzaldehyde
            "COC1=CC(=CC=C1O)C=O",   # Vanillin
        ]
    
    def test_tokenizer_initialization(self):
        """Test tokenizer initialization."""
        tokenizer = MoleculeTokenizer(vocab_size=100)
        
        assert tokenizer.vocab_size == 100
        assert tokenizer.max_length == 128
        assert hasattr(tokenizer, 'vocab')
        assert hasattr(tokenizer, 'special_tokens')
        
        # Check special tokens
        assert '[PAD]' in tokenizer.special_tokens
        assert '[UNK]' in tokenizer.special_tokens
        assert '[CLS]' in tokenizer.special_tokens
        assert '[SEP]' in tokenizer.special_tokens
    
    def test_build_vocab_from_smiles(self, tokenizer, smiles_list):
        """Test vocabulary building."""
        tokenizer.build_vocab_from_smiles(smiles_list)
        
        # Check vocabulary was built
        assert len(tokenizer.vocab) > len(tokenizer.special_tokens)
        
        # Check special tokens are in vocab
        for token in tokenizer.special_tokens:
            assert token in tokenizer.vocab
            assert tokenizer.vocab[token] < len(tokenizer.special_tokens)
    
    def test_encode_basic(self, tokenizer, smiles_list):
        """Test basic encoding functionality."""
        tokenizer.build_vocab_from_smiles(smiles_list)
        
        smiles = "CCO"
        encoded = tokenizer.encode(smiles)
        
        assert isinstance(encoded, dict)
        assert 'input_ids' in encoded
        assert 'attention_mask' in encoded
        assert isinstance(encoded['input_ids'], list)
        assert isinstance(encoded['attention_mask'], list)
        assert len(encoded['input_ids']) == len(encoded['attention_mask'])
    
    def test_encode_with_special_tokens(self, tokenizer, smiles_list):
        """Test encoding with special tokens."""
        tokenizer.build_vocab_from_smiles(smiles_list)
        
        smiles = "CCO"
        encoded = tokenizer.encode(smiles, add_special_tokens=True)
        
        # Should start with [CLS] and end with [SEP]
        cls_token_id = tokenizer.vocab['[CLS]']
        sep_token_id = tokenizer.vocab['[SEP]']
        
        assert encoded['input_ids'][0] == cls_token_id
        # Find first non-padding token from the end
        non_pad_ids = [id for id in encoded['input_ids'] if id != tokenizer.vocab['[PAD]']]
        assert non_pad_ids[-1] == sep_token_id
    
    def test_encode_with_padding(self, tokenizer, smiles_list):
        """Test encoding with padding."""
        tokenizer.build_vocab_from_smiles(smiles_list)
        
        smiles = "CC"  # Short SMILES
        max_length = 20
        
        encoded = tokenizer.encode(smiles, max_length=max_length, padding=True)
        
        assert len(encoded['input_ids']) == max_length
        assert len(encoded['attention_mask']) == max_length
        
        # Check padding tokens
        pad_token_id = tokenizer.vocab['[PAD]']
        assert pad_token_id in encoded['input_ids']
        
        # Attention mask should be 0 for padding tokens
        for i, token_id in enumerate(encoded['input_ids']):
            if token_id == pad_token_id:
                assert encoded['attention_mask'][i] == 0
            else:
                assert encoded['attention_mask'][i] == 1
    
    def test_encode_with_truncation(self, tokenizer, smiles_list):
        """Test encoding with truncation."""
        tokenizer.build_vocab_from_smiles(smiles_list)
        
        # Use a very long SMILES (repeated)
        long_smiles = "COC1=CC(=CC=C1O)C=O" * 10
        max_length = 20
        
        encoded = tokenizer.encode(long_smiles, max_length=max_length, truncation=True)
        
        assert len(encoded['input_ids']) == max_length
        assert len(encoded['attention_mask']) == max_length
    
    def test_decode(self, tokenizer, smiles_list):
        """Test decoding functionality."""
        tokenizer.build_vocab_from_smiles(smiles_list)
        
        smiles = "CCO"
        encoded = tokenizer.encode(smiles)
        decoded = tokenizer.decode(encoded['input_ids'])
        
        assert isinstance(decoded, str)
        # Decoded string should contain original characters (may have special tokens)
        assert any(char in decoded for char in "CCO")
    
    def test_extract_molecular_features(self, tokenizer):
        """Test molecular feature extraction."""
        smiles = "CCO"
        features = tokenizer.extract_molecular_features(smiles)
        
        if features:  # Only test if RDKit is available
            assert isinstance(features, dict)
            assert len(features) > 0
            
            # Check for expected features
            expected_features = ['molecular_weight', 'logp', 'num_rings', 'num_rotatable_bonds']
            for feature in expected_features:
                if feature in features:
                    assert isinstance(features[feature], (int, float))
    
    def test_extract_molecular_features_invalid(self, tokenizer):
        """Test molecular feature extraction with invalid input."""
        # Empty string
        features = tokenizer.extract_molecular_features("")
        assert features == {}
        
        # Non-string input
        features = tokenizer.extract_molecular_features(123)
        assert features == {}
        
        # Very long string
        long_smiles = "C" * 2000
        features = tokenizer.extract_molecular_features(long_smiles)
        assert features == {}
    
    def test_save_and_load_pretrained(self, tokenizer, smiles_list, tmp_path):
        """Test saving and loading pretrained tokenizer."""
        # Build vocabulary first
        tokenizer.build_vocab_from_smiles(smiles_list)
        
        save_dir = tmp_path / "test_tokenizer"
        
        # Save tokenizer
        tokenizer.save_pretrained(save_dir)
        
        # Check files exist
        assert (save_dir / "vocab.json").exists()
        assert (save_dir / "tokenizer_config.json").exists()
        
        # Load tokenizer
        loaded_tokenizer = MoleculeTokenizer.from_pretrained(save_dir)
        
        assert loaded_tokenizer.vocab_size == tokenizer.vocab_size
        assert loaded_tokenizer.vocab == tokenizer.vocab
        assert loaded_tokenizer.special_tokens == tokenizer.special_tokens
        
        # Test encoding consistency
        smiles = "CCO"
        original_encoded = tokenizer.encode(smiles)
        loaded_encoded = loaded_tokenizer.encode(smiles)
        
        assert original_encoded == loaded_encoded
    
    def test_tokenize_unknown_characters(self, tokenizer, smiles_list):
        """Test handling of unknown characters."""
        tokenizer.build_vocab_from_smiles(smiles_list)
        
        # SMILES with unknown characters
        unknown_smiles = "CCO@#$%"
        encoded = tokenizer.encode(unknown_smiles)
        
        # Unknown characters should be mapped to [UNK]
        unk_token_id = tokenizer.vocab['[UNK]']
        assert unk_token_id in encoded['input_ids']
    
    def test_batch_encoding(self, tokenizer, smiles_list):
        """Test batch encoding functionality."""
        tokenizer.build_vocab_from_smiles(smiles_list)
        
        batch_smiles = ["CCO", "CC(C)O", "CCC"]
        encoded_batch = tokenizer.encode_batch(batch_smiles, padding=True)
        
        assert isinstance(encoded_batch, dict)
        assert 'input_ids' in encoded_batch
        assert 'attention_mask' in encoded_batch
        
        # Check batch dimensions
        assert len(encoded_batch['input_ids']) == len(batch_smiles)
        assert len(encoded_batch['attention_mask']) == len(batch_smiles)
        
        # All sequences should have same length (due to padding)
        seq_lengths = [len(seq) for seq in encoded_batch['input_ids']]
        assert len(set(seq_lengths)) == 1  # All same length
    
    def test_vocabulary_size_limits(self, tokenizer):
        """Test vocabulary size limits."""
        # Test with many SMILES to potentially exceed vocab size
        many_smiles = [f"{'C' * i}O" for i in range(1, 200)]
        
        tokenizer.build_vocab_from_smiles(many_smiles)
        
        # Vocabulary should not exceed specified size
        assert len(tokenizer.vocab) <= tokenizer.vocab_size
    
    def test_special_token_handling(self, tokenizer, smiles_list):
        """Test special token handling."""
        tokenizer.build_vocab_from_smiles(smiles_list)
        
        # Encode without special tokens
        smiles = "CCO"
        encoded_no_special = tokenizer.encode(smiles, add_special_tokens=False)
        encoded_with_special = tokenizer.encode(smiles, add_special_tokens=True)
        
        # With special tokens should be longer
        assert len(encoded_with_special['input_ids']) > len(encoded_no_special['input_ids'])
        
        # Check for special tokens
        cls_id = tokenizer.vocab['[CLS]']
        sep_id = tokenizer.vocab['[SEP]']
        
        assert cls_id in encoded_with_special['input_ids']
        assert sep_id in encoded_with_special['input_ids']
        assert cls_id not in encoded_no_special['input_ids']
        assert sep_id not in encoded_no_special['input_ids']
    
    def test_encode_security_validation(self, tokenizer):
        """Test security validation in encoding."""
        # Test dangerous characters
        with pytest.raises(ValueError, match="suspicious patterns"):
            tokenizer.encode("CCO exec('malicious code')")
        
        # Test very long input
        with pytest.raises(ValueError, match="SMILES string too long"):
            tokenizer.encode("C" * 20000)
        
        # Test non-string input
        with pytest.raises(TypeError, match="SMILES must be a string"):
            tokenizer.encode(123)
        
        # Test excessive max_length
        with pytest.raises(ValueError, match="max_length too large"):
            tokenizer.encode("CCO", max_length=20000)
    
    def test_build_vocab_security(self, tokenizer):
        """Test security validation in vocabulary building."""
        # Test non-list input
        with pytest.raises(TypeError, match="smiles_list must be a list"):
            tokenizer.build_vocab_from_smiles("not a list")
        
        # Test too many SMILES
        with pytest.raises(ValueError, match="Too many SMILES strings"):
            tokenizer.build_vocab_from_smiles(["C"] * 2000000)
        
        # Test empty list
        with pytest.raises(ValueError, match="No valid SMILES strings provided"):
            tokenizer.build_vocab_from_smiles([])
        
        # Test invalid SMILES
        invalid_smiles = ["", "   ", 123, None, "C" * 2000]
        tokenizer.build_vocab_from_smiles(invalid_smiles)
        # Should not raise error but filter out invalid ones
        assert len(tokenizer.vocab) == len(tokenizer.special_tokens)  # Only special tokens
    
    def test_path_traversal_protection(self, tokenizer):
        """Test protection against path traversal attacks."""
        # Test save with dangerous path
        with pytest.warns(None):  # Should warn but not fail
            tokenizer.save_pretrained("../../../etc/passwd")
        
        # Test load with dangerous path
        with pytest.raises(ValueError, match="Path traversal detected"):
            MoleculeTokenizer.from_pretrained("../../../etc/passwd")
    
    def test_empty_vocabulary_handling(self, tokenizer):
        """Test handling of empty vocabulary."""
        # Try to encode without building vocabulary
        smiles = "CCO"
        encoded = tokenizer.encode(smiles)
        
        # Should still work with unknown tokens
        assert isinstance(encoded, dict)
        assert 'input_ids' in encoded
        assert 'attention_mask' in encoded
    
    def test_concurrent_access(self, tokenizer, smiles_list):
        """Test thread-safe access to tokenizer."""
        import threading
        import time
        
        tokenizer.build_vocab_from_smiles(smiles_list)
        results = []
        errors = []
        
        def encode_smiles(smiles, results, errors):
            try:
                encoded = tokenizer.encode(smiles)
                results.append(encoded)
            except Exception as e:
                errors.append(e)
        
        # Create multiple threads
        threads = []
        test_smiles = ["CCO", "CC(C)O", "CCC", "C1=CC=CC=C1"]
        
        for smiles in test_smiles:
            thread = threading.Thread(target=encode_smiles, args=(smiles, results, errors))
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Check results
        assert len(errors) == 0, f"Errors in concurrent access: {errors}"
        assert len(results) == len(test_smiles)
    
    @pytest.mark.parametrize("vocab_size", [50, 100, 500])
    def test_different_vocab_sizes(self, vocab_size, smiles_list):
        """Test tokenizer with different vocabulary sizes."""
        tokenizer = MoleculeTokenizer(vocab_size=vocab_size)
        tokenizer.build_vocab_from_smiles(smiles_list)
        
        assert tokenizer.vocab_size == vocab_size
        assert len(tokenizer.vocab) <= vocab_size
        
        # Test encoding still works
        encoded = tokenizer.encode("CCO")
        assert isinstance(encoded, dict)
        assert 'input_ids' in encoded
    
    @pytest.mark.parametrize("max_length", [10, 50, 128])
    def test_different_max_lengths(self, tokenizer, smiles_list, max_length):
        """Test tokenizer with different max lengths."""
        tokenizer.max_length = max_length
        tokenizer.build_vocab_from_smiles(smiles_list)
        
        smiles = "COC1=CC(=CC=C1O)C=O"  # Moderately long SMILES
        encoded = tokenizer.encode(smiles, padding=True)
        
        assert len(encoded['input_ids']) == max_length
        assert len(encoded['attention_mask']) == max_length