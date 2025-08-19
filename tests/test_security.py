"""Test security validation and input sanitization."""

import pytest
from unittest.mock import Mock, patch

from olfactory_transformer.utils.security import (
    SecurityValidator, 
    SecureTokenizer,
    APISecurityManager
)
from olfactory_transformer.core.tokenizer import MoleculeTokenizer


class TestSecurityValidator:
    """Test SecurityValidator class."""
    
    def setup_method(self):
        self.validator = SecurityValidator()
        
    def test_validate_smiles_valid(self):
        """Test validation of valid SMILES."""
        valid_smiles = [
            "CCO",  # Ethanol
            "C1=CC=CC=C1",  # Benzene
            "CC(C)O",  # Isopropanol
            "COC1=CC(=CC=C1O)C=O",  # Vanillin
        ]
        
        for smiles in valid_smiles:
            assert self.validator.validate_smiles(smiles)
            
    def test_validate_smiles_dangerous_patterns(self):
        """Test rejection of dangerous patterns."""
        dangerous_inputs = [
            "exec('import os')",
            "eval('malicious code')",
            "__import__('subprocess')",
            "open('/etc/passwd')", 
            "../../../etc/passwd",
            "CCO; rm -rf /",
            "<script>alert('xss')</script>",
            "javascript:alert(1)",
            "DROP TABLE users",
        ]
        
        for dangerous in dangerous_inputs:
            assert not self.validator.validate_smiles(dangerous)
            
    def test_validate_smiles_invalid_types(self):
        """Test rejection of invalid types."""
        invalid_inputs = [None, 123, [], {}, True]
        
        for invalid in invalid_inputs:
            assert not self.validator.validate_smiles(invalid)
            
    def test_validate_smiles_length_limits(self):
        """Test length validation."""
        # Empty string
        assert not self.validator.validate_smiles("")
        assert not self.validator.validate_smiles("   ")
        
        # Too long
        long_smiles = "C" * 1001
        assert not self.validator.validate_smiles(long_smiles)
        
        # Just at limit
        limit_smiles = "C" * 1000
        assert self.validator.validate_smiles(limit_smiles)
        
    def test_validate_smiles_invalid_characters(self):
        """Test rejection of invalid characters."""
        invalid_chars = [
            "CCO\x00",  # Null byte
            "CCO\n\r",  # Control chars
            "CCO\t",    # Tab
            "CCO♠",     # Unicode
            "CCO€",     # Currency symbol
        ]
        
        for invalid in invalid_chars:
            assert not self.validator.validate_smiles(invalid)
            
    def test_sanitize_input(self):
        """Test input sanitization."""
        dangerous = "exec('malicious'); CCO"
        sanitized = self.validator.sanitize_input(dangerous)
        
        assert "exec" not in sanitized
        assert "CCO" in sanitized
        
    def test_sanitize_input_invalid_types(self):
        """Test sanitization with invalid types."""
        assert self.validator.sanitize_input(None) == ""
        assert self.validator.sanitize_input(123) == ""
        
    def test_validate_file_path(self):
        """Test file path validation."""
        # Valid paths
        assert self.validator.validate_file_path("model.pth")
        assert self.validator.validate_file_path("data/molecules.csv")
        
        # Invalid paths
        assert not self.validator.validate_file_path("../../../etc/passwd")
        assert not self.validator.validate_file_path("/etc/passwd")
        assert not self.validator.validate_file_path("~/secrets")
        assert not self.validator.validate_file_path(None)


class TestSecureTokenizer:
    """Test SecureTokenizer wrapper."""
    
    def setup_method(self):
        self.base_tokenizer = Mock()
        self.secure_tokenizer = SecureTokenizer(self.base_tokenizer)
        
    def test_encode_valid_smiles(self):
        """Test encoding valid SMILES."""
        self.base_tokenizer.encode.return_value = {"input_ids": [1, 2, 3]}
        
        result = self.secure_tokenizer.encode("CCO")
        
        assert result == {"input_ids": [1, 2, 3]}
        self.base_tokenizer.encode.assert_called_once_with("CCO")
        
    def test_encode_dangerous_smiles(self):
        """Test rejection of dangerous SMILES."""
        with pytest.raises(ValueError, match="Invalid or dangerous input"):
            self.secure_tokenizer.encode("exec('malicious')")
            
        self.base_tokenizer.encode.assert_not_called()
        
    def test_decode_valid_output(self):
        """Test decoding valid output."""
        self.base_tokenizer.decode.return_value = "CCO"
        
        result = self.secure_tokenizer.decode([1, 2, 3])
        
        assert result == "CCO"
        self.base_tokenizer.decode.assert_called_once_with([1, 2, 3])
        
    def test_decode_invalid_output(self):
        """Test sanitization of invalid decoded output."""
        self.base_tokenizer.decode.return_value = "exec('bad')"
        
        result = self.secure_tokenizer.decode([1, 2, 3])
        
        # Should be sanitized
        assert "exec" not in result


class TestAPISecurityManager:
    """Test API security management."""
    
    def setup_method(self):
        self.api_security = APISecurityManager()
        
    def test_validate_api_request_valid(self):
        """Test validation of valid API request."""
        request = {"smiles": "CCO"}
        assert self.api_security.validate_api_request(request)
        
    def test_validate_api_request_missing_smiles(self):
        """Test rejection of request without SMILES."""
        request = {"other_field": "value"}
        assert not self.api_security.validate_api_request(request)
        
    def test_validate_api_request_invalid_smiles(self):
        """Test rejection of request with invalid SMILES."""
        request = {"smiles": "exec('malicious')"}
        assert not self.api_security.validate_api_request(request)
        
    def test_rate_limiting(self):
        """Test rate limiting functionality."""
        client_id = "test_client"
        
        # Should allow initial requests
        for i in range(10):
            assert self.api_security.check_rate_limit(client_id, max_requests=10)
            
        # Should block after limit
        assert not self.api_security.check_rate_limit(client_id, max_requests=10)
        
    @patch('time.time')
    def test_rate_limiting_window(self, mock_time):
        """Test rate limiting window functionality."""
        client_id = "test_client"
        
        # Start at time 0
        mock_time.return_value = 0
        
        # Fill up the limit
        for i in range(5):
            assert self.api_security.check_rate_limit(client_id, max_requests=5, window_minutes=1)
            
        # Should be blocked
        assert not self.api_security.check_rate_limit(client_id, max_requests=5, window_minutes=1)
        
        # Move time forward past window
        mock_time.return_value = 120  # 2 minutes later
        
        # Should be allowed again
        assert self.api_security.check_rate_limit(client_id, max_requests=5, window_minutes=1)


@pytest.fixture
def sample_molecules():
    """Sample molecules for testing."""
    return [
        "CCO",  # Ethanol
        "C1=CC=CC=C1",  # Benzene
        "CC(C)O",  # Isopropanol
        "COC1=CC(=CC=C1O)C=O",  # Vanillin
    ]


class TestIntegratedSecurity:
    """Test integrated security with real tokenizer."""
    
    def test_tokenizer_security_integration(self, sample_molecules):
        """Test security integration with actual tokenizer."""
        tokenizer = MoleculeTokenizer()
        
        # Valid molecules should work
        for smiles in sample_molecules:
            result = tokenizer.encode(smiles)
            assert "input_ids" in result
            assert "attention_mask" in result
            
    def test_tokenizer_blocks_dangerous_input(self):
        """Test tokenizer blocks dangerous input."""
        tokenizer = MoleculeTokenizer()
        
        dangerous_inputs = [
            "exec('import os')",
            "eval('malicious')",
            "../../../etc/passwd",
        ]
        
        for dangerous in dangerous_inputs:
            with pytest.raises(ValueError):
                tokenizer.encode(dangerous)


class TestSecurityPerformance:
    """Test security validation performance."""
    
    def test_validation_performance(self, sample_molecules):
        """Test that security validation doesn't significantly impact performance."""
        import time
        
        validator = SecurityValidator()
        
        # Measure validation time
        start_time = time.time()
        for _ in range(1000):
            for smiles in sample_molecules:
                validator.validate_smiles(smiles)
        end_time = time.time()
        
        # Should complete 4000 validations in reasonable time (< 1 second)
        assert (end_time - start_time) < 1.0
        
    def test_sanitization_performance(self):
        """Test sanitization performance."""
        import time
        
        validator = SecurityValidator()
        test_text = "CCO is ethanol and safe for testing"
        
        start_time = time.time()
        for _ in range(10000):
            validator.sanitize_input(test_text)
        end_time = time.time()
        
        # Should complete 10000 sanitizations quickly
        assert (end_time - start_time) < 1.0