#!/usr/bin/env python3
"""Basic functionality test without heavy dependencies."""

import sys
import os
from pathlib import Path

# Add repo to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test basic imports without torch."""
    print("Testing imports...")
    
    try:
        from olfactory_transformer.core.config import OlfactoryConfig, ScentPrediction, SensorReading
        print("✓ Config classes imported successfully")
        
        # Test config creation
        config = OlfactoryConfig(vocab_size=100, hidden_size=64)
        print(f"✓ Config created: vocab_size={config.vocab_size}, hidden_size={config.hidden_size}")
        
        # Test config serialization
        config_dict = config.to_dict()
        config2 = OlfactoryConfig.from_dict(config_dict)
        print("✓ Config serialization working")
        
        # Test prediction containers
        prediction = ScentPrediction(
            primary_notes=["floral", "fresh"],
            intensity=7.5,
            confidence=0.85
        )
        print(f"✓ ScentPrediction created: {prediction.primary_notes}, intensity={prediction.intensity}")
        
        return True
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_tokenizer_basic():
    """Test tokenizer without molecular features."""
    print("\nTesting tokenizer...")
    
    try:
        from olfactory_transformer.core.tokenizer import MoleculeTokenizer
        
        # Create tokenizer
        tokenizer = MoleculeTokenizer(vocab_size=100, max_length=50)
        print("✓ Tokenizer created")
        
        # Test SMILES tokenization
        smiles = "CCO"  # Ethanol
        tokens = tokenizer.tokenize_smiles(smiles)
        print(f"✓ SMILES tokenized: '{smiles}' -> {tokens}")
        
        # Build vocabulary
        sample_smiles = ["CCO", "CC(C)O", "CCC", "CCCC"]
        tokenizer.build_vocab_from_smiles(sample_smiles)
        print(f"✓ Vocabulary built: {len(tokenizer)} tokens")
        
        # Test encoding
        encoded = tokenizer.encode(smiles, padding=True, truncation=True)
        print(f"✓ Encoding successful: {len(encoded['input_ids'])} token IDs")
        
        # Test decoding
        decoded = tokenizer.decode(encoded['input_ids'][:5])
        print(f"✓ Decoding successful: '{decoded}'")
        
        return True
        
    except Exception as e:
        print(f"❌ Tokenizer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_utilities():
    """Test utility classes."""
    print("\nTesting utilities...")
    
    try:
        # Test sensor interface (mock mode)
        from olfactory_transformer.sensors.enose import ENoseInterface, SensorConfig
        
        enose = ENoseInterface(sensors=["TGS2600", "TGS2602"])
        print("✓ ENose interface created")
        
        if enose.connect():
            print("✓ Mock sensor connection successful")
            
            reading = enose.read_single()
            print(f"✓ Sensor reading: {len(reading.gas_sensors)} sensors")
            
            enose.disconnect()
            print("✓ Sensor disconnection successful")
        
        # Test sensor config
        sensor_config = SensorConfig(
            name="TGS2600",
            sensor_type="gas",
            calibration_scale=1.0
        )
        print(f"✓ Sensor config created: {sensor_config.name}")
        
        return True
        
    except Exception as e:
        print(f"❌ Utilities test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_package_structure():
    """Test package structure and imports."""
    print("\nTesting package structure...")
    
    try:
        # Test main package import (without torch-dependent parts)
        import olfactory_transformer
        print(f"✓ Main package imported, version: {olfactory_transformer.__version__}")
        
        # Test submodules exist
        import olfactory_transformer.core
        import olfactory_transformer.sensors
        import olfactory_transformer.utils
        print("✓ All submodules accessible")
        
        # Test design module
        import olfactory_transformer.design
        print("✓ Design module accessible")
        
        return True
        
    except Exception as e:
        print(f"❌ Package structure test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🧬 Olfactory Transformer - Basic Functionality Test")
    print("=" * 55)
    print("Testing core functionality without heavy ML dependencies...")
    print()
    
    tests = [
        test_imports,
        test_tokenizer_basic,
        test_utilities,
        test_package_structure,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            failed += 1
        print()
    
    print("=" * 55)
    print(f"Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All basic functionality tests PASSED!")
        print("✓ Generation 1 - MAKE IT WORK is functional")
        print()
        print("Note: Full ML functionality requires PyTorch installation")
        print("Next: Generation 2 - MAKE IT ROBUST")
        return True
    else:
        print("❌ Some tests failed. Check implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)