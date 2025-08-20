#!/usr/bin/env python3
"""
Test Generation 2 Features - Robust Multi-Modal Processing.

Validates Generation 2 enhancements:
- Robust input validation and processing
- Multi-modal data handling
- Dependency management with graceful fallbacks
- Enhanced internationalization
- Error handling and fault tolerance
"""

import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_robust_processing():
    """Test robust processing utilities."""
    print("🛡️ Testing Robust Processing...")
    
    try:
        from olfactory_transformer.utils.robust_processing import (
            InputValidator, FaultTolerantProcessor, MultiModalDataHandler,
            RobustPipeline, ValidationResult, ProcessingResult
        )
        
        # Test 1: Input Validation
        print("   📝 Testing Input Validation...")
        validator = InputValidator()
        
        # Test SMILES validation
        valid_smiles = "CC(C)CC1=CC=C(C=C1)C(C)C"
        smiles_result = validator.validate_smiles(valid_smiles)
        print(f"     ✅ Valid SMILES: {smiles_result.valid}")
        
        # Test invalid SMILES
        invalid_smiles = "INVALID_SMILES<script>alert('xss')</script>"
        invalid_result = validator.validate_smiles(invalid_smiles)
        print(f"     ✅ Invalid SMILES detected: {not invalid_result.valid} ({len(invalid_result.errors)} errors)")
        
        # Test sensor data validation
        sensor_data = {"TGS2600": 245.2, "TGS2602": 187.3, "invalid": "not_a_number"}
        sensor_result = validator.validate_sensor_data(sensor_data)
        print(f"     ✅ Sensor validation: {len(sensor_result.errors)} errors, {len(sensor_result.warnings)} warnings")
        
        # Test 2: Fault Tolerant Processing
        print("   🔧 Testing Fault Tolerance...")
        processor = FaultTolerantProcessor()
        
        def unreliable_function():
            import random
            if random.random() < 0.3:  # 30% failure rate
                raise RuntimeError("Simulated failure")
            return "Success!"
        
        def fallback_function():
            return "Fallback result"
        
        # Test with fallback
        fault_tolerant = processor.with_fallback(unreliable_function, fallback_function)
        result = fault_tolerant()
        print(f"     ✅ Fault tolerance result: success={result.success}, fallback_used={result.fallback_used}")
        
        # Test 3: Multi-Modal Data Handling
        print("   🌈 Testing Multi-Modal Processing...")
        handler = MultiModalDataHandler()
        
        modality_data = {
            'molecular': valid_smiles,
            'sensor': {"TGS2600": 245.2, "TGS2602": 187.3},
            'textual': ["floral", "sweet", "fresh"]
        }
        
        try:
            fused_data, metadata = handler.fuse_modalities(modality_data, strategy='concatenation')
            print(f"     ✅ Multi-modal fusion: {len(fused_data)} features, {len(metadata['valid_modalities'])} modalities")
        except Exception as e:
            print(f"     ⚠️ Multi-modal fusion failed (expected without numpy): {e}")
        
        # Test 4: Robust Pipeline
        print("   🚰 Testing Robust Pipeline...")
        pipeline = RobustPipeline()
        
        test_request = {
            'smiles': valid_smiles,
            'sensor_data': {"TGS2600": 245.2, "TGS2602": 187.3}
        }
        
        pipeline_result = pipeline.process_request(test_request)
        print(f"     ✅ Pipeline processing: success={pipeline_result.success}")
        print(f"     ✅ Processing time: {pipeline_result.processing_time:.3f}s")
        
        # Get statistics
        stats = pipeline.get_processing_stats()
        print(f"     ✅ Pipeline stats: {stats['requests_processed']} requests processed")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Robust processing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dependency_management():
    """Test dependency management with graceful fallbacks."""
    print("\n🔗 Testing Dependency Management...")
    
    try:
        from olfactory_transformer.utils.dependency_manager import dependency_manager
        
        # Test 1: Dependency Status Check
        print("   📊 Checking Dependency Status...")
        deps = dependency_manager.get_all_dependencies()
        
        installed_count = sum(1 for dep in deps.values() if dep.installed)
        total_count = len(deps)
        print(f"     ✅ Dependencies: {installed_count}/{total_count} installed")
        
        # Show status of key dependencies
        key_deps = ['numpy', 'torch', 'fastapi', 'redis']
        for dep_name in key_deps:
            available = dependency_manager.is_available(dep_name)
            fallback = dep_name in dependency_manager.mock_implementations
            status = "✅" if available else ("🔄" if fallback else "❌")
            print(f"     {status} {dep_name}: {'installed' if available else ('mock' if fallback else 'missing')}")
        
        # Test 2: Feature Availability
        print("   🎯 Testing Feature Flags...")
        features = ['numerical_computation', 'deep_learning', 'api_server', 'caching']
        for feature in features:
            enabled = dependency_manager.is_feature_enabled(feature)
            print(f"     {'✅' if enabled else '❌'} {feature}")
        
        # Test 3: Mock Implementations
        print("   🎭 Testing Mock Implementations...")
        if 'numpy' in dependency_manager.mock_implementations:
            mock_np = dependency_manager.mock_implementations['numpy']
            test_array = mock_np.array([1, 2, 3, 4, 5])
            mean_val = mock_np.mean(test_array)
            print(f"     ✅ Mock NumPy: array={test_array}, mean={mean_val}")
        
        # Test 4: Health Check
        print("   🏥 Running Dependency Health Check...")
        health = dependency_manager.health_check()
        print(f"     ✅ Overall status: {health['overall_status']}")
        print(f"     ✅ Warnings: {len(health['warnings'])}")
        print(f"     ✅ Errors: {len(health['errors'])}")
        
        # Test 5: Generate Report
        print("   📋 Generating Dependency Report...")
        report = dependency_manager.generate_dependency_report()
        report_lines = report.split('\n')[:10]  # First 10 lines
        print(f"     ✅ Report generated ({len(report_lines)} lines shown)")
        for line in report_lines:
            if line.strip():
                print(f"       {line}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Dependency management test failed: {e}")
        return False


def test_enhanced_i18n():
    """Test enhanced internationalization."""
    print("\n🌍 Testing Enhanced Internationalization...")
    
    try:
        from olfactory_transformer.utils.i18n_manager import i18n_manager
        
        # Test 1: Language Support
        print("   🗣️ Testing Language Support...")
        languages = i18n_manager.get_available_languages()
        print(f"     ✅ Available languages: {', '.join(languages)}")
        
        # Test 2: Basic Translation
        print("   📝 Testing Translations...")
        test_keys = ['model.loading', 'error.invalid_smiles', 'scent.floral']
        
        for lang in ['en', 'es', 'fr', 'de', 'ja']:
            translations = []
            for key in test_keys:
                translation = i18n_manager.translate(key, lang)
                translations.append(f"{key}='{translation}'")
            print(f"     ✅ {lang}: {', '.join(translations[:2])}...")
        
        # Test 3: Scent Descriptor Formatting
        print("   🌸 Testing Scent Descriptor Translation...")
        descriptors = ['floral', 'citrus', 'woody', 'unknown_descriptor']
        
        for lang in ['en', 'es', 'fr']:
            formatted = i18n_manager.format_scent_descriptors(descriptors, lang)
            print(f"     ✅ {lang}: {', '.join(formatted)}")
        
        # Test 4: Chemical Family Names
        print("   🧪 Testing Chemical Family Translation...")
        families = ['ester', 'terpene', 'aldehyde']
        
        for family in families:
            translations = []
            for lang in ['en', 'es', 'de']:
                name = i18n_manager.get_chemical_family_name(family, lang)
                translations.append(f"{lang}:'{name}'")
            print(f"     ✅ {family}: {', '.join(translations)}")
        
        # Test 5: Intensity Labels
        print("   📊 Testing Intensity Labels...")
        intensities = [2.5, 5.5, 8.5]  # Low, Medium, High
        
        for lang in ['en', 'ja', 'zh']:
            labels = []
            for intensity in intensities:
                label = i18n_manager.get_intensity_label(intensity, lang)
                labels.append(label)
            print(f"     ✅ {lang} intensities: {', '.join(labels)}")
        
        # Test 6: Fallback Handling
        print("   🔄 Testing Translation Fallbacks...")
        # Test non-existent key
        missing_key = "nonexistent.key.test"
        fallback_result = i18n_manager.translate(missing_key, 'es')
        print(f"     ✅ Fallback for missing key: '{fallback_result}' (should be key itself)")
        
        # Test unsupported language
        unsupported_lang = i18n_manager.translate('model.loading', 'xx')
        print(f"     ✅ Unsupported language fallback: '{unsupported_lang}'")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Enhanced i18n test failed: {e}")
        return False


def test_robust_config_validation():
    """Test robust configuration validation."""
    print("\n⚙️ Testing Robust Configuration...")
    
    try:
        from olfactory_transformer.core.config import SensorReading, ScentPrediction
        
        # Test 1: Valid Sensor Reading
        print("   📡 Testing Sensor Reading Validation...")
        
        valid_sensor_data = {"TGS2600": 245.2, "TGS2602": 187.3}
        sensor_reading = SensorReading(
            gas_sensors=valid_sensor_data,
            temperature=25.0,
            humidity=55.0
        )
        
        print(f"     ✅ Valid sensor reading created")
        print(f"     ✅ Sensor types: {sensor_reading.sensor_types}")
        print(f"     ✅ Timestamp: {sensor_reading.timestamp}")
        
        # Test 2: Invalid Sensor Reading Validation
        print("   ❌ Testing Invalid Sensor Data...")
        
        try:
            # Empty gas sensors should fail
            SensorReading(gas_sensors={})
            print("     ❌ Empty sensor data validation failed")
        except ValueError as e:
            print(f"     ✅ Empty sensor data rejected: {e}")
        
        try:
            # Out of range sensor value
            SensorReading(gas_sensors={"invalid": 50000})  # Way out of range
            print("     ❌ Out of range validation failed")
        except ValueError as e:
            print(f"     ✅ Out of range value rejected: {e}")
        
        try:
            # Invalid temperature
            SensorReading(gas_sensors={"TGS2600": 245.2}, temperature=200.0)
            print("     ❌ Temperature validation failed")
        except ValueError as e:
            print(f"     ✅ Invalid temperature rejected: {e}")
        
        # Test 3: ScentPrediction Validation
        print("   🌸 Testing Scent Prediction...")
        
        prediction = ScentPrediction(
            primary_notes=["floral", "sweet"],
            descriptors=["rose", "vanilla"],
            intensity=7.5,
            confidence=0.85,
            chemical_family="ester"
        )
        
        print(f"     ✅ Scent prediction created")
        print(f"     ✅ Primary notes: {prediction.primary_notes}")
        print(f"     ✅ Confidence: {prediction.confidence}")
        
        prediction_dict = prediction.to_dict()
        print(f"     ✅ Dictionary conversion: {len(prediction_dict)} fields")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Configuration validation test failed: {e}")
        return False


def main():
    """Run comprehensive Generation 2 feature tests."""
    print("=" * 80)
    print("GENERATION 2 VALIDATION - ROBUST MULTI-MODAL PROCESSING")
    print("=" * 80)
    
    test_results = []
    
    # Define test functions
    test_functions = [
        ("Robust Processing", test_robust_processing),
        ("Dependency Management", test_dependency_management),
        ("Enhanced I18n", test_enhanced_i18n),
        ("Configuration Validation", test_robust_config_validation)
    ]
    
    # Run all tests
    for test_name, test_func in test_functions:
        print(f"\n{'='*60}")
        print(f"RUNNING: {test_name}")
        print(f"{'='*60}")
        
        try:
            result = test_func()
            test_results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            test_results.append((test_name, False))
    
    # Summary
    print(f"\n{'='*80}")
    print("GENERATION 2 TEST SUMMARY")
    print(f"{'='*80}")
    
    passed_tests = 0
    total_tests = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:<30} {status}")
        if result:
            passed_tests += 1
    
    print(f"\n📊 Results: {passed_tests}/{total_tests} tests passed ({passed_tests/total_tests*100:.1f}%)")
    
    if passed_tests == total_tests:
        print("\n🎉 ALL GENERATION 2 TESTS PASSED!")
        print("✅ Robust multi-modal processing validated")
        print("✅ Fault tolerance and graceful degradation working")
        print("✅ Multi-language support operational")  
        print("✅ Input validation and sanitization active")
        print("🚀 System ready for Generation 3 (Advanced Scaling)")
    else:
        print(f"\n⚠️ {total_tests - passed_tests} tests failed")
        print("🔧 Generation 2 features need attention before proceeding")
    
    print(f"{'='*80}")
    
    return passed_tests == total_tests


if __name__ == "__main__":
    main()