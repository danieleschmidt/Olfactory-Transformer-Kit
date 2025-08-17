"""Testing utilities for development and validation."""

import time
import random
import logging
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from .mock_model import MockOlfactoryTransformer


class LoadTester:
    """Load testing utility for development."""
    
    def __init__(self, model=None):
        self.model = model or MockOlfactoryTransformer()
        self.results = []
    
    def run_load_test(self, 
                      test_smiles: List[str], 
                      concurrent_requests: int = 10,
                      total_requests: int = 100) -> Dict[str, Any]:
        """Run a load test with concurrent requests."""
        logging.info(f"🧪 Starting load test: {total_requests} requests, {concurrent_requests} concurrent")
        
        start_time = time.time()
        success_count = 0
        error_count = 0
        response_times = []
        
        with ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
            # Submit all requests
            futures = []
            for i in range(total_requests):
                smiles = random.choice(test_smiles)
                future = executor.submit(self._single_request, smiles, i)
                futures.append(future)
            
            # Collect results
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result['success']:
                        success_count += 1
                        response_times.append(result['response_time'])
                    else:
                        error_count += 1
                except Exception as e:
                    error_count += 1
                    logging.error(f"Request failed: {e}")
        
        total_time = time.time() - start_time
        
        return {
            'total_requests': total_requests,
            'concurrent_requests': concurrent_requests,
            'success_count': success_count,
            'error_count': error_count,
            'total_time': total_time,
            'requests_per_second': total_requests / total_time,
            'average_response_time': sum(response_times) / len(response_times) if response_times else 0,
            'min_response_time': min(response_times) if response_times else 0,
            'max_response_time': max(response_times) if response_times else 0,
            'success_rate': success_count / total_requests * 100
        }
    
    def _single_request(self, smiles: str, request_id: int) -> Dict[str, Any]:
        """Execute a single request and measure performance."""
        start_time = time.time()
        
        try:
            prediction = self.model.predict_scent(smiles)
            response_time = time.time() - start_time
            
            return {
                'request_id': request_id,
                'smiles': smiles,
                'success': True,
                'response_time': response_time,
                'prediction': prediction
            }
        except Exception as e:
            response_time = time.time() - start_time
            return {
                'request_id': request_id,
                'smiles': smiles,
                'success': False,
                'response_time': response_time,
                'error': str(e)
            }


class ValidationSuite:
    """Comprehensive validation suite for the olfactory system."""
    
    def __init__(self, model=None):
        self.model = model or MockOlfactoryTransformer()
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run comprehensive validation tests."""
        logging.info("🔬 Running comprehensive validation suite")
        
        results = {
            'timestamp': time.time(),
            'basic_functionality': self._test_basic_functionality(),
            'edge_cases': self._test_edge_cases(),
            'performance': self._test_performance(),
            'reliability': self._test_reliability(),
            'sensor_integration': self._test_sensor_integration()
        }
        
        # Calculate overall score
        scores = [r.get('score', 0) for r in results.values() if isinstance(r, dict) and 'score' in r]
        results['overall_score'] = sum(scores) / len(scores) if scores else 0
        results['test_passed'] = results['overall_score'] > 80
        
        return results
    
    def _test_basic_functionality(self) -> Dict[str, Any]:
        """Test basic model functionality."""
        test_cases = [
            "CCO",  # Ethanol
            "CC(C)CC1=CC=C(C=C1)C(C)C",  # Lily of the valley
            "CC1=CC=C(C=C1)C(C)(C)C",  # p-tert-Butyl toluene
            "CCOC(=O)C1=CC=CC=C1"  # Ethyl benzoate
        ]
        
        results = []
        for smiles in test_cases:
            try:
                prediction = self.model.predict_scent(smiles)
                results.append({
                    'smiles': smiles,
                    'success': True,
                    'has_notes': len(prediction.primary_notes) > 0,
                    'intensity_valid': 0 <= prediction.intensity <= 10,
                    'confidence_valid': 0 <= prediction.confidence <= 1
                })
            except Exception as e:
                results.append({
                    'smiles': smiles,
                    'success': False,
                    'error': str(e)
                })
        
        success_rate = sum(1 for r in results if r['success']) / len(results) * 100
        
        return {
            'test_name': 'Basic Functionality',
            'total_tests': len(test_cases),
            'passed': sum(1 for r in results if r.get('success', False)),
            'success_rate': success_rate,
            'score': success_rate,
            'details': results
        }
    
    def _test_edge_cases(self) -> Dict[str, Any]:
        """Test edge cases and error handling."""
        edge_cases = [
            "",  # Empty string
            "INVALID_SMILES",  # Invalid SMILES
            "C" * 200,  # Very long string
            "C",  # Single carbon
            "C1=CC=CC=C1" * 10  # Very complex molecule
        ]
        
        results = []
        for case in edge_cases:
            try:
                prediction = self.model.predict_scent(case)
                results.append({
                    'input': case[:50] + "..." if len(case) > 50 else case,
                    'handled_gracefully': True,
                    'has_valid_output': hasattr(prediction, 'primary_notes')
                })
            except Exception as e:
                results.append({
                    'input': case[:50] + "..." if len(case) > 50 else case,
                    'handled_gracefully': False,
                    'error': str(e)
                })
        
        graceful_handling_rate = sum(1 for r in results if r.get('handled_gracefully', False)) / len(results) * 100
        
        return {
            'test_name': 'Edge Cases',
            'total_tests': len(edge_cases),
            'gracefully_handled': sum(1 for r in results if r.get('handled_gracefully', False)),
            'graceful_handling_rate': graceful_handling_rate,
            'score': graceful_handling_rate,
            'details': results
        }
    
    def _test_performance(self) -> Dict[str, Any]:
        """Test performance characteristics."""
        test_smiles = ["CCO", "CC1=CC=CC=C1", "CCOC(=O)C"]
        
        # Single request performance
        start_time = time.time()
        for _ in range(100):
            self.model.predict_scent(random.choice(test_smiles))
        single_thread_time = time.time() - start_time
        
        # Throughput calculation
        throughput = 100 / single_thread_time
        
        return {
            'test_name': 'Performance',
            'requests_tested': 100,
            'total_time': single_thread_time,
            'throughput_rps': throughput,
            'average_response_time': single_thread_time / 100,
            'score': min(100, throughput / 10)  # Score based on throughput
        }
    
    def _test_reliability(self) -> Dict[str, Any]:
        """Test system reliability under stress."""
        test_smiles = ["CCO", "CC1=CC=CC=C1", "CCOC(=O)C"]
        
        # Run multiple iterations
        success_count = 0
        total_count = 50
        
        for _ in range(total_count):
            try:
                prediction = self.model.predict_scent(random.choice(test_smiles))
                if hasattr(prediction, 'primary_notes') and prediction.primary_notes:
                    success_count += 1
            except Exception:
                pass
        
        reliability_score = success_count / total_count * 100
        
        return {
            'test_name': 'Reliability',
            'total_requests': total_count,
            'successful_requests': success_count,
            'reliability_percentage': reliability_score,
            'score': reliability_score
        }
    
    def _test_sensor_integration(self) -> Dict[str, Any]:
        """Test sensor data processing."""
        test_sensor_data = [
            {'TGS2600': 245.0, 'TGS2602': 180.0, 'TGS2610': 320.0},
            {'MQ3': 100.0, 'MQ135': 200.0, 'MQ7': 150.0},
            {'BME680': 50000.0, 'SHT31_temp': 25.0, 'SHT31_humidity': 60.0}
        ]
        
        results = []
        for sensor_data in test_sensor_data:
            try:
                prediction = self.model.predict_from_sensors(sensor_data)
                results.append({
                    'sensor_data': sensor_data,
                    'success': True,
                    'has_prediction': hasattr(prediction, 'primary_notes')
                })
            except Exception as e:
                results.append({
                    'sensor_data': sensor_data,
                    'success': False,
                    'error': str(e)
                })
        
        success_rate = sum(1 for r in results if r['success']) / len(results) * 100
        
        return {
            'test_name': 'Sensor Integration',
            'total_tests': len(test_sensor_data),
            'passed': sum(1 for r in results if r.get('success', False)),
            'success_rate': success_rate,
            'score': success_rate,
            'details': results
        }


def run_development_benchmark():
    """Run a comprehensive development benchmark."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    print("🧪 Olfactory Transformer Development Benchmark")
    print("=" * 50)
    
    # Initialize components
    model = MockOlfactoryTransformer()
    validator = ValidationSuite(model)
    load_tester = LoadTester(model)
    
    # Run validation suite
    print("\n🔬 Running validation suite...")
    validation_results = validator.run_all_tests()
    
    print(f"✅ Validation completed - Overall score: {validation_results['overall_score']:.1f}/100")
    for test_name, result in validation_results.items():
        if isinstance(result, dict) and 'score' in result:
            print(f"  {test_name}: {result['score']:.1f}/100")
    
    # Run load test
    print("\n⚡ Running load test...")
    test_smiles = ["CCO", "CC1=CC=CC=C1", "CCOC(=O)C", "CC(C)CC1=CC=C(C=C1)C(C)C"]
    load_results = load_tester.run_load_test(
        test_smiles=test_smiles,
        concurrent_requests=5,
        total_requests=50
    )
    
    print(f"✅ Load test completed:")
    print(f"  Requests per second: {load_results['requests_per_second']:.1f}")
    print(f"  Success rate: {load_results['success_rate']:.1f}%")
    print(f"  Average response time: {load_results['average_response_time']*1000:.1f}ms")
    
    # Summary
    print("\n📊 Development Benchmark Summary")
    print("=" * 50)
    print(f"🎯 Overall System Score: {validation_results['overall_score']:.1f}/100")
    print(f"⚡ Performance: {load_results['requests_per_second']:.1f} req/sec")
    print(f"🛡️ Reliability: {load_results['success_rate']:.1f}% success rate")
    print(f"🚀 Status: {'READY FOR DEVELOPMENT' if validation_results['overall_score'] > 80 else 'NEEDS ATTENTION'}")
    
    return {
        'validation': validation_results,
        'load_test': load_results,
        'overall_ready': validation_results['overall_score'] > 80 and load_results['success_rate'] > 95
    }


if __name__ == '__main__':
    run_development_benchmark()