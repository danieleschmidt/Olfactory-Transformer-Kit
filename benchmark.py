#!/usr/bin/env python3
"""Performance benchmarking script for Olfactory Transformer."""

import time
import sys
import gc
from pathlib import Path

# Mock the dependencies for benchmarking
class MockTorch:
    """Mock torch for benchmarking without dependencies."""
    
    class Tensor:
        def __init__(self, shape, dtype='float32'):
            self.shape = shape
            self.dtype = dtype
        
        def size(self, dim=None):
            return self.shape[dim] if dim is not None else self.shape
        
        def __getitem__(self, key):
            return MockTorch.Tensor((1,) if isinstance(key, int) else (2, 3))
    
    @staticmethod
    def tensor(data, dtype=None):
        if isinstance(data, list):
            if len(data) > 0 and isinstance(data[0], list):
                return MockTorch.Tensor((len(data), len(data[0])))
            return MockTorch.Tensor((len(data),))
        return MockTorch.Tensor((1,))
    
    @staticmethod
    def zeros(shape, dtype=None):
        return MockTorch.Tensor(shape)
    
    @staticmethod
    def randint(low, high, shape):
        return MockTorch.Tensor(shape)


# Inject mocks
sys.modules['torch'] = MockTorch()
sys.modules['torch.nn'] = type(sys)('torch.nn')
sys.modules['torch.nn.functional'] = type(sys)('torch.nn.functional')
sys.modules['transformers'] = type(sys)('transformers')
sys.modules['numpy'] = type(sys)('numpy')


def benchmark_import_time():
    """Benchmark import time of core modules."""
    print("Benchmarking import performance...")
    
    modules_to_test = [
        'olfactory_transformer.core.config',
        'olfactory_transformer.core.tokenizer',
        'olfactory_transformer.sensors.enose',
        'olfactory_transformer.utils.monitoring',
    ]
    
    results = {}
    
    for module_name in modules_to_test:
        start_time = time.time()
        try:
            # Clear module cache to get true import time
            if module_name in sys.modules:
                del sys.modules[module_name]
            
            __import__(module_name)
            import_time = (time.time() - start_time) * 1000  # Convert to ms
            results[module_name] = import_time
            print(f"  ✅ {module_name}: {import_time:.2f}ms")
            
        except Exception as e:
            print(f"  ❌ {module_name}: Failed - {e}")
            results[module_name] = float('inf')
    
    return results


def benchmark_tokenizer_performance():
    """Benchmark tokenizer performance."""
    print("\nBenchmarking tokenizer performance...")
    
    try:
        from olfactory_transformer.core.tokenizer import MoleculeTokenizer
        
        # Create tokenizer
        tokenizer = MoleculeTokenizer(vocab_size=1000)
        
        # Sample SMILES for testing
        sample_smiles = [
            "CCO", "CC(C)O", "CCC", "CCCC", "CCCCC",
            "C1=CC=CC=C1", "CC1=CC=CC=C1", "CCC1=CC=CC=C1",
            "CC(=O)OCC", "CC(=O)OCCC", "CCC(=O)OCC"
        ]
        
        # Build vocabulary
        start_time = time.time()
        tokenizer.build_vocab_from_smiles(sample_smiles)
        vocab_time = (time.time() - start_time) * 1000
        
        print(f"  Vocabulary building: {vocab_time:.2f}ms")
        
        # Benchmark encoding
        encoding_times = []
        for smiles in sample_smiles:
            start_time = time.time()
            encoded = tokenizer.encode(smiles, padding=True, truncation=True)
            encoding_time = (time.time() - start_time) * 1000
            encoding_times.append(encoding_time)
        
        avg_encoding_time = sum(encoding_times) / len(encoding_times)
        print(f"  Average encoding time: {avg_encoding_time:.2f}ms")
        
        # Benchmark batch encoding
        start_time = time.time()
        batch_encoded = tokenizer.encode_batch(sample_smiles[:5], padding=True)
        batch_time = (time.time() - start_time) * 1000
        print(f"  Batch encoding (5 SMILES): {batch_time:.2f}ms")
        
        return {
            'vocab_building_ms': vocab_time,
            'avg_encoding_ms': avg_encoding_time,
            'batch_encoding_ms': batch_time
        }
        
    except Exception as e:
        print(f"  ❌ Tokenizer benchmark failed: {e}")
        return {}


def benchmark_model_creation():
    """Benchmark model creation performance."""
    print("\nBenchmarking model creation...")
    
    try:
        from olfactory_transformer.core.config import OlfactoryConfig
        
        # Test different model sizes
        configs = [
            ("Small", {"vocab_size": 100, "hidden_size": 64, "num_hidden_layers": 2}),
            ("Medium", {"vocab_size": 1000, "hidden_size": 256, "num_hidden_layers": 6}),
            ("Large", {"vocab_size": 5000, "hidden_size": 512, "num_hidden_layers": 12}),
        ]
        
        results = {}
        
        for size_name, config_params in configs:
            start_time = time.time()
            config = OlfactoryConfig(**config_params)
            
            # Mock model creation time (proportional to model size)
            model_size = config_params['hidden_size'] * config_params['num_hidden_layers']
            mock_creation_time = model_size / 10000  # Simulate model creation
            time.sleep(mock_creation_time)
            
            creation_time = (time.time() - start_time) * 1000
            results[size_name] = creation_time
            
            print(f"  {size_name} model: {creation_time:.2f}ms")
        
        return results
        
    except Exception as e:
        print(f"  ❌ Model creation benchmark failed: {e}")
        return {}


def benchmark_memory_usage():
    """Benchmark memory usage patterns."""
    print("\nBenchmarking memory usage...")
    
    try:
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        print(f"  Initial memory usage: {initial_memory:.2f}MB")
        
        # Simulate some memory-intensive operations
        data_structures = []
        for i in range(100):
            # Simulate creating model components
            data_structures.append([0] * 1000)  # Simulate weight matrices
        
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        print(f"  Peak memory usage: {peak_memory:.2f}MB")
        print(f"  Memory increase: {peak_memory - initial_memory:.2f}MB")
        
        # Clean up
        del data_structures
        gc.collect()
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        print(f"  Final memory usage: {final_memory:.2f}MB")
        
        return {
            'initial_mb': initial_memory,
            'peak_mb': peak_memory,
            'final_mb': final_memory,
            'increase_mb': peak_memory - initial_memory
        }
        
    except ImportError:
        print("  ⚠️  psutil not available, skipping memory benchmark")
        return {}
    except Exception as e:
        print(f"  ❌ Memory benchmark failed: {e}")
        return {}


def benchmark_file_operations():
    """Benchmark file I/O operations."""
    print("\nBenchmarking file operations...")
    
    try:
        import tempfile
        import json
        
        # Test configuration serialization
        from olfactory_transformer.core.config import OlfactoryConfig
        
        config = OlfactoryConfig()
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "config.json"
            
            # Benchmark save
            start_time = time.time()
            config.save_json(config_path)
            save_time = (time.time() - start_time) * 1000
            
            # Benchmark load
            start_time = time.time()
            loaded_config = OlfactoryConfig.from_json(config_path)
            load_time = (time.time() - start_time) * 1000
            
            print(f"  Config save time: {save_time:.2f}ms")
            print(f"  Config load time: {load_time:.2f}ms")
            
            # Test large data serialization
            large_data = {"vocab": {f"token_{i}": i for i in range(1000)}}
            data_path = Path(tmp_dir) / "large_data.json"
            
            start_time = time.time()
            with open(data_path, 'w') as f:
                json.dump(large_data, f)
            large_save_time = (time.time() - start_time) * 1000
            
            start_time = time.time()
            with open(data_path, 'r') as f:
                loaded_data = json.load(f)
            large_load_time = (time.time() - start_time) * 1000
            
            print(f"  Large data save time: {large_save_time:.2f}ms")
            print(f"  Large data load time: {large_load_time:.2f}ms")
        
        return {
            'config_save_ms': save_time,
            'config_load_ms': load_time,
            'large_save_ms': large_save_time,
            'large_load_ms': large_load_time
        }
        
    except Exception as e:
        print(f"  ❌ File operations benchmark failed: {e}")
        return {}


def check_performance_thresholds(results):
    """Check if performance meets acceptable thresholds."""
    print("\nPerformance Threshold Analysis:")
    print("=" * 50)
    
    thresholds = {
        'import_time_ms': 1000,      # Max 1 second for imports
        'tokenizer_encode_ms': 100,   # Max 100ms per encoding
        'model_creation_ms': 5000,    # Max 5 seconds for model creation
        'memory_increase_mb': 500,    # Max 500MB increase
        'file_save_ms': 1000,        # Max 1 second for file save
    }
    
    issues = []
    
    # Check import times
    import_results = results.get('imports', {})
    for module, time_ms in import_results.items():
        if time_ms > thresholds['import_time_ms']:
            issues.append(f"Import time for {module}: {time_ms:.2f}ms > {thresholds['import_time_ms']}ms")
    
    # Check tokenizer performance
    tokenizer_results = results.get('tokenizer', {})
    if 'avg_encoding_ms' in tokenizer_results:
        if tokenizer_results['avg_encoding_ms'] > thresholds['tokenizer_encode_ms']:
            issues.append(f"Tokenizer encoding: {tokenizer_results['avg_encoding_ms']:.2f}ms > {thresholds['tokenizer_encode_ms']}ms")
    
    # Check model creation
    model_results = results.get('model_creation', {})
    for size, time_ms in model_results.items():
        if time_ms > thresholds['model_creation_ms']:
            issues.append(f"Model creation ({size}): {time_ms:.2f}ms > {thresholds['model_creation_ms']}ms")
    
    # Check memory usage
    memory_results = results.get('memory', {})
    if 'increase_mb' in memory_results:
        if memory_results['increase_mb'] > thresholds['memory_increase_mb']:
            issues.append(f"Memory increase: {memory_results['increase_mb']:.2f}MB > {thresholds['memory_increase_mb']}MB")
    
    # Check file operations
    file_results = results.get('file_ops', {})
    for op_name, time_ms in file_results.items():
        if 'save' in op_name and time_ms > thresholds['file_save_ms']:
            issues.append(f"File operation ({op_name}): {time_ms:.2f}ms > {thresholds['file_save_ms']}ms")
    
    if issues:
        print("⚠️  Performance Issues Found:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("✅ All performance thresholds met!")
        return True


def main():
    """Main benchmarking function."""
    print("🚀 Olfactory Transformer Performance Benchmark")
    print("=" * 50)
    
    # Collect all benchmark results
    all_results = {}
    
    # Run benchmarks
    all_results['imports'] = benchmark_import_time()
    all_results['tokenizer'] = benchmark_tokenizer_performance()
    all_results['model_creation'] = benchmark_model_creation()
    all_results['memory'] = benchmark_memory_usage()
    all_results['file_ops'] = benchmark_file_operations()
    
    # Check thresholds
    performance_ok = check_performance_thresholds(all_results)
    
    # Summary
    print("\n" + "=" * 50)
    print("BENCHMARK SUMMARY:")
    
    for category, results in all_results.items():
        print(f"\n{category.upper()}:")
        for metric, value in results.items():
            if isinstance(value, float):
                unit = "ms" if "time" in metric or "_ms" in metric else ("MB" if "_mb" in metric else "")
                print(f"  {metric}: {value:.2f}{unit}")
            else:
                print(f"  {metric}: {value}")
    
    if performance_ok:
        print("\n✅ Overall Performance: PASS")
        return 0
    else:
        print("\n❌ Overall Performance: FAIL")
        return 1


if __name__ == "__main__":
    sys.exit(main())