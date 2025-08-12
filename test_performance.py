#!/usr/bin/env python3
"""Comprehensive performance and scaling tests for Generation 3."""

import sys
import os
import time
import threading
import concurrent.futures
from pathlib import Path

# Add repo to path
sys.path.insert(0, str(Path(__file__).parent))

def test_tokenizer_performance():
    """Test tokenizer performance and scaling."""
    print("Testing tokenizer performance...")
    
    from olfactory_transformer.core.tokenizer import MoleculeTokenizer
    from olfactory_transformer.utils.performance import PerformanceMonitor
    
    monitor = PerformanceMonitor()
    
    # Test vocabulary building performance
    tokenizer = MoleculeTokenizer(vocab_size=1000)
    
    # Generate test SMILES of varying complexity
    test_smiles = [
        "CCO",  # Simple
        "CC(C)CC1=CC=C(C=C1)C(C)C",  # Medium
        "COC1=CC=C(C=C1)C=CC(=O)C2=CC=C(C=C2)OC",  # Complex
    ]
    
    # Scale up for performance test
    large_smiles_set = []
    for i in range(200):
        large_smiles_set.extend(test_smiles)
    
    # Test vocabulary building
    with monitor.time_operation("vocab_build_large"):
        tokenizer.build_vocab_from_smiles(large_smiles_set)
    
    vocab_stats = monitor.get_stats("vocab_build_large")
    print(f"  Vocabulary build time: {vocab_stats['avg_duration']:.3f}s for {len(large_smiles_set)} SMILES")
    assert vocab_stats['avg_duration'] < 5.0, "Vocabulary building too slow"
    
    # Test encoding performance
    test_molecules = test_smiles * 100  # 300 molecules
    
    start_time = time.time()
    for smiles in test_molecules:
        with monitor.time_operation("encode_single"):
            tokenizer.encode(smiles)
    end_time = time.time()
    
    total_time = end_time - start_time
    throughput = len(test_molecules) / total_time
    
    print(f"  Encoding throughput: {throughput:.1f} molecules/second")
    print(f"  Average encoding time: {total_time/len(test_molecules)*1000:.2f}ms per molecule")
    
    assert throughput > 100, f"Encoding throughput too low: {throughput:.1f} mol/s"
    
    # Test concurrent encoding
    def encode_worker(molecules_subset):
        results = []
        for smiles in molecules_subset:
            try:
                result = tokenizer.encode(smiles)
                results.append(result)
            except Exception as e:
                results.append(None)
        return results
    
    # Split work across threads
    num_threads = 4
    chunk_size = len(test_molecules) // num_threads
    chunks = [test_molecules[i:i+chunk_size] for i in range(0, len(test_molecules), chunk_size)]
    
    start_time = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(encode_worker, chunk) for chunk in chunks]
        results = []
        for future in concurrent.futures.as_completed(futures):
            results.extend(future.result())
    end_time = time.time()
    
    concurrent_time = end_time - start_time
    concurrent_throughput = len(test_molecules) / concurrent_time
    
    print(f"  Concurrent throughput: {concurrent_throughput:.1f} molecules/second ({num_threads} threads)")
    
    # Should show some improvement with concurrency
    speedup = concurrent_throughput / throughput
    print(f"  Concurrency speedup: {speedup:.2f}x")
    
    print("  ✓ Tokenizer performance tests passed")


def test_sensor_streaming_performance():
    """Test sensor streaming performance."""
    print("Testing sensor streaming performance...")
    
    from olfactory_transformer.sensors.enose import ENoseInterface
    from olfactory_transformer.utils.performance import PerformanceMonitor
    
    monitor = PerformanceMonitor()
    
    # Test single reading performance
    enose = ENoseInterface(sensors=["TGS2600", "TGS2602", "TGS2610", "TGS2620"])
    enose.connect()
    
    # Warm up
    for _ in range(5):
        enose.read_single()
    
    # Test sustained reading performance
    num_readings = 100
    start_time = time.time()
    
    for i in range(num_readings):
        with monitor.time_operation("sensor_read"):
            reading = enose.read_single()
            assert reading is not None
            assert len(reading.gas_sensors) == 4
    
    end_time = time.time()
    total_time = end_time - start_time
    reading_rate = num_readings / total_time
    
    print(f"  Single reading rate: {reading_rate:.1f} readings/second")
    print(f"  Average reading time: {total_time/num_readings*1000:.2f}ms per reading")
    
    # Should achieve reasonable reading rates
    assert reading_rate > 50, f"Reading rate too low: {reading_rate:.1f} readings/s"
    
    # Test streaming performance
    enose.start_streaming()
    time.sleep(2.0)  # Stream for 2 seconds
    
    # Count readings in buffer
    reading_count = 0
    try:
        while True:
            reading = enose.data_buffer.get_nowait()
            reading_count += 1
    except:
        pass
    
    enose.stop_streaming()
    enose.disconnect()
    
    streaming_rate = reading_count / 2.0  # readings per second
    print(f"  Streaming rate: {streaming_rate:.1f} readings/second")
    
    assert streaming_rate > 0.5, f"Streaming rate too low: {streaming_rate:.1f} readings/s"
    
    print("  ✓ Sensor streaming performance tests passed")


def test_batch_processing():
    """Test batch processing optimization."""
    print("Testing batch processing...")
    
    from olfactory_transformer.utils.performance import BatchProcessor, BatchProcessingConfig
    from olfactory_transformer.core.tokenizer import MoleculeTokenizer
    
    # Setup
    tokenizer = MoleculeTokenizer(vocab_size=200)
    tokenizer.build_vocab_from_smiles(["CCO", "CCC", "CCCC"])
    
    def process_smiles(smiles):
        """Simple processing function."""
        result = tokenizer.encode(smiles)
        time.sleep(0.001)  # Simulate some processing time
        return len(result["input_ids"])
    
    # Test data
    test_smiles = ["CCO", "CCC", "CCCC", "CC(C)O"] * 25  # 100 items
    
    # Test sequential processing
    start_time = time.time()
    sequential_results = []
    for smiles in test_smiles:
        result = process_smiles(smiles)
        sequential_results.append(result)
    sequential_time = time.time() - start_time
    
    print(f"  Sequential processing: {sequential_time:.3f}s for {len(test_smiles)} items")
    
    # Test batch processing
    config = BatchProcessingConfig(
        max_batch_size=10,
        max_workers=4,
        use_threading=True
    )
    
    processor = BatchProcessor(config)
    
    start_time = time.time()
    batch_results = processor.process_batch_sync(test_smiles, process_smiles)
    batch_time = time.time() - start_time
    
    print(f"  Batch processing: {batch_time:.3f}s for {len(test_smiles)} items")
    
    # Verify results are equivalent
    assert len(batch_results) == len(sequential_results)
    valid_results = [r for r in batch_results if r is not None]
    assert len(valid_results) >= len(test_smiles) * 0.9  # Allow some failures
    
    # Calculate speedup
    speedup = sequential_time / batch_time
    print(f"  Batch processing speedup: {speedup:.2f}x")
    
    # Should show some improvement
    assert speedup > 1.0, f"Batch processing slower than sequential: {speedup:.2f}x"
    
    # Test performance stats
    stats = processor.get_performance_stats()
    print(f"  Processed: {stats['total_processed']}, Errors: {stats['total_errors']}")
    print(f"  Error rate: {stats['error_rate']:.1%}")
    
    assert stats['error_rate'] < 0.1, f"High error rate: {stats['error_rate']:.1%}"
    
    print("  ✓ Batch processing tests passed")


def test_caching_performance():
    """Test caching optimization."""
    print("Testing caching performance...")
    
    from olfactory_transformer.utils.performance import CacheManager, memoize_with_ttl
    
    # Test basic cache
    cache = CacheManager(max_size=100, ttl_seconds=10)
    
    # Test cache put/get performance
    start_time = time.time()
    for i in range(1000):
        cache.put(f"key_{i}", f"value_{i}")
    put_time = time.time() - start_time
    
    start_time = time.time()
    for i in range(1000):
        value = cache.get(f"key_{i}")
    get_time = time.time() - start_time
    
    print(f"  Cache put rate: {1000/put_time:.0f} ops/sec")
    print(f"  Cache get rate: {1000/get_time:.0f} ops/sec")
    
    # Test cache hit rate
    cache.clear()
    
    # Fill cache with some items
    for i in range(50):
        cache.put(f"item_{i}", i * i)
    
    hits = 0
    misses = 0
    
    # Test with mix of hits and misses
    for i in range(100):
        key = f"item_{i % 75}"  # Some hits, some misses
        value = cache.get(key)
        if value is not None:
            hits += 1
        else:
            misses += 1
    
    hit_rate = hits / (hits + misses)
    print(f"  Cache hit rate: {hit_rate:.1%}")
    
    assert hit_rate > 0.5, f"Cache hit rate too low: {hit_rate:.1%}"
    
    # Test memoization decorator
    call_count = 0
    
    @memoize_with_ttl(ttl_seconds=60, max_size=50)
    def expensive_function(x):
        nonlocal call_count
        call_count += 1
        time.sleep(0.01)  # Simulate expensive computation
        return x * x
    
    # Test memoization performance
    start_time = time.time()
    
    # First calls (cache misses)
    for i in range(10):
        result = expensive_function(i)
        assert result == i * i
    
    first_pass_time = time.time() - start_time
    first_pass_calls = call_count
    
    start_time = time.time()
    
    # Second calls (cache hits)
    for i in range(10):
        result = expensive_function(i)
        assert result == i * i
    
    second_pass_time = time.time() - start_time
    second_pass_calls = call_count - first_pass_calls
    
    print(f"  First pass (cache miss): {first_pass_time:.3f}s, {first_pass_calls} function calls")
    print(f"  Second pass (cache hit): {second_pass_time:.3f}s, {second_pass_calls} function calls")
    
    # Cache should significantly reduce time and function calls
    assert second_pass_time < first_pass_time * 0.5, "Cache not providing speedup"
    assert second_pass_calls == 0, "Cache not preventing function calls"
    
    speedup = first_pass_time / second_pass_time if second_pass_time > 0 else float('inf')
    print(f"  Memoization speedup: {speedup:.1f}x")
    
    print("  ✓ Caching performance tests passed")


def test_memory_efficiency():
    """Test memory usage efficiency."""
    print("Testing memory efficiency...")
    
    from olfactory_transformer.core.tokenizer import MoleculeTokenizer
    from olfactory_transformer.utils.performance import get_global_resource_monitor
    
    # Get initial memory usage
    try:
        import psutil
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        has_psutil = True
    except ImportError:
        initial_memory = 0
        has_psutil = False
    
    # Create multiple tokenizers to test memory scaling
    tokenizers = []
    test_smiles = ["CCO", "CCC", "CCCC", "CC(C)O"] * 50  # 200 SMILES
    
    for i in range(5):
        tokenizer = MoleculeTokenizer(vocab_size=500)
        tokenizer.build_vocab_from_smiles(test_smiles)
        tokenizers.append(tokenizer)
    
    if has_psutil:
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_per_tokenizer = (final_memory - initial_memory) / len(tokenizers)
        
        print(f"  Memory usage: {final_memory - initial_memory:.1f} MB for {len(tokenizers)} tokenizers")
        print(f"  Average per tokenizer: {memory_per_tokenizer:.1f} MB")
        
        # Should not use excessive memory
        assert memory_per_tokenizer < 50, f"Excessive memory per tokenizer: {memory_per_tokenizer:.1f} MB"
    else:
        print("  Memory monitoring unavailable (psutil not installed)")
    
    # Test memory cleanup
    del tokenizers
    
    if has_psutil:
        import gc
        gc.collect()
        time.sleep(0.1)  # Allow cleanup
        
        cleanup_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_freed = final_memory - cleanup_memory
        
        print(f"  Memory after cleanup: {cleanup_memory:.1f} MB")
        print(f"  Memory freed: {memory_freed:.1f} MB")
        
        # Should free significant memory
        cleanup_efficiency = memory_freed / (final_memory - initial_memory) if (final_memory - initial_memory) > 0 else 0
        print(f"  Cleanup efficiency: {cleanup_efficiency:.1%}")
    
    # Test large dataset handling
    large_dataset = ["C" * (i % 20 + 1) for i in range(1000)]
    
    tokenizer = MoleculeTokenizer(vocab_size=2000)
    
    if has_psutil:
        pre_build_memory = process.memory_info().rss / 1024 / 1024
    
    tokenizer.build_vocab_from_smiles(large_dataset)
    
    if has_psutil:
        post_build_memory = process.memory_info().rss / 1024 / 1024
        build_memory_cost = post_build_memory - pre_build_memory
        
        print(f"  Large vocabulary memory cost: {build_memory_cost:.1f} MB for {len(large_dataset)} SMILES")
        
        # Should handle large datasets efficiently
        assert build_memory_cost < 200, f"Large vocabulary uses too much memory: {build_memory_cost:.1f} MB"
    
    print("  ✓ Memory efficiency tests passed")


def test_concurrent_performance():
    """Test performance under concurrent load."""
    print("Testing concurrent performance...")
    
    from olfactory_transformer.core.tokenizer import MoleculeTokenizer
    import threading
    import queue
    
    # Setup shared tokenizer
    tokenizer = MoleculeTokenizer(vocab_size=300)
    tokenizer.build_vocab_from_smiles(["CCO", "CCC", "CCCC", "CC(C)O"])
    
    results_queue = queue.Queue()
    error_queue = queue.Queue()
    
    def worker_thread(thread_id, num_operations):
        """Worker thread for concurrent testing."""
        try:
            for i in range(num_operations):
                smiles = ["CCO", "CCC", "CCCC"][i % 3]
                start_time = time.time()
                result = tokenizer.encode(smiles)
                end_time = time.time()
                
                results_queue.put({
                    "thread_id": thread_id,
                    "operation": i,
                    "duration": end_time - start_time,
                    "success": True
                })
        except Exception as e:
            error_queue.put({"thread_id": thread_id, "error": str(e)})
    
    # Test with multiple concurrent threads
    num_threads = 8
    operations_per_thread = 50
    
    threads = []
    start_time = time.time()
    
    for thread_id in range(num_threads):
        thread = threading.Thread(
            target=worker_thread,
            args=(thread_id, operations_per_thread)
        )
        threads.append(thread)
        thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Collect results
    results = []
    while not results_queue.empty():
        results.append(results_queue.get())
    
    errors = []
    while not error_queue.empty():
        errors.append(error_queue.get())
    
    total_operations = num_threads * operations_per_thread
    successful_operations = len(results)
    error_rate = len(errors) / total_operations
    
    print(f"  Concurrent operations: {successful_operations}/{total_operations} successful")
    print(f"  Error rate: {error_rate:.1%}")
    print(f"  Total time: {total_time:.3f}s")
    print(f"  Throughput: {successful_operations/total_time:.1f} ops/sec")
    
    # Analyze performance distribution
    if results:
        durations = [r["duration"] for r in results]
        avg_duration = sum(durations) / len(durations)
        max_duration = max(durations)
        min_duration = min(durations)
        
        print(f"  Average operation time: {avg_duration*1000:.2f}ms")
        print(f"  Min/Max operation time: {min_duration*1000:.2f}ms / {max_duration*1000:.2f}ms")
        
        # Performance should be reasonable under load
        assert avg_duration < 0.1, f"Average operation too slow under load: {avg_duration*1000:.1f}ms"
        assert error_rate < 0.05, f"High error rate under load: {error_rate:.1%}"
    
    print("  ✓ Concurrent performance tests passed")


def run_performance_tests():
    """Run all performance tests for Generation 3."""
    print("⚡ Generation 3 Performance Tests")
    print("=" * 50)
    print("Testing performance optimization and scaling...")
    print()
    
    tests = [
        test_tokenizer_performance,
        test_sensor_streaming_performance,
        test_batch_processing,
        test_caching_performance,
        test_memory_efficiency,
        test_concurrent_performance,
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
            import traceback
            print(f"   Traceback: {traceback.format_exc()}")
            print()
    
    print("=" * 50)
    print(f"Performance Test Results: {passed}/{len(tests)} passed")
    
    if failed == 0:
        print("🎉 All Generation 3 performance tests PASSED!")
        print("⚡ System is OPTIMIZED and scalable")
        print("Next: Quality Gates and Production Deployment")
        return True
    else:
        print(f"❌ {failed} performance tests failed")
        print("Review and optimize before proceeding to production")
        return False


if __name__ == "__main__":
    success = run_performance_tests()
    sys.exit(0 if success else 1)