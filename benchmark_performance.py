#!/usr/bin/env python3
"""
Performance benchmarking script for Olfactory Transformer
Tests throughput, latency, memory usage, and scalability
"""

import time
import logging
import statistics
from typing import List, Dict, Any
import concurrent.futures
import threading
import gc
import sys
from pathlib import Path

import torch
import psutil
import numpy as np

# Add the project root to path
sys.path.insert(0, str(Path(__file__).parent))

from olfactory_transformer import OlfactoryTransformer, MoleculeTokenizer
from olfactory_transformer.core.config import OlfactoryConfig
from olfactory_transformer.utils.observability import observability_manager
from olfactory_transformer.utils.reliability import reliability_manager


class PerformanceBenchmark:
    """Comprehensive performance benchmarking suite."""
    
    def __init__(self):
        self.results = {}
        self.setup_logging()
        
        # Test molecules for benchmarking
        self.test_molecules = [
            "CCO",  # Ethanol
            "CC(C)O",  # Isopropanol  
            "CCC",  # Propane
            "C1=CC=CC=C1",  # Benzene
            "CC(=O)OCC",  # Ethyl acetate
            "COC1=CC(=CC=C1O)C=O",  # Vanillin
            "CC(C)CC1=CC=C(C=C1)C(C)C",  # Lily aldehyde
            "C1=CC=C(C=C1)C=O",  # Benzaldehyde
            "CCc1ccc(cc1)O",  # 4-Ethylphenol
            "CC(C)(C)C1=CC=C(C=C1)O",  # 4-tert-Butylphenol
        ]
        
    def setup_logging(self):
        """Setup logging for benchmark."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def create_test_model(self, config_name: str = "small") -> tuple:
        """Create model and tokenizer for testing."""
        if config_name == "small":
            config = OlfactoryConfig(
                vocab_size=200,
                hidden_size=128,
                num_hidden_layers=4,
                num_attention_heads=8,
                molecular_features=64,
                max_position_embeddings=256,
            )
        elif config_name == "medium":
            config = OlfactoryConfig(
                vocab_size=500,
                hidden_size=256,
                num_hidden_layers=8,
                num_attention_heads=8,
                molecular_features=128,
                max_position_embeddings=512,
            )
        elif config_name == "large":
            config = OlfactoryConfig(
                vocab_size=1000,
                hidden_size=512,
                num_hidden_layers=12,
                num_attention_heads=16,
                molecular_features=256,
                max_position_embeddings=1024,
            )
        else:
            raise ValueError(f"Unknown config: {config_name}")
        
        model = OlfactoryTransformer(config)
        model.eval()
        
        tokenizer = MoleculeTokenizer(vocab_size=config.vocab_size)
        tokenizer.build_vocab_from_smiles(self.test_molecules + [
            "CCCC", "CCCCC", "CCCCCC", "C1CCCC1", "CC1=CC=CC=C1"
        ])
        
        return model, tokenizer, config
    
    def benchmark_single_inference(self, model, tokenizer, num_runs: int = 100) -> Dict[str, Any]:
        """Benchmark single inference performance."""
        self.logger.info(f"Running single inference benchmark ({num_runs} runs)...")
        
        latencies = []
        memory_usage = []
        
        # Warmup
        for _ in range(10):
            _ = model.predict_scent(self.test_molecules[0], tokenizer)
        
        # Clear memory and start fresh
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        # Benchmark runs
        for i in range(num_runs):
            molecule = self.test_molecules[i % len(self.test_molecules)]
            
            # Memory before
            memory_before = psutil.Process().memory_info().rss / 1024**2  # MB
            if torch.cuda.is_available():
                gpu_memory_before = torch.cuda.memory_allocated() / 1024**2  # MB
            
            # Time inference
            start_time = time.time()
            
            try:
                prediction = model.predict_scent(molecule, tokenizer)
                success = True
            except Exception as e:
                self.logger.error(f"Inference failed: {e}")
                success = False
            
            end_time = time.time()
            
            # Memory after
            memory_after = psutil.Process().memory_info().rss / 1024**2  # MB
            
            if success:
                latency_ms = (end_time - start_time) * 1000
                latencies.append(latency_ms)
                memory_usage.append(memory_after - memory_before)
        
        return {
            "num_runs": len(latencies),
            "success_rate": len(latencies) / num_runs,
            "avg_latency_ms": statistics.mean(latencies) if latencies else 0,
            "median_latency_ms": statistics.median(latencies) if latencies else 0,
            "p95_latency_ms": np.percentile(latencies, 95) if latencies else 0,
            "p99_latency_ms": np.percentile(latencies, 99) if latencies else 0,
            "min_latency_ms": min(latencies) if latencies else 0,
            "max_latency_ms": max(latencies) if latencies else 0,
            "avg_memory_mb": statistics.mean(memory_usage) if memory_usage else 0,
        }
    
    def benchmark_throughput(self, model, tokenizer, duration_seconds: int = 30) -> Dict[str, Any]:
        """Benchmark maximum throughput."""
        self.logger.info(f"Running throughput benchmark ({duration_seconds}s)...")
        
        start_time = time.time()
        end_time = start_time + duration_seconds
        
        total_requests = 0
        successful_requests = 0
        errors = []
        
        while time.time() < end_time:
            molecule = self.test_molecules[total_requests % len(self.test_molecules)]
            total_requests += 1
            
            try:
                _ = model.predict_scent(molecule, tokenizer)
                successful_requests += 1
            except Exception as e:
                errors.append(str(e))
        
        actual_duration = time.time() - start_time
        
        return {
            "duration_seconds": actual_duration,
            "total_requests": total_requests,
            "successful_requests": successful_requests,
            "error_count": len(errors),
            "throughput_rps": successful_requests / actual_duration,
            "success_rate": successful_requests / total_requests if total_requests > 0 else 0,
        }
    
    def benchmark_concurrent_load(self, model, tokenizer, num_threads: int = 4, 
                                 requests_per_thread: int = 25) -> Dict[str, Any]:
        """Benchmark concurrent load handling."""
        self.logger.info(f"Running concurrent load benchmark ({num_threads} threads, "
                        f"{requests_per_thread} requests each)...")
        
        def worker_task(thread_id: int) -> Dict[str, Any]:
            """Worker task for concurrent testing."""
            thread_latencies = []
            thread_errors = 0
            
            for i in range(requests_per_thread):
                molecule = self.test_molecules[(thread_id * requests_per_thread + i) % len(self.test_molecules)]
                
                start_time = time.time()
                try:
                    _ = model.predict_scent(molecule, tokenizer)
                    latency = (time.time() - start_time) * 1000
                    thread_latencies.append(latency)
                except Exception:
                    thread_errors += 1
            
            return {
                "thread_id": thread_id,
                "latencies": thread_latencies,
                "errors": thread_errors,
            }
        
        # Run concurrent workers
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker_task, i) for i in range(num_threads)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        end_time = time.time()
        
        # Aggregate results
        all_latencies = []
        total_errors = 0
        
        for result in results:
            all_latencies.extend(result["latencies"])
            total_errors += result["errors"]
        
        total_requests = num_threads * requests_per_thread
        successful_requests = len(all_latencies)
        
        return {
            "num_threads": num_threads,
            "requests_per_thread": requests_per_thread,
            "total_requests": total_requests,
            "successful_requests": successful_requests,
            "total_errors": total_errors,
            "duration_seconds": end_time - start_time,
            "concurrent_throughput_rps": successful_requests / (end_time - start_time),
            "avg_latency_ms": statistics.mean(all_latencies) if all_latencies else 0,
            "p95_latency_ms": np.percentile(all_latencies, 95) if all_latencies else 0,
            "success_rate": successful_requests / total_requests if total_requests > 0 else 0,
        }
    
    def benchmark_memory_scaling(self, model, tokenizer, max_batch_size: int = 32) -> Dict[str, Any]:
        """Benchmark memory usage scaling."""
        self.logger.info("Running memory scaling benchmark...")
        
        memory_results = []
        
        for batch_size in [1, 2, 4, 8, 16, min(32, max_batch_size)]:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            # Measure memory before
            memory_before = psutil.Process().memory_info().rss / 1024**2
            gpu_memory_before = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
            
            try:
                # Simulate batch processing
                start_time = time.time()
                for i in range(batch_size):
                    molecule = self.test_molecules[i % len(self.test_molecules)]
                    _ = model.predict_scent(molecule, tokenizer)
                
                inference_time = time.time() - start_time
                
                # Measure memory after
                memory_after = psutil.Process().memory_info().rss / 1024**2
                gpu_memory_after = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
                
                memory_results.append({
                    "batch_size": batch_size,
                    "cpu_memory_mb": memory_after - memory_before,
                    "gpu_memory_mb": gpu_memory_after - gpu_memory_before,
                    "inference_time_s": inference_time,
                    "memory_per_item_mb": (memory_after - memory_before) / batch_size,
                })
                
            except Exception as e:
                self.logger.error(f"Memory scaling test failed at batch_size {batch_size}: {e}")
                break
        
        return {
            "memory_scaling_results": memory_results,
            "max_successful_batch_size": max(r["batch_size"] for r in memory_results) if memory_results else 0,
        }
    
    def benchmark_model_sizes(self) -> Dict[str, Any]:
        """Benchmark different model sizes."""
        self.logger.info("Running model size comparison benchmark...")
        
        size_results = {}
        
        for size in ["small", "medium"]:  # Skip large for CI/memory constraints
            self.logger.info(f"Benchmarking {size} model...")
            
            try:
                model, tokenizer, config = self.create_test_model(size)
                
                # Model statistics
                num_params = sum(p.numel() for p in model.parameters())
                model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2
                
                # Performance tests
                single_inference = self.benchmark_single_inference(model, tokenizer, num_runs=50)
                throughput = self.benchmark_throughput(model, tokenizer, duration_seconds=15)
                
                size_results[size] = {
                    "config": {
                        "vocab_size": config.vocab_size,
                        "hidden_size": config.hidden_size,
                        "num_layers": config.num_hidden_layers,
                        "num_attention_heads": config.num_attention_heads,
                    },
                    "model_stats": {
                        "num_parameters": num_params,
                        "model_size_mb": model_size_mb,
                    },
                    "performance": {
                        "single_inference": single_inference,
                        "throughput": throughput,
                    }
                }
                
                # Cleanup
                del model, tokenizer
                gc.collect()
                
            except Exception as e:
                self.logger.error(f"Failed to benchmark {size} model: {e}")
                size_results[size] = {"error": str(e)}
        
        return size_results
    
    def run_full_benchmark(self) -> Dict[str, Any]:
        """Run complete performance benchmark suite."""
        self.logger.info("Starting comprehensive performance benchmark...")
        
        benchmark_results = {
            "benchmark_info": {
                "timestamp": time.time(),
                "system_info": self.get_system_info(),
            }
        }
        
        try:
            # Model size comparison
            benchmark_results["model_sizes"] = self.benchmark_model_sizes()
            
            # Detailed benchmarks with medium model
            self.logger.info("Running detailed benchmarks with medium model...")
            model, tokenizer, config = self.create_test_model("medium")
            
            benchmark_results["detailed_benchmarks"] = {
                "single_inference": self.benchmark_single_inference(model, tokenizer, num_runs=100),
                "throughput": self.benchmark_throughput(model, tokenizer, duration_seconds=30),
                "concurrent_load": self.benchmark_concurrent_load(model, tokenizer, num_threads=4, requests_per_thread=25),
                "memory_scaling": self.benchmark_memory_scaling(model, tokenizer, max_batch_size=16),
            }
            
            # System status
            benchmark_results["system_status"] = {
                "reliability": reliability_manager.get_system_status(),
                "observability": observability_manager.get_observability_status(),
            }
            
        except Exception as e:
            self.logger.error(f"Benchmark failed: {e}")
            benchmark_results["error"] = str(e)
        
        return benchmark_results
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        return {
            "cpu_count": psutil.cpu_count(),
            "memory_total_gb": psutil.virtual_memory().total / 1024**3,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "gpu_name": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        }
    
    def print_results(self, results: Dict[str, Any]):
        """Print benchmark results in a readable format."""
        print("\n" + "="*60)
        print("OLFACTORY TRANSFORMER PERFORMANCE BENCHMARK RESULTS")
        print("="*60)
        
        # System info
        system_info = results.get("benchmark_info", {}).get("system_info", {})
        print(f"\n📊 System Information:")
        print(f"  CPU Cores: {system_info.get('cpu_count', 'Unknown')}")
        print(f"  Memory: {system_info.get('memory_total_gb', 0):.1f} GB")
        print(f"  PyTorch: {system_info.get('torch_version', 'Unknown')}")
        print(f"  CUDA Available: {system_info.get('cuda_available', False)}")
        if system_info.get('cuda_available'):
            print(f"  GPU: {system_info.get('gpu_name', 'Unknown')}")
        
        # Model sizes comparison
        model_sizes = results.get("model_sizes", {})
        if model_sizes:
            print(f"\n🏗️  Model Size Comparison:")
            for size, data in model_sizes.items():
                if "error" not in data:
                    config = data["config"]
                    stats = data["model_stats"]
                    perf = data["performance"]["single_inference"]
                    print(f"  {size.upper()} Model:")
                    print(f"    Parameters: {stats['num_parameters']:,}")
                    print(f"    Size: {stats['model_size_mb']:.1f} MB")
                    print(f"    Avg Latency: {perf['avg_latency_ms']:.1f} ms")
                    print(f"    P95 Latency: {perf['p95_latency_ms']:.1f} ms")
        
        # Detailed benchmarks
        detailed = results.get("detailed_benchmarks", {})
        
        if "single_inference" in detailed:
            perf = detailed["single_inference"]
            print(f"\n⚡ Single Inference Performance:")
            print(f"  Average Latency: {perf['avg_latency_ms']:.1f} ms")
            print(f"  Median Latency: {perf['median_latency_ms']:.1f} ms")
            print(f"  P95 Latency: {perf['p95_latency_ms']:.1f} ms")
            print(f"  P99 Latency: {perf['p99_latency_ms']:.1f} ms")
            print(f"  Success Rate: {perf['success_rate']:.1%}")
        
        if "throughput" in detailed:
            throughput = detailed["throughput"]
            print(f"\n🚀 Throughput Performance:")
            print(f"  Throughput: {throughput['throughput_rps']:.1f} requests/second")
            print(f"  Success Rate: {throughput['success_rate']:.1%}")
        
        if "concurrent_load" in detailed:
            concurrent = detailed["concurrent_load"]
            print(f"\n🔄 Concurrent Load Performance:")
            print(f"  Threads: {concurrent['num_threads']}")
            print(f"  Concurrent Throughput: {concurrent['concurrent_throughput_rps']:.1f} requests/second")
            print(f"  Avg Latency: {concurrent['avg_latency_ms']:.1f} ms")
            print(f"  P95 Latency: {concurrent['p95_latency_ms']:.1f} ms")
            print(f"  Success Rate: {concurrent['success_rate']:.1%}")
        
        if "memory_scaling" in detailed:
            memory = detailed["memory_scaling"]
            results_list = memory["memory_scaling_results"]
            if results_list:
                print(f"\n💾 Memory Scaling:")
                print(f"  Max Batch Size: {memory['max_successful_batch_size']}")
                for result in results_list:
                    print(f"    Batch {result['batch_size']}: {result['memory_per_item_mb']:.2f} MB/item")
        
        print(f"\n✅ Benchmark completed successfully!")
        print("="*60)


def main():
    """Main benchmark execution."""
    benchmark = PerformanceBenchmark()
    
    print("🧪 Starting Olfactory Transformer Performance Benchmark")
    print("This may take several minutes to complete...\n")
    
    results = benchmark.run_full_benchmark()
    benchmark.print_results(results)
    
    # Optionally save results to file
    results_file = Path("benchmark_results.json")
    try:
        import json
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n💾 Detailed results saved to {results_file}")
    except Exception as e:
        print(f"\n⚠️  Could not save results to file: {e}")


if __name__ == "__main__":
    main()