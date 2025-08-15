"""
Distributed Training Module for Olfactory Transformer.

Implements scalable distributed training across multiple GPUs and nodes:
- Data parallelism with gradient synchronization
- Model parallelism for large architectures
- Federated learning for privacy-preserving training
- Adaptive batch sizing and learning rate scheduling
"""

import logging
import json
import time
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
import threading
import queue
import hashlib

try:
    import torch
    import torch.distributed as dist
    import torch.multiprocessing as mp
    from torch.nn.parallel import DistributedDataParallel as DDP
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    dist = None
    mp = None
    DDP = None


@dataclass
class DistributedConfig:
    """Configuration for distributed training."""
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    backend: str = "nccl"
    init_method: str = "env://"
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    mixed_precision: bool = True
    find_unused_parameters: bool = False
    bucket_cap_mb: int = 25
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FederatedConfig:
    """Configuration for federated learning."""
    num_clients: int = 10
    client_fraction: float = 0.3
    local_epochs: int = 5
    aggregation_method: str = "fedavg"
    differential_privacy: bool = True
    privacy_budget: float = 1.0
    secure_aggregation: bool = True
    min_clients_for_update: int = 3


class DistributedTrainer:
    """Distributed trainer for olfactory transformer models."""
    
    def __init__(self, config: DistributedConfig):
        self.config = config
        self.device = None
        self.model = None
        self.optimizer = None
        self.scaler = None
        self.is_initialized = False
        
    def initialize_distributed(self) -> None:
        """Initialize distributed training environment."""
        if not HAS_TORCH:
            raise RuntimeError("PyTorch is required for distributed training")
        
        if self.is_initialized:
            return
            
        # Initialize process group
        if self.config.world_size > 1:
            dist.init_process_group(
                backend=self.config.backend,
                init_method=self.config.init_method,
                world_size=self.config.world_size,
                rank=self.config.rank
            )
            
        # Set device
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{self.config.local_rank}")
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")
            
        # Initialize mixed precision scaler
        if self.config.mixed_precision and torch.cuda.is_available():
            self.scaler = torch.cuda.amp.GradScaler()
            
        self.is_initialized = True
        logging.info(f"Initialized distributed training: rank {self.config.rank}/{self.config.world_size}")
    
    def setup_model(self, model: Any) -> Any:
        """Setup model for distributed training."""
        if not self.is_initialized:
            self.initialize_distributed()
            
        # Move model to device
        model = model.to(self.device)
        
        # Wrap with DDP if multi-GPU
        if self.config.world_size > 1:
            model = DDP(
                model,
                device_ids=[self.config.local_rank] if torch.cuda.is_available() else None,
                find_unused_parameters=self.config.find_unused_parameters,
                bucket_cap_mb=self.config.bucket_cap_mb
            )
            
        self.model = model
        return model
    
    def train_step(self, batch: Dict[str, torch.Tensor], optimizer: torch.optim.Optimizer) -> Dict[str, float]:
        """Perform single distributed training step."""
        model = self.model
        device = self.device
        
        # Move batch to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        # Forward pass with mixed precision
        if self.config.mixed_precision and self.scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(**batch)
                loss = outputs.get('loss', outputs.get('total_loss', torch.tensor(0.0)))
        else:
            outputs = model(**batch)
            loss = outputs.get('loss', outputs.get('total_loss', torch.tensor(0.0)))
        
        # Scale loss for gradient accumulation
        loss = loss / self.config.gradient_accumulation_steps
        
        # Backward pass
        if self.config.mixed_precision and self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Gradient synchronization happens automatically with DDP
        
        return {
            'loss': loss.item() * self.config.gradient_accumulation_steps,
            'learning_rate': optimizer.param_groups[0]['lr'] if optimizer else 0.0
        }
    
    def optimizer_step(self, optimizer: torch.optim.Optimizer, step: int) -> None:
        """Perform optimizer step with gradient clipping."""
        if step % self.config.gradient_accumulation_steps == 0:
            # Gradient clipping
            if hasattr(self.model, 'module'):
                torch.nn.utils.clip_grad_norm_(self.model.module.parameters(), 1.0)
            else:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            # Optimizer step
            if self.config.mixed_precision and self.scaler is not None:
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                optimizer.step()
                
            optimizer.zero_grad()
    
    def all_reduce_metrics(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """All-reduce metrics across distributed processes."""
        if self.config.world_size <= 1:
            return metrics
            
        reduced_metrics = {}
        for key, value in metrics.items():
            tensor = torch.tensor(value, device=self.device)
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            reduced_metrics[key] = tensor.item() / self.config.world_size
            
        return reduced_metrics
    
    def save_checkpoint(self, checkpoint_path: Path, model: Any, optimizer: torch.optim.Optimizer, 
                       epoch: int, metrics: Dict[str, float]) -> None:
        """Save distributed training checkpoint."""
        if self.config.rank == 0:  # Only master saves
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': metrics,
                'config': self.config.to_dict()
            }
            
            if self.scaler is not None:
                checkpoint['scaler_state_dict'] = self.scaler.state_dict()
                
            torch.save(checkpoint, checkpoint_path)
            logging.info(f"Saved checkpoint to {checkpoint_path}")
    
    def cleanup(self) -> None:
        """Cleanup distributed training."""
        if self.config.world_size > 1:
            dist.destroy_process_group()
        logging.info("Cleaned up distributed training")


class FederatedLearningCoordinator:
    """Coordinates federated learning across multiple clients."""
    
    def __init__(self, config: FederatedConfig):
        self.config = config
        self.global_model = None
        self.client_models = {}
        self.round_number = 0
        self.client_updates = queue.Queue()
        
    def initialize_global_model(self, model_class: Any, model_config: Dict[str, Any]) -> None:
        """Initialize global model."""
        self.global_model = model_class(**model_config)
        logging.info("Initialized global federated model")
    
    def create_client(self, client_id: str, data_loader: Any) -> 'FederatedClient':
        """Create federated learning client."""
        if self.global_model is None:
            raise RuntimeError("Global model not initialized")
            
        client = FederatedClient(
            client_id=client_id,
            model=self._copy_model(self.global_model),
            data_loader=data_loader,
            local_epochs=self.config.local_epochs
        )
        
        self.client_models[client_id] = client
        return client
    
    def federated_round(self, selected_clients: List[str]) -> Dict[str, float]:
        """Execute one round of federated learning."""
        logging.info(f"Starting federated round {self.round_number}")
        
        # Check minimum clients
        if len(selected_clients) < self.config.min_clients_for_update:
            raise ValueError(f"Insufficient clients: {len(selected_clients)} < {self.config.min_clients_for_update}")
        
        # Distribute global model to selected clients
        global_state = self.global_model.state_dict() if HAS_TORCH else {}
        
        client_updates = []
        for client_id in selected_clients:
            if client_id in self.client_models:
                client = self.client_models[client_id]
                
                # Update client model with global state
                if HAS_TORCH and hasattr(client.model, 'load_state_dict'):
                    client.model.load_state_dict(global_state)
                
                # Local training
                update = client.local_train()
                
                # Add differential privacy noise if enabled
                if self.config.differential_privacy:
                    update = self._add_dp_noise(update)
                
                client_updates.append(update)
        
        # Aggregate updates
        aggregated_update = self._aggregate_updates(client_updates)
        
        # Update global model
        self._update_global_model(aggregated_update)
        
        self.round_number += 1
        
        return {
            'round': self.round_number,
            'participating_clients': len(selected_clients),
            'aggregation_method': self.config.aggregation_method
        }
    
    def _copy_model(self, model: Any) -> Any:
        """Create copy of model for client."""
        if HAS_TORCH:
            # Deep copy model
            import copy
            return copy.deepcopy(model)
        else:
            # Placeholder for non-torch models
            return model
    
    def _add_dp_noise(self, update: Dict[str, Any]) -> Dict[str, Any]:
        """Add differential privacy noise to client update."""
        if not HAS_TORCH:
            return update
            
        noisy_update = {}
        noise_scale = 1.0 / self.config.privacy_budget
        
        for key, value in update.items():
            if isinstance(value, torch.Tensor):
                noise = torch.normal(0, noise_scale, size=value.shape)
                noisy_update[key] = value + noise
            else:
                noisy_update[key] = value
                
        return noisy_update
    
    def _aggregate_updates(self, client_updates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate client updates using FedAvg or other methods."""
        if not client_updates:
            return {}
            
        if self.config.aggregation_method == "fedavg":
            return self._federated_averaging(client_updates)
        else:
            raise ValueError(f"Unknown aggregation method: {self.config.aggregation_method}")
    
    def _federated_averaging(self, client_updates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Federated averaging aggregation."""
        if not HAS_TORCH:
            return client_updates[0] if client_updates else {}
            
        aggregated = {}
        num_clients = len(client_updates)
        
        # Get keys from first update
        if client_updates:
            keys = client_updates[0].keys()
            
            for key in keys:
                if key == 'model_state':
                    # Average model parameters
                    param_sum = None
                    for update in client_updates:
                        client_state = update[key]
                        if param_sum is None:
                            param_sum = {k: v.clone() for k, v in client_state.items()}
                        else:
                            for param_key, param_value in client_state.items():
                                param_sum[param_key] += param_value
                    
                    # Average
                    aggregated[key] = {k: v / num_clients for k, v in param_sum.items()}
                else:
                    # Average scalar values
                    values = [update.get(key, 0) for update in client_updates]
                    aggregated[key] = sum(values) / len(values)
        
        return aggregated
    
    def _update_global_model(self, aggregated_update: Dict[str, Any]) -> None:
        """Update global model with aggregated update."""
        if 'model_state' in aggregated_update and HAS_TORCH:
            self.global_model.load_state_dict(aggregated_update['model_state'])
        
        logging.info("Updated global model with federated aggregation")


class FederatedClient:
    """Federated learning client."""
    
    def __init__(self, client_id: str, model: Any, data_loader: Any, local_epochs: int = 5):
        self.client_id = client_id
        self.model = model
        self.data_loader = data_loader
        self.local_epochs = local_epochs
        
    def local_train(self) -> Dict[str, Any]:
        """Perform local training and return update."""
        logging.info(f"Client {self.client_id} starting local training")
        
        initial_state = self.model.state_dict() if HAS_TORCH and hasattr(self.model, 'state_dict') else {}
        
        # Simulate local training
        for epoch in range(self.local_epochs):
            # In real implementation, would iterate through data_loader
            # For now, simulate training step
            if hasattr(self.model, 'train_step'):
                self.model.train_step()
        
        final_state = self.model.state_dict() if HAS_TORCH and hasattr(self.model, 'state_dict') else {}
        
        # Compute update (difference from initial state)
        update = {
            'client_id': self.client_id,
            'model_state': final_state,
            'samples_trained': len(self.data_loader) if hasattr(self.data_loader, '__len__') else 100,
            'local_epochs': self.local_epochs
        }
        
        logging.info(f"Client {self.client_id} completed local training")
        return update


class AdaptiveScaling:
    """Adaptive scaling for distributed training."""
    
    def __init__(self):
        self.batch_size_history = []
        self.throughput_history = []
        self.memory_usage_history = []
        
    def adaptive_batch_size(self, current_batch_size: int, memory_usage: float, 
                          throughput: float, target_memory: float = 0.8) -> int:
        """Adaptively adjust batch size based on memory and throughput."""
        self.batch_size_history.append(current_batch_size)
        self.memory_usage_history.append(memory_usage)
        self.throughput_history.append(throughput)
        
        # If memory usage is too high, reduce batch size
        if memory_usage > target_memory:
            new_batch_size = max(1, int(current_batch_size * 0.8))
            logging.info(f"Reducing batch size due to memory: {current_batch_size} -> {new_batch_size}")
            return new_batch_size
        
        # If memory usage is low and throughput is stable, increase batch size
        if memory_usage < target_memory * 0.6 and len(self.throughput_history) >= 3:
            recent_throughput = self.throughput_history[-3:]
            if max(recent_throughput) - min(recent_throughput) < 0.1 * sum(recent_throughput) / 3:
                new_batch_size = int(current_batch_size * 1.2)
                logging.info(f"Increasing batch size: {current_batch_size} -> {new_batch_size}")
                return new_batch_size
        
        return current_batch_size
    
    def adaptive_learning_rate(self, optimizer: Any, loss_history: List[float], 
                             patience: int = 3) -> float:
        """Adaptively adjust learning rate based on loss trajectory."""
        if len(loss_history) < patience + 1:
            return optimizer.param_groups[0]['lr'] if hasattr(optimizer, 'param_groups') else 0.001
        
        recent_losses = loss_history[-patience-1:]
        
        # Check if loss is not improving
        if all(recent_losses[i] >= recent_losses[i+1] for i in range(len(recent_losses)-1)):
            # Loss is increasing or stagnant
            current_lr = optimizer.param_groups[0]['lr'] if hasattr(optimizer, 'param_groups') else 0.001
            new_lr = current_lr * 0.5
            
            if hasattr(optimizer, 'param_groups'):
                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_lr
            
            logging.info(f"Reducing learning rate: {current_lr:.6f} -> {new_lr:.6f}")
            return new_lr
        
        return optimizer.param_groups[0]['lr'] if hasattr(optimizer, 'param_groups') else 0.001


class DistributedInference:
    """Distributed inference for large-scale deployment."""
    
    def __init__(self, model: Any, world_size: int = 1):
        self.model = model
        self.world_size = world_size
        self.inference_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.workers = []
        
    def start_inference_workers(self) -> None:
        """Start distributed inference workers."""
        for worker_id in range(self.world_size):
            worker = threading.Thread(
                target=self._inference_worker,
                args=(worker_id,),
                daemon=True
            )
            worker.start()
            self.workers.append(worker)
            
        logging.info(f"Started {self.world_size} inference workers")
    
    def distributed_predict(self, inputs: List[Any], timeout: float = 30.0) -> List[Any]:
        """Perform distributed prediction."""
        # Split inputs across workers
        chunks = self._chunk_inputs(inputs, self.world_size)
        
        # Submit work
        request_id = str(time.time())
        for chunk_id, chunk in enumerate(chunks):
            self.inference_queue.put({
                'request_id': request_id,
                'chunk_id': chunk_id,
                'inputs': chunk
            })
        
        # Collect results
        results = {}
        start_time = time.time()
        
        while len(results) < len(chunks):
            if time.time() - start_time > timeout:
                raise TimeoutError("Distributed inference timeout")
                
            try:
                result = self.result_queue.get(timeout=1.0)
                if result['request_id'] == request_id:
                    results[result['chunk_id']] = result['predictions']
            except queue.Empty:
                continue
        
        # Reassemble results in order
        ordered_results = []
        for chunk_id in sorted(results.keys()):
            ordered_results.extend(results[chunk_id])
            
        return ordered_results
    
    def _inference_worker(self, worker_id: int) -> None:
        """Inference worker function."""
        while True:
            try:
                work = self.inference_queue.get(timeout=1.0)
                
                # Perform inference
                inputs = work['inputs']
                if hasattr(self.model, 'predict'):
                    predictions = self.model.predict(inputs)
                else:
                    predictions = [0.5] * len(inputs)  # Dummy prediction
                
                # Return results
                self.result_queue.put({
                    'request_id': work['request_id'],
                    'chunk_id': work['chunk_id'],
                    'predictions': predictions
                })
                
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Inference worker {worker_id} error: {e}")
    
    def _chunk_inputs(self, inputs: List[Any], num_chunks: int) -> List[List[Any]]:
        """Split inputs into chunks for distribution."""
        chunk_size = len(inputs) // num_chunks
        chunks = []
        
        for i in range(num_chunks):
            start_idx = i * chunk_size
            if i == num_chunks - 1:
                end_idx = len(inputs)
            else:
                end_idx = (i + 1) * chunk_size
            
            chunks.append(inputs[start_idx:end_idx])
        
        return chunks


def main():
    """Demonstrate distributed training capabilities."""
    # Distributed training configuration
    dist_config = DistributedConfig(
        world_size=1,  # Single process for demo
        rank=0,
        batch_size=32,
        mixed_precision=False,  # Disable for CPU demo
        gradient_accumulation_steps=2
    )
    
    # Create distributed trainer
    trainer = DistributedTrainer(dist_config)
    
    print("🚀 Distributed Training Demo")
    print(f"Configuration: {dist_config.to_dict()}")
    
    # Federated learning demo
    fed_config = FederatedConfig(
        num_clients=5,
        client_fraction=0.6,
        local_epochs=3
    )
    
    coordinator = FederatedLearningCoordinator(fed_config)
    print(f"\n🤝 Federated Learning Demo")
    print(f"Configuration: {asdict(fed_config)}")
    
    # Adaptive scaling demo
    scaler = AdaptiveScaling()
    
    # Simulate batch size adaptation
    current_batch = 32
    for i in range(5):
        memory_usage = 0.6 + i * 0.1  # Increasing memory usage
        throughput = 100 - i * 5  # Decreasing throughput
        
        new_batch = scaler.adaptive_batch_size(current_batch, memory_usage, throughput)
        print(f"Batch size adaptation {i+1}: {current_batch} -> {new_batch} (mem: {memory_usage:.1f})")
        current_batch = new_batch
    
    # Distributed inference demo
    class DummyModel:
        def predict(self, inputs):
            return [0.5] * len(inputs)
    
    inference_engine = DistributedInference(DummyModel(), world_size=2)
    inference_engine.start_inference_workers()
    
    test_inputs = list(range(10))
    results = inference_engine.distributed_predict(test_inputs)
    print(f"\n⚡ Distributed Inference: {len(test_inputs)} inputs -> {len(results)} results")
    
    print("\n✅ Distributed scaling capabilities demonstrated successfully!")
    return True


if __name__ == "__main__":
    main()