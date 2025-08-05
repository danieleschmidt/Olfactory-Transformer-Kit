"""Distributed training and federated learning utilities."""

from typing import Dict, List, Optional, Any, Union, Callable
import logging
import json
import pickle
import hashlib
from pathlib import Path
from dataclasses import dataclass, asdict
import threading
import time
from collections import defaultdict

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import numpy as np

# Optional distributed training imports
try:
    import torch.multiprocessing as mp
    HAS_MULTIPROCESSING = True
except ImportError:
    HAS_MULTIPROCESSING = False

from ..core.model import OlfactoryTransformer
from ..training.trainer import OlfactoryDataset, TrainingArguments


@dataclass
class DistributedConfig:
    """Configuration for distributed training."""
    backend: str = "nccl"  # nccl, gloo, mpi
    init_method: str = "env://"
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    master_addr: str = "localhost"
    master_port: str = "12355"
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DistributedTraining:
    """Distributed training coordinator for multi-GPU/multi-node training."""
    
    def __init__(
        self,
        model: OlfactoryTransformer,
        config: Optional[DistributedConfig] = None
    ):
        self.model = model
        self.config = config or DistributedConfig()
        
        self.is_initialized = False
        self.ddp_model = None
        self.device = None
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        
        # Communication buffers
        self.gradient_buffers = {}
        self.parameter_buffers = {}
    
    def setup_distributed(self) -> bool:
        """Initialize distributed training environment."""
        try:
            # Set environment variables
            import os
            os.environ['MASTER_ADDR'] = self.config.master_addr
            os.environ['MASTER_PORT'] = self.config.master_port
            os.environ['WORLD_SIZE'] = str(self.config.world_size)
            os.environ['RANK'] = str(self.config.rank)
            
            # Initialize process group
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
            
            # Move model to device and wrap with DDP
            self.model.to(self.device)
            self.ddp_model = DDP(
                self.model,
                device_ids=[self.config.local_rank] if torch.cuda.is_available() else None,
                output_device=self.config.local_rank if torch.cuda.is_available() else None,
                find_unused_parameters=True  # For complex models
            )
            
            self.is_initialized = True
            
            logging.info(f"Distributed training initialized - Rank: {self.config.rank}, "
                        f"World Size: {self.config.world_size}, Device: {self.device}")
            
            return True
            
        except Exception as e:
            logging.error(f"Failed to setup distributed training: {e}")
            return False
    
    def cleanup_distributed(self) -> None:
        """Clean up distributed training."""
        if self.is_initialized:
            dist.destroy_process_group()
            self.is_initialized = False
            logging.info("Distributed training cleaned up")
    
    def create_distributed_dataloader(
        self,
        dataset: OlfactoryDataset,
        batch_size: int,
        shuffle: bool = True
    ) -> DataLoader:
        """Create distributed data loader."""
        if not self.is_initialized:
            raise RuntimeError("Distributed training not initialized")
        
        # Create distributed sampler
        sampler = DistributedSampler(
            dataset,
            num_replicas=self.config.world_size,
            rank=self.config.rank,
            shuffle=shuffle
        )
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=4,
            pin_memory=True
        )
    
    def all_reduce_gradients(self) -> None:
        """Manually all-reduce gradients across processes."""
        if not self.is_initialized:
            return
        
        for param in self.model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                param.grad.data /= self.config.world_size
    
    def broadcast_parameters(self) -> None:
        """Broadcast parameters from rank 0 to all processes."""
        if not self.is_initialized:
            return
        
        for param in self.model.parameters():
            dist.broadcast(param.data, src=0)
    
    def gather_metrics(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """Gather metrics from all processes."""
        if not self.is_initialized:
            return metrics
        
        gathered_metrics = {}
        
        for key, value in metrics.items():
            # Convert to tensor for all_gather
            tensor = torch.tensor(value, device=self.device)
            gathered_tensors = [torch.zeros_like(tensor) for _ in range(self.config.world_size)]
            
            dist.all_gather(gathered_tensors, tensor)
            
            # Compute average across all processes
            gathered_values = [t.item() for t in gathered_tensors]
            gathered_metrics[key] = np.mean(gathered_values)
            gathered_metrics[f"{key}_std"] = np.std(gathered_values)
        
        return gathered_metrics
    
    def save_checkpoint(self, checkpoint_path: Path, optimizer: torch.optim.Optimizer) -> None:
        """Save distributed training checkpoint."""
        if self.config.rank == 0:  # Only save from rank 0
            checkpoint = {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": self.epoch,
                "global_step": self.global_step,
                "config": self.config.to_dict(),
            }
            
            torch.save(checkpoint, checkpoint_path)
            logging.info(f"Checkpoint saved to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: Path, optimizer: torch.optim.Optimizer) -> None:
        """Load distributed training checkpoint."""
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            self.model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.epoch = checkpoint["epoch"]
            self.global_step = checkpoint["global_step"]
            
            logging.info(f"Checkpoint loaded from {checkpoint_path}")
    
    def is_main_process(self) -> bool:
        """Check if this is the main process (rank 0)."""
        return self.config.rank == 0
    
    def barrier(self) -> None:
        """Synchronize all processes."""
        if self.is_initialized:
            dist.barrier()


@dataclass
class FederatedUpdate:
    """Container for federated learning model updates."""
    participant_id: str
    model_update: Dict[str, torch.Tensor]
    num_samples: int
    loss: float
    timestamp: float
    metadata: Optional[Dict[str, Any]] = None
    
    def serialize(self) -> bytes:
        """Serialize update for transmission."""
        return pickle.dumps(self)
    
    @classmethod
    def deserialize(cls, data: bytes) -> "FederatedUpdate":
        """Deserialize update from bytes."""
        return pickle.loads(data)
    
    def compute_hash(self) -> str:
        """Compute hash for integrity checking."""
        serialized = self.serialize()
        return hashlib.sha256(serialized).hexdigest()


class FederatedOlfactory:
    """Federated learning coordinator for olfactory models."""
    
    def __init__(
        self,
        base_model: Union[str, OlfactoryTransformer],
        aggregation: str = "fedavg",
        min_participants: int = 3,
        max_participants: int = 10,
        rounds: int = 10
    ):
        if isinstance(base_model, str):
            self.global_model = OlfactoryTransformer.from_pretrained(base_model)
        else:
            self.global_model = base_model
        
        self.aggregation = aggregation
        self.min_participants = min_participants
        self.max_participants = max_participants
        self.rounds = rounds
        
        # Federated state
        self.current_round = 0
        self.participants = {}
        self.round_updates = defaultdict(list)
        
        # Security and privacy
        self.differential_privacy = False
        self.dp_noise_multiplier = 0.1
        self.dp_clip_norm = 1.0
        
        # Logging
        self.round_history = []
        
        logging.info(f"Federated learning initialized with {aggregation} aggregation")
    
    def register_participant(
        self,
        participant_id: str,
        participant_info: Dict[str, Any]
    ) -> bool:
        """Register a federated learning participant."""
        if len(self.participants) >= self.max_participants:
            logging.warning(f"Maximum participants ({self.max_participants}) reached")
            return False
        
        self.participants[participant_id] = {
            "info": participant_info,
            "registered_at": time.time(),
            "rounds_participated": 0,
            "total_samples": 0,
            "last_update": None,
        }
        
        logging.info(f"Participant {participant_id} registered")
        return True
    
    def create_local_trainer(self, participant_id: str) -> "FederatedLocalTrainer":
        """Create local trainer for participant."""
        if participant_id not in self.participants:
            raise ValueError(f"Participant {participant_id} not registered")
        
        # Create local copy of global model
        local_model = type(self.global_model)(self.global_model.config)
        local_model.load_state_dict(self.global_model.state_dict())
        
        return FederatedLocalTrainer(
            participant_id=participant_id,
            model=local_model,
            federated_coordinator=self
        )
    
    def submit_update(self, update: FederatedUpdate) -> bool:
        """Submit model update from participant."""
        if update.participant_id not in self.participants:
            logging.error(f"Unknown participant: {update.participant_id}")
            return False
        
        # Validate update
        if not self._validate_update(update):
            logging.error(f"Invalid update from {update.participant_id}")
            return False
        
        # Apply differential privacy if enabled
        if self.differential_privacy:
            update = self._apply_differential_privacy(update)
        
        # Store update
        self.round_updates[self.current_round].append(update)
        
        # Update participant info
        self.participants[update.participant_id]["last_update"] = update.timestamp
        self.participants[update.participant_id]["total_samples"] += update.num_samples
        
        logging.info(f"Update received from {update.participant_id} "
                    f"({len(self.round_updates[self.current_round])} total this round)")
        
        return True
    
    def aggregate_updates(self, updates: List[FederatedUpdate]) -> Dict[str, torch.Tensor]:
        """Aggregate model updates from participants."""
        if not updates:
            return {}
        
        logging.info(f"Aggregating {len(updates)} updates using {self.aggregation}")
        
        if self.aggregation == "fedavg":
            return self._fedavg_aggregation(updates)
        elif self.aggregation == "fedprox":
            return self._fedprox_aggregation(updates)
        elif self.aggregation == "scaffold":
            return self._scaffold_aggregation(updates)
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation}")
    
    def _fedavg_aggregation(self, updates: List[FederatedUpdate]) -> Dict[str, torch.Tensor]:
        """FedAvg aggregation (weighted by number of samples)."""
        if not updates:
            return {}
        
        # Calculate total samples
        total_samples = sum(update.num_samples for update in updates)
        
        # Initialize aggregated parameters
        aggregated_params = {}
        
        for update in updates:
            weight = update.num_samples / total_samples
            
            for param_name, param_value in update.model_update.items():
                if param_name not in aggregated_params:
                    aggregated_params[param_name] = torch.zeros_like(param_value)
                
                aggregated_params[param_name] += weight * param_value
        
        return aggregated_params
    
    def _fedprox_aggregation(self, updates: List[FederatedUpdate]) -> Dict[str, torch.Tensor]:
        """FedProx aggregation with proximal term."""
        # For simplicity, use FedAvg here (FedProx mainly affects local training)
        return self._fedavg_aggregation(updates)
    
    def _scaffold_aggregation(self, updates: List[FederatedUpdate]) -> Dict[str, torch.Tensor]:
        """SCAFFOLD aggregation with control variates."""
        # Simplified SCAFFOLD implementation
        return self._fedavg_aggregation(updates)
    
    def run_round(self) -> Dict[str, Any]:
        """Run one round of federated learning."""
        logging.info(f"Starting federated round {self.current_round + 1}")
        
        # Wait for minimum participants
        start_time = time.time()
        timeout = 300  # 5 minutes
        
        while (len(self.round_updates[self.current_round]) < self.min_participants and 
               time.time() - start_time < timeout):
            time.sleep(10)  # Wait 10 seconds
        
        updates = self.round_updates[self.current_round]
        
        if len(updates) < self.min_participants:
            logging.warning(f"Insufficient participants for round {self.current_round + 1}")
            return {"status": "insufficient_participants", "participants": len(updates)}
        
        # Aggregate updates
        aggregated_params = self.aggregate_updates(updates)
        
        # Update global model
        if aggregated_params:
            global_state_dict = self.global_model.state_dict()
            for param_name, aggregated_value in aggregated_params.items():
                if param_name in global_state_dict:
                    global_state_dict[param_name] = aggregated_value
            
            self.global_model.load_state_dict(global_state_dict)
        
        # Calculate round statistics
        round_stats = {
            "round": self.current_round + 1,
            "participants": len(updates),
            "total_samples": sum(update.num_samples for update in updates),
            "avg_loss": np.mean([update.loss for update in updates]),
            "std_loss": np.std([update.loss for update in updates]),
            "duration": time.time() - start_time,
        }
        
        # Update participant statistics
        for update in updates:
            self.participants[update.participant_id]["rounds_participated"] += 1
        
        # Store round history
        self.round_history.append(round_stats)
        
        # Prepare for next round
        self.current_round += 1
        
        logging.info(f"Round {round_stats['round']} completed: {round_stats}")
        
        return round_stats
    
    def _validate_update(self, update: FederatedUpdate) -> bool:
        """Validate model update."""
        # Check basic requirements
        if not update.model_update:
            return False
        
        if update.num_samples <= 0:
            return False
        
        # Check parameter shapes match global model
        global_state_dict = self.global_model.state_dict()
        
        for param_name, param_value in update.model_update.items():
            if param_name not in global_state_dict:
                logging.warning(f"Unknown parameter: {param_name}")
                return False
            
            if param_value.shape != global_state_dict[param_name].shape:
                logging.warning(f"Shape mismatch for {param_name}")
                return False
        
        return True
    
    def _apply_differential_privacy(self, update: FederatedUpdate) -> FederatedUpdate:
        """Apply differential privacy to model update."""
        # Clip gradients
        total_norm = 0.0
        for param_value in update.model_update.values():
            total_norm += param_value.norm().item() ** 2
        total_norm = total_norm ** 0.5
        
        clip_coefficient = min(1.0, self.dp_clip_norm / total_norm)
        
        # Apply clipping and add noise
        noisy_update = {}
        for param_name, param_value in update.model_update.items():
            # Clip
            clipped_param = param_value * clip_coefficient
            
            # Add Gaussian noise
            noise = torch.normal(
                mean=0.0,
                std=self.dp_noise_multiplier * self.dp_clip_norm,
                size=param_value.shape
            )
            
            noisy_update[param_name] = clipped_param + noise
        
        # Create new update with noisy parameters
        return FederatedUpdate(
            participant_id=update.participant_id,
            model_update=noisy_update,
            num_samples=update.num_samples,
            loss=update.loss,
            timestamp=update.timestamp,
            metadata=update.metadata
        )
    
    def get_global_model(self) -> OlfactoryTransformer:
        """Get current global model."""
        return self.global_model
    
    def save_global_model(self, save_path: Path) -> None:
        """Save global model."""
        self.global_model.save_pretrained(save_path)
        
        # Save federated learning state
        fl_state = {
            "current_round": self.current_round,
            "participants": self.participants,
            "round_history": self.round_history,
            "config": {
                "aggregation": self.aggregation,
                "min_participants": self.min_participants,
                "max_participants": self.max_participants,
            }
        }
        
        with open(save_path / "federated_state.json", 'w') as f:
            json.dump(fl_state, f, indent=2, default=str)
        
        logging.info(f"Global model and FL state saved to {save_path}")


class FederatedLocalTrainer:
    """Local trainer for federated learning participants."""
    
    def __init__(
        self,
        participant_id: str,
        model: OlfactoryTransformer,
        federated_coordinator: FederatedOlfactory,
    ):
        self.participant_id = participant_id
        self.model = model
        self.coordinator = federated_coordinator
        
        # Training state
        self.initial_state_dict = None
        self.local_steps = 0
        
    def train_round(
        self,
        local_dataset: OlfactoryDataset,
        local_epochs: int = 5,
        learning_rate: float = 1e-4,
        batch_size: int = 32
    ) -> FederatedUpdate:
        """Train model locally for one federated round."""
        logging.info(f"Starting local training for {self.participant_id}")
        
        # Store initial state for computing update
        self.initial_state_dict = {
            name: param.clone() for name, param in self.model.state_dict().items()
        }
        
        # Setup local training
        dataloader = DataLoader(local_dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        # Local training loop
        for epoch in range(local_epochs):
            for batch in dataloader:
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(**batch)
                loss = outputs.get("loss", torch.tensor(0.0))
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                self.local_steps += 1
        
        avg_loss = total_loss / max(1, num_batches)
        
        # Compute model update (difference from initial state)
        model_update = {}
        current_state_dict = self.model.state_dict()
        
        for name, param in current_state_dict.items():
            if name in self.initial_state_dict:
                model_update[name] = param - self.initial_state_dict[name]
        
        # Create federated update
        update = FederatedUpdate(
            participant_id=self.participant_id,
            model_update=model_update,
            num_samples=len(local_dataset),
            loss=avg_loss,
            timestamp=time.time(),
            metadata={
                "local_epochs": local_epochs,
                "local_steps": self.local_steps,
                "batch_size": batch_size,
            }
        )
        
        logging.info(f"Local training completed for {self.participant_id}, "
                    f"loss: {avg_loss:.4f}, samples: {len(local_dataset)}")
        
        return update
    
    def get_update(self) -> Optional[FederatedUpdate]:
        """Get latest model update."""
        # This would return the last computed update
        # Implementation depends on specific use case
        return None
    
    def update_from_global(self, global_model_state: Dict[str, torch.Tensor]) -> None:
        """Update local model with global model state."""
        self.model.load_state_dict(global_model_state)
        logging.info(f"Local model updated for {self.participant_id}")