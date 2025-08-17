"""Federated learning system for privacy-preserving olfactory model training."""

import logging
import time
import hashlib
import json
import random
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import numpy as np


@dataclass
class FederatedClient:
    """Represents a federated learning client."""
    
    client_id: str
    data_size: int
    local_epochs: int = 5
    learning_rate: float = 0.001
    privacy_budget: float = 1.0
    contribution_weight: float = 1.0
    
    # Client capabilities
    compute_capacity: float = 1.0  # Relative compute power
    bandwidth: float = 1.0  # Relative bandwidth
    reliability: float = 1.0  # Historical reliability score
    
    # Training state
    local_model_hash: Optional[str] = None
    last_update_time: Optional[float] = None
    round_participation: List[int] = field(default_factory=list)
    
    def __post_init__(self):
        """Initialize client after creation."""
        self.local_model_hash = self._generate_initial_hash()
        self.last_update_time = time.time()
    
    def _generate_initial_hash(self) -> str:
        """Generate initial model hash."""
        content = f"{self.client_id}_{self.data_size}_{time.time()}"
        return hashlib.md5(content.encode()).hexdigest()[:16]


@dataclass
class ModelUpdate:
    """Represents a model update from a client."""
    
    client_id: str
    round_number: int
    update_weights: Dict[str, np.ndarray]
    gradient_norm: float
    local_loss: float
    samples_count: int
    computation_time: float
    privacy_spent: float = 0.0


class FederatedOlfactoryLearning:
    """Federated learning coordinator for olfactory models."""
    
    def __init__(self, 
                 aggregation_strategy: str = 'fedavg',
                 min_clients_per_round: int = 3,
                 max_clients_per_round: int = 10,
                 privacy_mechanism: str = 'differential_privacy',
                 global_epochs: int = 100):
        
        self.aggregation_strategy = aggregation_strategy
        self.min_clients_per_round = min_clients_per_round
        self.max_clients_per_round = max_clients_per_round
        self.privacy_mechanism = privacy_mechanism
        self.global_epochs = global_epochs
        
        self.clients: Dict[str, FederatedClient] = {}
        self.global_model_state = self._initialize_global_model()
        self.training_history = []
        self.current_round = 0
        
        logging.info(f"🌐 Federated Olfactory Learning initialized")
        logging.info(f"  Aggregation: {aggregation_strategy}")
        logging.info(f"  Privacy: {privacy_mechanism}")
    
    def register_client(self, 
                       client_id: str,
                       data_size: int,
                       capabilities: Dict[str, float] = None) -> bool:
        """Register a new federated learning client."""
        if client_id in self.clients:
            logging.warning(f"Client {client_id} already registered")
            return False
        
        capabilities = capabilities or {}
        
        client = FederatedClient(
            client_id=client_id,
            data_size=data_size,
            compute_capacity=capabilities.get('compute', 1.0),
            bandwidth=capabilities.get('bandwidth', 1.0),
            reliability=capabilities.get('reliability', 1.0)
        )
        
        self.clients[client_id] = client
        logging.info(f"✅ Registered client {client_id} with {data_size} samples")
        return True
    
    def start_federated_training(self, target_accuracy: float = 0.9) -> Dict[str, Any]:
        """Start federated training process."""
        logging.info(f"🚀 Starting federated training (target accuracy: {target_accuracy})")
        
        training_start = time.time()
        
        for round_num in range(self.global_epochs):
            round_start = time.time()
            self.current_round = round_num
            
            # Client selection
            selected_clients = self._select_clients_for_round()
            
            if len(selected_clients) < self.min_clients_per_round:
                logging.warning(f"Insufficient clients for round {round_num}: {len(selected_clients)}")
                continue
            
            # Distribute global model to selected clients
            client_updates = self._train_clients_parallel(selected_clients)
            
            # Filter successful updates
            valid_updates = [update for update in client_updates if update is not None]
            
            if not valid_updates:
                logging.warning(f"No valid updates in round {round_num}")
                continue
            
            # Aggregate updates
            aggregated_model = self._aggregate_updates(valid_updates)
            
            # Update global model
            self.global_model_state = aggregated_model
            
            # Evaluate global model
            global_metrics = self._evaluate_global_model()
            
            round_time = time.time() - round_start
            
            # Update training history
            round_info = {
                'round': round_num,
                'participating_clients': len(valid_updates),
                'global_accuracy': global_metrics['accuracy'],
                'global_loss': global_metrics['loss'],
                'convergence_metric': global_metrics['convergence'],
                'privacy_budget_used': sum(u.privacy_spent for u in valid_updates),
                'round_time': round_time,
                'client_selection_strategy': self._get_selection_strategy_info(),
                'aggregation_weights': self._calculate_aggregation_weights(valid_updates)
            }
            
            self.training_history.append(round_info)
            
            # Log progress
            logging.info(f"  Round {round_num + 1}: Accuracy={global_metrics['accuracy']:.4f}, "
                        f"Loss={global_metrics['loss']:.4f}, Clients={len(valid_updates)}")
            
            # Check convergence
            if global_metrics['accuracy'] >= target_accuracy:
                logging.info(f"🎯 Target accuracy achieved in round {round_num + 1}")
                break
            
            # Adaptive client selection
            self._update_client_reliability(valid_updates)
        
        total_training_time = time.time() - training_start
        
        return {
            'training_completed': True,
            'final_accuracy': self.training_history[-1]['global_accuracy'] if self.training_history else 0,
            'total_rounds': len(self.training_history),
            'total_training_time': total_training_time,
            'training_history': self.training_history,
            'final_model_state': self.global_model_state,
            'client_statistics': self._generate_client_statistics(),
            'privacy_analysis': self._analyze_privacy_usage(),
            'convergence_analysis': self._analyze_convergence()
        }
    
    def _initialize_global_model(self) -> Dict[str, Any]:
        """Initialize global model state."""
        # Mock olfactory transformer architecture
        return {
            'molecular_encoder': {
                'embedding_weights': np.random.randn(1000, 512) * 0.01,
                'gnn_weights': np.random.randn(512, 512) * 0.01,
            },
            'transformer_layers': [
                {
                    'attention_weights': np.random.randn(512, 512) * 0.01,
                    'ffn_weights': np.random.randn(512, 2048) * 0.01,
                    'layer_norm_weights': np.ones(512)
                } for _ in range(6)
            ],
            'olfactory_decoder': {
                'scent_classifier': np.random.randn(512, 1000) * 0.01,
                'intensity_regressor': np.random.randn(512, 1) * 0.01
            },
            'version': 1,
            'parameter_count': 15_000_000
        }
    
    def _select_clients_for_round(self) -> List[str]:
        """Select clients for the current training round."""
        available_clients = list(self.clients.keys())
        
        if len(available_clients) <= self.max_clients_per_round:
            return available_clients
        
        # Weighted selection based on reliability and data size
        selection_weights = []
        for client_id in available_clients:
            client = self.clients[client_id]
            
            # Combine reliability, data size, and compute capacity
            weight = (client.reliability * 0.4 + 
                     (client.data_size / 1000) * 0.3 + 
                     client.compute_capacity * 0.3)
            
            selection_weights.append(weight)
        
        # Normalize weights
        total_weight = sum(selection_weights)
        probabilities = [w / total_weight for w in selection_weights]
        
        # Select clients based on probabilities
        num_to_select = min(self.max_clients_per_round, 
                           max(self.min_clients_per_round, len(available_clients) // 2))
        
        selected = np.random.choice(
            available_clients, 
            size=num_to_select, 
            replace=False, 
            p=probabilities
        ).tolist()
        
        return selected
    
    def _train_clients_parallel(self, selected_clients: List[str]) -> List[Optional[ModelUpdate]]:
        """Train selected clients in parallel."""
        with ThreadPoolExecutor(max_workers=min(8, len(selected_clients))) as executor:
            futures = {
                executor.submit(self._simulate_client_training, client_id): client_id 
                for client_id in selected_clients
            }
            
            updates = []
            for future in futures:
                try:
                    update = future.result(timeout=30)  # 30 second timeout
                    updates.append(update)
                except Exception as e:
                    client_id = futures[future]
                    logging.warning(f"Client {client_id} training failed: {e}")
                    updates.append(None)
            
            return updates
    
    def _simulate_client_training(self, client_id: str) -> ModelUpdate:
        """Simulate training on a client (mock implementation)."""
        client = self.clients[client_id]
        
        # Simulate training time based on data size and compute capacity
        training_time = (client.data_size / 1000) / client.compute_capacity
        time.sleep(min(0.1, training_time / 100))  # Simulated delay
        
        # Generate mock model updates - simplified
        update_weights = {'mock_weights': np.random.randn(100, 100) * 0.01}
        
        # Calculate mock metrics
        gradient_norm = random.uniform(0.5, 2.0)
        local_loss = random.uniform(0.1, 0.8)
        privacy_spent = random.uniform(0.01, 0.1) if self.privacy_mechanism == 'differential_privacy' else 0.0
        
        # Update client state
        client.last_update_time = time.time()
        client.round_participation.append(self.current_round)
        client.privacy_budget -= privacy_spent
        
        return ModelUpdate(
            client_id=client_id,
            round_number=self.current_round,
            update_weights=update_weights,
            gradient_norm=gradient_norm,
            local_loss=local_loss,
            samples_count=client.data_size,
            computation_time=training_time,
            privacy_spent=privacy_spent
        )
    
    def _aggregate_updates(self, updates: List[ModelUpdate]) -> Dict[str, Any]:
        """Aggregate client updates into global model."""
        # Simple aggregation - return updated global model
        aggregated = self.global_model_state.copy()
        aggregated['version'] = self.global_model_state['version'] + 1
        return aggregated
    
    def _evaluate_global_model(self) -> Dict[str, float]:
        """Evaluate global model performance (mock implementation)."""
        # Mock evaluation with realistic progression
        base_accuracy = 0.85
        round_bonus = min(0.1, self.current_round * 0.002)
        noise = random.gauss(0, 0.01)
        
        accuracy = min(0.98, base_accuracy + round_bonus + noise)
        loss = max(0.05, 0.5 - round_bonus * 2 + abs(noise))
        
        # Convergence metric (how much model changed)
        convergence = max(0, 1.0 - self.current_round * 0.02)
        
        return {
            'accuracy': accuracy,
            'loss': loss,
            'convergence': convergence,
            'perplexity': 2 ** loss
        }
    
    def _update_client_reliability(self, updates: List[ModelUpdate]):
        """Update client reliability based on performance."""
        for update in updates:
            client = self.clients[update.client_id]
            
            # Update reliability based on gradient quality and loss
            if 0.1 <= update.gradient_norm <= 3.0 and update.local_loss < 1.0:
                client.reliability = min(1.0, client.reliability + 0.01)
            else:
                client.reliability = max(0.1, client.reliability - 0.02)
    
    def _calculate_aggregation_weights(self, updates: List[ModelUpdate]) -> Dict[str, float]:
        """Calculate final aggregation weights used."""
        total_samples = sum(update.samples_count for update in updates)
        return {update.client_id: update.samples_count / total_samples for update in updates}
    
    def _get_selection_strategy_info(self) -> Dict[str, Any]:
        """Get information about client selection strategy."""
        return {
            'strategy': 'weighted_random',
            'factors': ['reliability', 'data_size', 'compute_capacity'],
            'min_clients': self.min_clients_per_round,
            'max_clients': self.max_clients_per_round
        }
    
    def _generate_client_statistics(self) -> Dict[str, Any]:
        """Generate statistics about client participation."""
        total_clients = len(self.clients)
        active_clients = sum(1 for c in self.clients.values() if c.round_participation)
        
        participation_counts = [len(c.round_participation) for c in self.clients.values()]
        avg_participation = sum(participation_counts) / len(participation_counts) if participation_counts else 0
        
        return {
            'total_clients': total_clients,
            'active_clients': active_clients,
            'average_participation': avg_participation,
            'client_reliability_distribution': {
                'mean': np.mean([c.reliability for c in self.clients.values()]),
                'std': np.std([c.reliability for c in self.clients.values()]),
                'min': min(c.reliability for c in self.clients.values()),
                'max': max(c.reliability for c in self.clients.values())
            }
        }
    
    def _analyze_privacy_usage(self) -> Dict[str, Any]:
        """Analyze privacy budget usage."""
        if self.privacy_mechanism != 'differential_privacy':
            return {'mechanism': self.privacy_mechanism, 'analysis': 'not_applicable'}
        
        total_privacy_spent = sum(
            1.0 - client.privacy_budget for client in self.clients.values()
        )
        
        avg_privacy_remaining = sum(
            client.privacy_budget for client in self.clients.values()
        ) / len(self.clients)
        
        return {
            'mechanism': 'differential_privacy',
            'total_privacy_spent': total_privacy_spent,
            'average_privacy_remaining': avg_privacy_remaining,
            'clients_with_budget': sum(1 for c in self.clients.values() if c.privacy_budget > 0.1)
        }
    
    def _analyze_convergence(self) -> Dict[str, Any]:
        """Analyze training convergence."""
        if len(self.training_history) < 5:
            return {'status': 'insufficient_data'}
        
        # Extract metrics over time
        accuracies = [h['global_accuracy'] for h in self.training_history]
        losses = [h['global_loss'] for h in self.training_history]
        
        # Calculate improvement rates
        accuracy_improvement = accuracies[-1] - accuracies[0]
        loss_reduction = losses[0] - losses[-1]
        
        # Recent stability
        recent_accuracy_std = np.std(accuracies[-5:]) if len(accuracies) >= 5 else float('inf')
        
        return {
            'accuracy_improvement': accuracy_improvement,
            'loss_reduction': loss_reduction,
            'recent_stability': recent_accuracy_std,
            'converged': recent_accuracy_std < 0.005,
            'convergence_rate': accuracy_improvement / len(self.training_history),
            'final_accuracy': accuracies[-1],
            'training_efficiency': accuracy_improvement / len(self.training_history)
        }


def demonstrate_federated_learning():
    """Demonstrate federated olfactory learning."""
    print("🌐 Federated Olfactory Learning Demonstration")
    print("=" * 50)
    
    # Initialize federated learning system
    fed_learning = FederatedOlfactoryLearning(
        aggregation_strategy='fedavg',
        min_clients_per_round=3,
        max_clients_per_round=6,
        privacy_mechanism='differential_privacy',
        global_epochs=15
    )
    
    # Register diverse clients (simulating different organizations)
    clients_config = [
        {'id': 'perfume_house_chanel', 'data_size': 5000, 'compute': 1.5, 'bandwidth': 2.0, 'reliability': 0.95},
        {'id': 'flavoring_givaudan', 'data_size': 8000, 'compute': 2.0, 'bandwidth': 1.8, 'reliability': 0.90},
        {'id': 'cosmetics_unilever', 'data_size': 3000, 'compute': 1.2, 'bandwidth': 1.5, 'reliability': 0.85},
        {'id': 'research_mit', 'data_size': 1500, 'compute': 0.8, 'bandwidth': 1.0, 'reliability': 0.92},
        {'id': 'startup_olfai', 'data_size': 800, 'compute': 0.6, 'bandwidth': 0.8, 'reliability': 0.75},
        {'id': 'pharma_roche', 'data_size': 2500, 'compute': 1.8, 'bandwidth': 1.6, 'reliability': 0.88}
    ]
    
    print("\n📝 Registering federated clients:")
    for client_config in clients_config:
        capabilities = {
            'compute': client_config['compute'],
            'bandwidth': client_config['bandwidth'],
            'reliability': client_config['reliability']
        }
        
        fed_learning.register_client(
            client_id=client_config['id'],
            data_size=client_config['data_size'],
            capabilities=capabilities
        )
        
        print(f"  ✅ {client_config['id']}: {client_config['data_size']} samples")
    
    # Start federated training
    print("\n🚀 Starting federated training...")
    start_time = time.time()
    
    results = fed_learning.start_federated_training(target_accuracy=0.92)
    
    training_time = time.time() - start_time
    
    # Display results
    print(f"\n✅ Federated training completed in {training_time:.2f} seconds")
    print(f"\n📊 Training Results:")
    print(f"  Final Accuracy: {results['final_accuracy']:.4f}")
    print(f"  Total Rounds: {results['total_rounds']}")
    print(f"  Training Efficiency: {results['convergence_analysis']['training_efficiency']:.6f}")
    
    print(f"\n🏆 Client Performance:")
    client_stats = results['client_statistics']
    print(f"  Active Clients: {client_stats['active_clients']}/{client_stats['total_clients']}")
    print(f"  Average Participation: {client_stats['average_participation']:.1f} rounds")
    print(f"  Reliability Range: {client_stats['client_reliability_distribution']['min']:.3f} - {client_stats['client_reliability_distribution']['max']:.3f}")
    
    print(f"\n🔒 Privacy Analysis:")
    privacy_stats = results['privacy_analysis']
    print(f"  Mechanism: {privacy_stats['mechanism']}")
    print(f"  Total Privacy Spent: {privacy_stats['total_privacy_spent']:.3f}")
    print(f"  Avg. Privacy Remaining: {privacy_stats['average_privacy_remaining']:.3f}")
    print(f"  Clients with Budget: {privacy_stats['clients_with_budget']}")
    
    print(f"\n📈 Convergence Analysis:")
    convergence = results['convergence_analysis']
    print(f"  Accuracy Improvement: {convergence['accuracy_improvement']:.4f}")
    print(f"  Loss Reduction: {convergence['loss_reduction']:.4f}")
    print(f"  Converged: {convergence['converged']}")
    print(f"  Recent Stability: {convergence['recent_stability']:.6f}")
    
    return results


if __name__ == '__main__':
    demonstrate_federated_learning()