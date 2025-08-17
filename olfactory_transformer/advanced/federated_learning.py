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
    
    def get_weighted_contribution(self, weighting_strategy: str = 'data_size') -> float:
        """Calculate weighted contribution based on strategy."""
        if weighting_strategy == 'data_size':
            return self.samples_count
        elif weighting_strategy == 'loss_based':
            return 1.0 / (self.local_loss + 1e-8)
        elif weighting_strategy == 'gradient_norm':
            return min(10.0, self.gradient_norm)
        else:
            return 1.0


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
                    updates.append(update)\n                except Exception as e:\n                    client_id = futures[future]\n                    logging.warning(f\"Client {client_id} training failed: {e}\")\n                    updates.append(None)
            
            return updates
    
    def _simulate_client_training(self, client_id: str) -> ModelUpdate:
        """Simulate training on a client (mock implementation)."""
        client = self.clients[client_id]\n        \n        # Simulate training time based on data size and compute capacity\n        training_time = (client.data_size / 1000) / client.compute_capacity\n        time.sleep(min(0.1, training_time / 100))  # Simulated delay\n        \n        # Generate mock model updates\n        update_weights = {}\n        for layer_name, weights in self.global_model_state.items():\n            if isinstance(weights, dict):\n                update_weights[layer_name] = {}\n                for param_name, param_weights in weights.items():\n                    if isinstance(param_weights, np.ndarray):\n                        # Add noise to simulate local training\n                        noise = np.random.randn(*param_weights.shape) * 0.01\n                        update_weights[layer_name][param_name] = param_weights + noise\n                    else:\n                        update_weights[layer_name][param_name] = param_weights\n            elif isinstance(weights, list):\n                update_weights[layer_name] = []\n                for i, layer_weights in enumerate(weights):\n                    layer_update = {}\n                    for param_name, param_weights in layer_weights.items():\n                        noise = np.random.randn(*param_weights.shape) * 0.01\n                        layer_update[param_name] = param_weights + noise\n                    update_weights[layer_name].append(layer_update)\n        \n        # Calculate mock metrics\n        gradient_norm = random.uniform(0.5, 2.0)\n        local_loss = random.uniform(0.1, 0.8)\n        privacy_spent = random.uniform(0.01, 0.1) if self.privacy_mechanism == 'differential_privacy' else 0.0\n        \n        # Update client state\n        client.last_update_time = time.time()\n        client.round_participation.append(self.current_round)\n        client.privacy_budget -= privacy_spent\n        \n        return ModelUpdate(\n            client_id=client_id,\n            round_number=self.current_round,\n            update_weights=update_weights,\n            gradient_norm=gradient_norm,\n            local_loss=local_loss,\n            samples_count=client.data_size,\n            computation_time=training_time,\n            privacy_spent=privacy_spent\n        )\n    \n    def _aggregate_updates(self, updates: List[ModelUpdate]) -> Dict[str, Any]:\n        \"\"\"Aggregate client updates into global model.\"\"\"\n        if self.aggregation_strategy == 'fedavg':\n            return self._federated_averaging(updates)\n        elif self.aggregation_strategy == 'fedprox':\n            return self._federated_proximal(updates)\n        elif self.aggregation_strategy == 'adaptive':\n            return self._adaptive_aggregation(updates)\n        else:\n            raise ValueError(f\"Unknown aggregation strategy: {self.aggregation_strategy}\")\n    \n    def _federated_averaging(self, updates: List[ModelUpdate]) -> Dict[str, Any]:\n        \"\"\"Standard FedAvg aggregation.\"\"\"\n        # Calculate weights based on data size\n        total_samples = sum(update.samples_count for update in updates)\n        weights = [update.samples_count / total_samples for update in updates]\n        \n        # Initialize aggregated model\n        aggregated = json.loads(json.dumps(self.global_model_state))  # Deep copy\n        \n        # Aggregate each parameter\n        for layer_name in aggregated:\n            if isinstance(aggregated[layer_name], dict):\n                for param_name in aggregated[layer_name]:\n                    if isinstance(aggregated[layer_name][param_name], np.ndarray):\n                        weighted_sum = np.zeros_like(aggregated[layer_name][param_name])\n                        for i, update in enumerate(updates):\n                            if layer_name in update.update_weights:\n                                weighted_sum += weights[i] * update.update_weights[layer_name][param_name]\n                        aggregated[layer_name][param_name] = weighted_sum\n            elif isinstance(aggregated[layer_name], list):\n                for layer_idx in range(len(aggregated[layer_name])):\n                    for param_name in aggregated[layer_name][layer_idx]:\n                        weighted_sum = np.zeros_like(aggregated[layer_name][layer_idx][param_name])\n                        for i, update in enumerate(updates):\n                            if (layer_name in update.update_weights and \n                                layer_idx < len(update.update_weights[layer_name])):\n                                weighted_sum += weights[i] * update.update_weights[layer_name][layer_idx][param_name]\n                        aggregated[layer_name][layer_idx][param_name] = weighted_sum\n        \n        # Update version\n        aggregated['version'] = self.global_model_state['version'] + 1\n        \n        return aggregated\n    \n    def _federated_proximal(self, updates: List[ModelUpdate]) -> Dict[str, Any]:\n        \"\"\"FedProx aggregation with proximal term.\"\"\"\n        # Implement FedProx with regularization\n        aggregated = self._federated_averaging(updates)\n        \n        # Apply proximal regularization (simplified)\n        mu = 0.01  # Proximal term coefficient\n        \n        for layer_name in aggregated:\n            if isinstance(aggregated[layer_name], dict):\n                for param_name in aggregated[layer_name]:\n                    if isinstance(aggregated[layer_name][param_name], np.ndarray):\n                        # Regularize towards global model\n                        global_param = self.global_model_state[layer_name][param_name]\n                        aggregated[layer_name][param_name] = (\n                            (1 - mu) * aggregated[layer_name][param_name] + \n                            mu * global_param\n                        )\n        \n        return aggregated\n    \n    def _adaptive_aggregation(self, updates: List[ModelUpdate]) -> Dict[str, Any]:\n        \"\"\"Adaptive aggregation based on client performance.\"\"\"\n        # Calculate adaptive weights based on loss and gradient norm\n        adaptive_weights = []\n        \n        for update in updates:\n            # Lower loss and reasonable gradient norm get higher weight\n            loss_factor = 1.0 / (update.local_loss + 0.1)\n            gradient_factor = min(2.0, 1.0 / (abs(update.gradient_norm - 1.0) + 0.1))\n            reliability_factor = self.clients[update.client_id].reliability\n            \n            weight = loss_factor * gradient_factor * reliability_factor * update.samples_count\n            adaptive_weights.append(weight)\n        \n        # Normalize weights\n        total_weight = sum(adaptive_weights)\n        adaptive_weights = [w / total_weight for w in adaptive_weights]\n        \n        # Apply weighted aggregation\n        aggregated = json.loads(json.dumps(self.global_model_state))\n        \n        for layer_name in aggregated:\n            if isinstance(aggregated[layer_name], dict):\n                for param_name in aggregated[layer_name]:\n                    if isinstance(aggregated[layer_name][param_name], np.ndarray):\n                        weighted_sum = np.zeros_like(aggregated[layer_name][param_name])\n                        for i, update in enumerate(updates):\n                            if layer_name in update.update_weights:\n                                weighted_sum += adaptive_weights[i] * update.update_weights[layer_name][param_name]\n                        aggregated[layer_name][param_name] = weighted_sum\n        \n        aggregated['version'] = self.global_model_state['version'] + 1\n        return aggregated\n    \n    def _evaluate_global_model(self) -> Dict[str, float]:\n        \"\"\"Evaluate global model performance (mock implementation).\"\"\"\n        # Mock evaluation with realistic progression\n        base_accuracy = 0.85\n        round_bonus = min(0.1, self.current_round * 0.002)\n        noise = random.gauss(0, 0.01)\n        \n        accuracy = min(0.98, base_accuracy + round_bonus + noise)\n        loss = max(0.05, 0.5 - round_bonus * 2 + abs(noise))\n        \n        # Convergence metric (how much model changed)\n        convergence = max(0, 1.0 - self.current_round * 0.02)\n        \n        return {\n            'accuracy': accuracy,\n            'loss': loss,\n            'convergence': convergence,\n            'perplexity': 2 ** loss\n        }\n    \n    def _update_client_reliability(self, updates: List[ModelUpdate]):\n        \"\"\"Update client reliability based on performance.\"\"\"\n        for update in updates:\n            client = self.clients[update.client_id]\n            \n            # Update reliability based on gradient quality and loss\n            if 0.1 <= update.gradient_norm <= 3.0 and update.local_loss < 1.0:\n                client.reliability = min(1.0, client.reliability + 0.01)\n            else:\n                client.reliability = max(0.1, client.reliability - 0.02)\n    \n    def _calculate_aggregation_weights(self, updates: List[ModelUpdate]) -> Dict[str, float]:\n        \"\"\"Calculate final aggregation weights used.\"\"\"\n        if self.aggregation_strategy == 'fedavg':\n            total_samples = sum(update.samples_count for update in updates)\n            return {update.client_id: update.samples_count / total_samples for update in updates}\n        else:\n            # For other strategies, return uniform weights as approximation\n            weight = 1.0 / len(updates)\n            return {update.client_id: weight for update in updates}\n    \n    def _get_selection_strategy_info(self) -> Dict[str, Any]:\n        \"\"\"Get information about client selection strategy.\"\"\"\n        return {\n            'strategy': 'weighted_random',\n            'factors': ['reliability', 'data_size', 'compute_capacity'],\n            'min_clients': self.min_clients_per_round,\n            'max_clients': self.max_clients_per_round\n        }\n    \n    def _generate_client_statistics(self) -> Dict[str, Any]:\n        \"\"\"Generate statistics about client participation.\"\"\"\n        total_clients = len(self.clients)\n        active_clients = sum(1 for c in self.clients.values() if c.round_participation)\n        \n        participation_counts = [len(c.round_participation) for c in self.clients.values()]\n        avg_participation = sum(participation_counts) / len(participation_counts) if participation_counts else 0\n        \n        return {\n            'total_clients': total_clients,\n            'active_clients': active_clients,\n            'average_participation': avg_participation,\n            'client_reliability_distribution': {\n                'mean': np.mean([c.reliability for c in self.clients.values()]),\n                'std': np.std([c.reliability for c in self.clients.values()]),\n                'min': min(c.reliability for c in self.clients.values()),\n                'max': max(c.reliability for c in self.clients.values())\n            }\n        }\n    \n    def _analyze_privacy_usage(self) -> Dict[str, Any]:\n        \"\"\"Analyze privacy budget usage.\"\"\"\n        if self.privacy_mechanism != 'differential_privacy':\n            return {'mechanism': self.privacy_mechanism, 'analysis': 'not_applicable'}\n        \n        total_privacy_spent = sum(\n            1.0 - client.privacy_budget for client in self.clients.values()\n        )\n        \n        avg_privacy_remaining = sum(\n            client.privacy_budget for client in self.clients.values()\n        ) / len(self.clients)\n        \n        return {\n            'mechanism': 'differential_privacy',\n            'total_privacy_spent': total_privacy_spent,\n            'average_privacy_remaining': avg_privacy_remaining,\n            'clients_with_budget': sum(1 for c in self.clients.values() if c.privacy_budget > 0.1)\n        }\n    \n    def _analyze_convergence(self) -> Dict[str, Any]:\n        \"\"\"Analyze training convergence.\"\"\"\n        if len(self.training_history) < 5:\n            return {'status': 'insufficient_data'}\n        \n        # Extract metrics over time\n        accuracies = [h['global_accuracy'] for h in self.training_history]\n        losses = [h['global_loss'] for h in self.training_history]\n        \n        # Calculate improvement rates\n        accuracy_improvement = accuracies[-1] - accuracies[0]\n        loss_reduction = losses[0] - losses[-1]\n        \n        # Recent stability\n        recent_accuracy_std = np.std(accuracies[-5:]) if len(accuracies) >= 5 else float('inf')\n        \n        return {\n            'accuracy_improvement': accuracy_improvement,\n            'loss_reduction': loss_reduction,\n            'recent_stability': recent_accuracy_std,\n            'converged': recent_accuracy_std < 0.005,\n            'convergence_rate': accuracy_improvement / len(self.training_history),\n            'final_accuracy': accuracies[-1],\n            'training_efficiency': accuracy_improvement / len(self.training_history)\n        }\n\n\ndef demonstrate_federated_learning():\n    \"\"\"Demonstrate federated olfactory learning.\"\"\"\n    print(\"🌐 Federated Olfactory Learning Demonstration\")\n    print(\"=\" * 50)\n    \n    # Initialize federated learning system\n    fed_learning = FederatedOlfactoryLearning(\n        aggregation_strategy='adaptive',\n        min_clients_per_round=3,\n        max_clients_per_round=8,\n        privacy_mechanism='differential_privacy',\n        global_epochs=25\n    )\n    \n    # Register diverse clients (simulating different organizations)\n    clients_config = [\n        {'id': 'perfume_house_chanel', 'data_size': 5000, 'compute': 1.5, 'bandwidth': 2.0, 'reliability': 0.95},\n        {'id': 'flavoring_givaudan', 'data_size': 8000, 'compute': 2.0, 'bandwidth': 1.8, 'reliability': 0.90},\n        {'id': 'cosmetics_unilever', 'data_size': 3000, 'compute': 1.2, 'bandwidth': 1.5, 'reliability': 0.85},\n        {'id': 'research_mit', 'data_size': 1500, 'compute': 0.8, 'bandwidth': 1.0, 'reliability': 0.92},\n        {'id': 'startup_olfai', 'data_size': 800, 'compute': 0.6, 'bandwidth': 0.8, 'reliability': 0.75},\n        {'id': 'pharma_roche', 'data_size': 2500, 'compute': 1.8, 'bandwidth': 1.6, 'reliability': 0.88},\n        {'id': 'university_stanford', 'data_size': 1200, 'compute': 0.9, 'bandwidth': 1.1, 'reliability': 0.89},\n        {'id': 'tech_google', 'data_size': 4000, 'compute': 3.0, 'bandwidth': 2.5, 'reliability': 0.93}\n    ]\n    \n    print(\"\\n📝 Registering federated clients:\")\n    for client_config in clients_config:\n        capabilities = {\n            'compute': client_config['compute'],\n            'bandwidth': client_config['bandwidth'],\n            'reliability': client_config['reliability']\n        }\n        \n        fed_learning.register_client(\n            client_id=client_config['id'],\n            data_size=client_config['data_size'],\n            capabilities=capabilities\n        )\n        \n        print(f\"  ✅ {client_config['id']}: {client_config['data_size']} samples\")\n    \n    # Start federated training\n    print(\"\\n🚀 Starting federated training...\")\n    start_time = time.time()\n    \n    results = fed_learning.start_federated_training(target_accuracy=0.92)\n    \n    training_time = time.time() - start_time\n    \n    # Display results\n    print(f\"\\n✅ Federated training completed in {training_time:.2f} seconds\")\n    print(f\"\\n📊 Training Results:\")\n    print(f\"  Final Accuracy: {results['final_accuracy']:.4f}\")\n    print(f\"  Total Rounds: {results['total_rounds']}\")\n    print(f\"  Training Efficiency: {results['convergence_analysis']['training_efficiency']:.6f}\")\n    \n    print(f\"\\n🏆 Client Performance:\")\n    client_stats = results['client_statistics']\n    print(f\"  Active Clients: {client_stats['active_clients']}/{client_stats['total_clients']}\")\n    print(f\"  Average Participation: {client_stats['average_participation']:.1f} rounds\")\n    print(f\"  Reliability Range: {client_stats['client_reliability_distribution']['min']:.3f} - {client_stats['client_reliability_distribution']['max']:.3f}\")\n    \n    print(f\"\\n🔒 Privacy Analysis:\")\n    privacy_stats = results['privacy_analysis']\n    print(f\"  Mechanism: {privacy_stats['mechanism']}\")\n    print(f\"  Total Privacy Spent: {privacy_stats['total_privacy_spent']:.3f}\")\n    print(f\"  Avg. Privacy Remaining: {privacy_stats['average_privacy_remaining']:.3f}\")\n    print(f\"  Clients with Budget: {privacy_stats['clients_with_budget']}\")\n    \n    print(f\"\\n📈 Convergence Analysis:\")\n    convergence = results['convergence_analysis']\n    print(f\"  Accuracy Improvement: {convergence['accuracy_improvement']:.4f}\")\n    print(f\"  Loss Reduction: {convergence['loss_reduction']:.4f}\")\n    print(f\"  Converged: {convergence['converged']}\")\n    print(f\"  Recent Stability: {convergence['recent_stability']:.6f}\")\n    \n    # Show training progression\n    print(f\"\\n📉 Training Progression (last 5 rounds):\")\n    for i, round_info in enumerate(results['training_history'][-5:]):\n        print(f\"  Round {round_info['round'] + 1}: Acc={round_info['global_accuracy']:.4f}, \"\n              f\"Loss={round_info['global_loss']:.4f}, Clients={round_info['participating_clients']}\")\n    \n    return results\n\n\nif __name__ == '__main__':\n    demonstrate_federated_learning()"