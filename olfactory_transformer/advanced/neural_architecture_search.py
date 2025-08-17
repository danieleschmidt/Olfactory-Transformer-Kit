"""Neural Architecture Search for optimal olfactory transformer architectures."""

import logging
import time
import random
import json
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed


@dataclass
class ArchitectureCandidate:
    """Represents a candidate neural architecture."""
    
    layers: List[Dict[str, Any]]
    connections: List[Tuple[int, int]]
    performance_score: float = 0.0
    complexity_score: float = 0.0
    efficiency_score: float = 0.0
    
    def total_parameters(self) -> int:
        """Calculate total number of parameters."""
        total = 0
        for layer in self.layers:
            if layer['type'] == 'transformer':
                hidden_size = layer.get('hidden_size', 512)
                num_heads = layer.get('num_heads', 8)
                total += hidden_size * hidden_size * 4  # Approximation
            elif layer['type'] == 'linear':
                input_size = layer.get('input_size', 512)
                output_size = layer.get('output_size', 512)
                total += input_size * output_size
            elif layer['type'] == 'gnn':
                hidden_size = layer.get('hidden_size', 256)
                total += hidden_size * hidden_size * 2
        return total
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert architecture to dictionary."""
        return {
            'layers': self.layers,
            'connections': self.connections,
            'performance_score': self.performance_score,
            'complexity_score': self.complexity_score,
            'efficiency_score': self.efficiency_score,
            'total_parameters': self.total_parameters()
        }


class NeuralArchitectureSearch:
    """Neural Architecture Search for olfactory transformers."""
    
    def __init__(self, 
                 population_size: int = 20,
                 max_generations: int = 50,
                 mutation_rate: float = 0.3,
                 crossover_rate: float = 0.7):
        self.population_size = population_size
        self.max_generations = max_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        
        # Architecture search space
        self.layer_types = ['transformer', 'gnn', 'linear', 'attention', 'conv1d']
        self.hidden_sizes = [128, 256, 512, 768, 1024, 1536, 2048]
        self.num_heads_options = [4, 8, 12, 16, 20, 24]
        
        self.population = []
        self.best_architectures = []
        
        logging.info(f"🧬 Neural Architecture Search initialized")
        logging.info(f"  Population size: {population_size}")
        logging.info(f"  Max generations: {max_generations}")
    
    def search_optimal_architecture(self, 
                                  objective_function,
                                  constraints: Dict[str, Any] = None) -> Dict[str, Any]:
        """Search for optimal olfactory transformer architecture."""
        logging.info("🧬 Starting Neural Architecture Search")
        
        constraints = constraints or {}
        max_parameters = constraints.get('max_parameters', 50_000_000)
        min_performance = constraints.get('min_performance', 0.8)
        
        # Initialize population
        self.population = self._initialize_population(max_parameters)
        
        search_history = []
        
        for generation in range(self.max_generations):
            generation_start = time.time()
            
            # Evaluate population
            self._evaluate_population(objective_function)
            
            # Track best architectures
            best_in_generation = max(self.population, key=lambda x: x.performance_score)
            self.best_architectures.append(best_in_generation)
            
            # Selection and reproduction
            new_population = self._select_and_reproduce()
            
            # Mutation
            new_population = self._mutate_population(new_population)
            
            # Update population
            self.population = new_population
            
            generation_time = time.time() - generation_start
            
            # Log progress
            avg_performance = sum(arch.performance_score for arch in self.population) / len(self.population)
            best_performance = best_in_generation.performance_score
            
            logging.info(f"  Generation {generation + 1}: Best={best_performance:.4f}, Avg={avg_performance:.4f}, Time={generation_time:.2f}s")
            
            search_history.append({
                'generation': generation + 1,
                'best_performance': best_performance,
                'average_performance': avg_performance,
                'best_architecture': best_in_generation.to_dict(),
                'population_diversity': self._calculate_diversity(),
                'generation_time': generation_time
            })
            
            # Early stopping criteria
            if best_performance > 0.95:  # Near-perfect performance
                logging.info(f"  Early stopping: Excellent performance achieved")
                break
        
        # Final evaluation and ranking
        final_best = max(self.best_architectures, key=lambda x: x.performance_score)
        
        return {
            'best_architecture': final_best.to_dict(),
            'search_history': search_history,
            'final_population': [arch.to_dict() for arch in self.population],
            'convergence_analysis': self._analyze_convergence(search_history),
            'pareto_front': self._calculate_pareto_front()
        }
    
    def _initialize_population(self, max_parameters: int) -> List[ArchitectureCandidate]:
        """Initialize random population of architectures."""
        population = []
        
        for _ in range(self.population_size):
            arch = self._generate_random_architecture(max_parameters)
            population.append(arch)
        
        return population
    
    def _generate_random_architecture(self, max_parameters: int) -> ArchitectureCandidate:
        """Generate a random architecture within constraints."""
        layers = []
        connections = []
        
        # Number of layers (between 3 and 12)
        num_layers = random.randint(3, 12)
        
        for i in range(num_layers):
            layer_type = random.choice(self.layer_types)
            
            if layer_type == 'transformer':
                layer = {
                    'type': 'transformer',
                    'hidden_size': random.choice(self.hidden_sizes),
                    'num_heads': random.choice(self.num_heads_options),
                    'intermediate_size': random.choice([2048, 3072, 4096]),
                    'dropout': random.uniform(0.0, 0.3)
                }
            elif layer_type == 'gnn':
                layer = {
                    'type': 'gnn',
                    'hidden_size': random.choice(self.hidden_sizes[:5]),  # Smaller for GNN
                    'num_layers': random.randint(2, 6),
                    'aggregation': random.choice(['mean', 'max', 'sum', 'attention'])
                }
            elif layer_type == 'linear':
                layer = {
                    'type': 'linear',
                    'input_size': random.choice(self.hidden_sizes),
                    'output_size': random.choice(self.hidden_sizes),
                    'activation': random.choice(['relu', 'gelu', 'swish', 'tanh'])
                }
            elif layer_type == 'attention':
                layer = {
                    'type': 'attention',
                    'hidden_size': random.choice(self.hidden_sizes),
                    'num_heads': random.choice(self.num_heads_options),
                    'attention_type': random.choice(['self', 'cross', 'multi_scale'])
                }
            else:  # conv1d
                layer = {
                    'type': 'conv1d',
                    'in_channels': random.choice([1, 3, 16, 32]),
                    'out_channels': random.choice([16, 32, 64, 128]),
                    'kernel_size': random.choice([3, 5, 7, 9]),
                    'stride': random.choice([1, 2])
                }
            
            layers.append(layer)
            
            # Add connections (sequential + some skip connections)
            if i > 0:
                connections.append((i-1, i))  # Sequential connection
                
                # Random skip connections
                if i > 1 and random.random() < 0.3:
                    skip_target = random.randint(0, i-2)
                    connections.append((skip_target, i))
        
        arch = ArchitectureCandidate(layers=layers, connections=connections)
        
        # Ensure parameter constraint
        if arch.total_parameters() > max_parameters:
            # Simplify architecture
            arch = self._simplify_architecture(arch, max_parameters)
        
        return arch
    
    def _simplify_architecture(self, arch: ArchitectureCandidate, max_parameters: int) -> ArchitectureCandidate:
        """Simplify architecture to meet parameter constraints."""
        while arch.total_parameters() > max_parameters and len(arch.layers) > 2:
            # Remove random layer or reduce size
            if random.random() < 0.5 and len(arch.layers) > 3:
                # Remove a layer
                idx_to_remove = random.randint(1, len(arch.layers) - 2)
                arch.layers.pop(idx_to_remove)
                # Update connections
                arch.connections = [(s, t-1 if t > idx_to_remove else t) 
                                   for s, t in arch.connections 
                                   if s != idx_to_remove and t != idx_to_remove]
            else:
                # Reduce layer size
                layer_idx = random.randint(0, len(arch.layers) - 1)
                layer = arch.layers[layer_idx]
                
                if layer['type'] == 'transformer' and layer['hidden_size'] > 256:
                    layer['hidden_size'] = max(256, layer['hidden_size'] // 2)
                elif layer['type'] == 'linear':
                    layer['input_size'] = max(128, layer['input_size'] // 2)
                    layer['output_size'] = max(128, layer['output_size'] // 2)
        
        return arch
    
    def _evaluate_population(self, objective_function):
        """Evaluate all architectures in the population."""
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(self._evaluate_architecture, arch, objective_function): arch 
                      for arch in self.population}
            
            for future in as_completed(futures):
                arch = futures[future]
                try:
                    performance, complexity, efficiency = future.result()
                    arch.performance_score = performance
                    arch.complexity_score = complexity
                    arch.efficiency_score = efficiency
                except Exception as e:
                    logging.warning(f"Architecture evaluation failed: {e}")
                    arch.performance_score = 0.0
                    arch.complexity_score = 1.0
                    arch.efficiency_score = 0.0
    
    def _evaluate_architecture(self, arch: ArchitectureCandidate, objective_function) -> Tuple[float, float, float]:
        """Evaluate a single architecture."""
        try:
            # Call objective function with architecture specification
            arch_spec = {
                'layers': arch.layers,
                'connections': arch.connections,
                'total_parameters': arch.total_parameters()
            }
            
            performance = objective_function(arch_spec)
            
            # Calculate complexity (normalized by parameter count)
            complexity = min(1.0, arch.total_parameters() / 10_000_000)
            
            # Calculate efficiency (performance per parameter)
            efficiency = performance / (arch.total_parameters() / 1_000_000 + 1)
            
            return performance, complexity, efficiency
        
        except Exception as e:
            logging.warning(f"Architecture evaluation error: {e}")
            return 0.0, 1.0, 0.0
    
    def _select_and_reproduce(self) -> List[ArchitectureCandidate]:
        """Select best architectures and create new generation."""
        # Tournament selection
        new_population = []
        
        # Keep best 20% unchanged (elitism)
        sorted_pop = sorted(self.population, key=lambda x: x.performance_score, reverse=True)
        elite_count = max(1, int(0.2 * self.population_size))
        new_population.extend(sorted_pop[:elite_count])
        
        # Fill rest through crossover and selection
        while len(new_population) < self.population_size:
            if random.random() < self.crossover_rate and len(new_population) < self.population_size - 1:
                # Crossover
                parent1 = self._tournament_selection()
                parent2 = self._tournament_selection()
                child1, child2 = self._crossover(parent1, parent2)
                new_population.extend([child1, child2])
            else:
                # Direct selection
                selected = self._tournament_selection()
                new_population.append(selected)
        
        return new_population[:self.population_size]
    
    def _tournament_selection(self, tournament_size: int = 3) -> ArchitectureCandidate:
        """Select architecture through tournament selection."""
        tournament = random.sample(self.population, min(tournament_size, len(self.population)))
        return max(tournament, key=lambda x: x.performance_score)
    
    def _crossover(self, parent1: ArchitectureCandidate, parent2: ArchitectureCandidate) -> Tuple[ArchitectureCandidate, ArchitectureCandidate]:
        """Create offspring through crossover."""
        # Layer-wise crossover
        min_layers = min(len(parent1.layers), len(parent2.layers))
        crossover_point = random.randint(1, min_layers - 1)
        
        child1_layers = parent1.layers[:crossover_point] + parent2.layers[crossover_point:]
        child2_layers = parent2.layers[:crossover_point] + parent1.layers[crossover_point:]
        
        # Simple connection inheritance
        child1_connections = parent1.connections[:len(child1_layers)-1]
        child2_connections = parent2.connections[:len(child2_layers)-1]
        
        child1 = ArchitectureCandidate(layers=child1_layers, connections=child1_connections)
        child2 = ArchitectureCandidate(layers=child2_layers, connections=child2_connections)
        
        return child1, child2
    
    def _mutate_population(self, population: List[ArchitectureCandidate]) -> List[ArchitectureCandidate]:
        """Apply mutations to population."""
        for arch in population:
            if random.random() < self.mutation_rate:
                self._mutate_architecture(arch)
        return population
    
    def _mutate_architecture(self, arch: ArchitectureCandidate):
        """Mutate a single architecture."""
        mutation_type = random.choice(['modify_layer', 'add_layer', 'remove_layer', 'modify_connection'])
        
        if mutation_type == 'modify_layer' and arch.layers:
            # Modify random layer
            layer_idx = random.randint(0, len(arch.layers) - 1)
            layer = arch.layers[layer_idx]
            
            if layer['type'] == 'transformer':
                if random.random() < 0.5:
                    layer['hidden_size'] = random.choice(self.hidden_sizes)
                else:
                    layer['num_heads'] = random.choice(self.num_heads_options)
            elif layer['type'] == 'linear':
                if random.random() < 0.5:
                    layer['input_size'] = random.choice(self.hidden_sizes)
                else:
                    layer['output_size'] = random.choice(self.hidden_sizes)
        
        elif mutation_type == 'add_layer' and len(arch.layers) < 15:
            # Add new layer
            new_layer_type = random.choice(self.layer_types)
            if new_layer_type == 'transformer':
                new_layer = {
                    'type': 'transformer',
                    'hidden_size': random.choice(self.hidden_sizes),
                    'num_heads': random.choice(self.num_heads_options),
                    'dropout': random.uniform(0.0, 0.3)
                }
            else:
                new_layer = {'type': new_layer_type}
            
            insert_idx = random.randint(0, len(arch.layers))
            arch.layers.insert(insert_idx, new_layer)
        
        elif mutation_type == 'remove_layer' and len(arch.layers) > 3:
            # Remove random layer
            remove_idx = random.randint(1, len(arch.layers) - 2)
            arch.layers.pop(remove_idx)
        
        elif mutation_type == 'modify_connection' and len(arch.connections) > 1:
            # Modify random connection
            conn_idx = random.randint(0, len(arch.connections) - 1)
            max_layer = len(arch.layers) - 1
            if max_layer > 0:
                source = random.randint(0, max_layer - 1)
                target = random.randint(source + 1, max_layer)
                arch.connections[conn_idx] = (source, target)
    
    def _calculate_diversity(self) -> float:
        """Calculate population diversity."""
        if len(self.population) < 2:
            return 0.0
        
        diversity_sum = 0.0
        comparisons = 0
        
        for i in range(len(self.population)):
            for j in range(i + 1, len(self.population)):
                # Simple diversity metric based on architecture differences
                arch1, arch2 = self.population[i], self.population[j]
                
                layer_diff = abs(len(arch1.layers) - len(arch2.layers))
                param_diff = abs(arch1.total_parameters() - arch2.total_parameters()) / 1_000_000
                
                diversity = layer_diff + param_diff
                diversity_sum += diversity
                comparisons += 1
        
        return diversity_sum / comparisons if comparisons > 0 else 0.0
    
    def _analyze_convergence(self, history: List[Dict]) -> Dict[str, Any]:
        """Analyze convergence of the search process."""
        if len(history) < 5:
            return {'status': 'insufficient_data'}
        
        # Performance improvement analysis
        best_scores = [h['best_performance'] for h in history]
        avg_scores = [h['average_performance'] for h in history]
        
        initial_best = best_scores[0]
        final_best = best_scores[-1]
        improvement = final_best - initial_best
        
        # Convergence rate
        recent_improvement = best_scores[-1] - best_scores[-5] if len(best_scores) >= 5 else 0
        
        # Diversity trend
        diversities = [h.get('population_diversity', 0) for h in history]
        diversity_trend = diversities[-1] - diversities[0] if diversities else 0
        
        return {
            'total_improvement': improvement,
            'recent_improvement': recent_improvement,
            'convergence_rate': recent_improvement / 5,
            'diversity_trend': diversity_trend,
            'final_diversity': diversities[-1] if diversities else 0,
            'converged': recent_improvement < 0.001,
            'search_efficiency': improvement / len(history)
        }
    
    def _calculate_pareto_front(self) -> List[Dict[str, Any]]:
        """Calculate Pareto front of performance vs complexity."""
        pareto_front = []
        
        for arch in self.population:
            is_dominated = False
            
            for other in self.population:
                if (other.performance_score >= arch.performance_score and 
                    other.efficiency_score >= arch.efficiency_score and
                    (other.performance_score > arch.performance_score or 
                     other.efficiency_score > arch.efficiency_score)):
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto_front.append(arch.to_dict())
        
        return pareto_front


def nas_objective_example(architecture_spec: Dict[str, Any]) -> float:
    """Example objective function for Neural Architecture Search."""
    layers = architecture_spec['layers']
    total_params = architecture_spec['total_parameters']
    
    # Base performance
    base_score = 0.7
    
    # Reward transformer layers
    transformer_count = sum(1 for layer in layers if layer['type'] == 'transformer')
    transformer_bonus = min(0.15, transformer_count * 0.03)
    
    # Reward balanced architecture
    layer_types = [layer['type'] for layer in layers]
    type_diversity = len(set(layer_types)) / len(layers)
    diversity_bonus = type_diversity * 0.1
    
    # Penalize excessive parameters
    param_penalty = min(0.2, (total_params - 5_000_000) / 10_000_000) if total_params > 5_000_000 else 0
    
    # Reward optimal depth
    depth = len(layers)
    depth_bonus = 0.05 if 6 <= depth <= 12 else -0.05
    
    # Add realistic noise
    noise = random.gauss(0, 0.03)
    
    score = base_score + transformer_bonus + diversity_bonus - param_penalty + depth_bonus + noise
    
    return max(0, min(1, score))


def demonstrate_nas():
    """Demonstrate Neural Architecture Search for olfactory models."""
    print("🧬 Neural Architecture Search for Olfactory Transformers")
    print("=" * 60)
    
    # Initialize NAS
    nas = NeuralArchitectureSearch(
        population_size=12,
        max_generations=15,
        mutation_rate=0.4,
        crossover_rate=0.6
    )
    
    # Define constraints
    constraints = {
        'max_parameters': 20_000_000,
        'min_performance': 0.8
    }
    
    # Run architecture search
    start_time = time.time()
    results = nas.search_optimal_architecture(
        objective_function=nas_objective_example,
        constraints=constraints
    )
    search_time = time.time() - start_time
    
    # Display results
    print(f"\n✅ Architecture search completed in {search_time:.2f} seconds")
    
    best_arch = results['best_architecture']
    print(f"\n🏆 Best Architecture:")
    print(f"  Layers: {len(best_arch['layers'])}")
    print(f"  Parameters: {best_arch['total_parameters']:,}")
    print(f"  Performance: {best_arch['performance_score']:.4f}")
    print(f"  Efficiency: {best_arch['efficiency_score']:.4f}")
    
    print(f"\n📊 Search Analysis:")
    convergence = results['convergence_analysis']
    print(f"  Total Improvement: {convergence['total_improvement']:.4f}")
    print(f"  Search Efficiency: {convergence['search_efficiency']:.4f}")
    print(f"  Converged: {convergence['converged']}")
    
    print(f"\n🏗️ Architecture Details:")
    for i, layer in enumerate(best_arch['layers'][:5]):  # Show first 5 layers
        print(f"  Layer {i+1}: {layer['type']} - {layer}")
    
    print(f"\n🔄 Pareto Front: {len(results['pareto_front'])} architectures")
    
    return results


if __name__ == '__main__':
    demonstrate_nas()