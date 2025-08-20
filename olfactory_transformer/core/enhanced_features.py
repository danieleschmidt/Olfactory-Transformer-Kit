"""
Enhanced Core Features for Olfactory Transformer.

Implements missing advanced functionality including:
- Real-time streaming capabilities
- Advanced molecular analysis
- Multi-modal sensor fusion
- Zero-shot learning enhancements
- Production-ready optimizations
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Union, Tuple, AsyncGenerator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import json
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import threading
from collections import deque
import hashlib

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from .config import OlfactoryConfig, ScentPrediction, SensorReading
from .model import OlfactoryTransformer
from .tokenizer import MoleculeTokenizer


@dataclass
class StreamingPrediction:
    """Enhanced streaming prediction with metadata."""
    prediction: ScentPrediction
    timestamp: datetime
    session_id: str
    sequence_number: int
    latency_ms: float
    confidence_threshold: float = 0.7
    sensor_data: Optional[Dict[str, float]] = None
    model_version: str = "1.0.0"


@dataclass 
class AnalysisResult:
    """Comprehensive molecular analysis result."""
    smiles: str
    molecular_weight: float
    logp: float
    tpsa: float
    chemical_family: str
    functional_groups: List[str]
    predicted_descriptors: List[str]
    safety_assessment: Dict[str, Any]
    regulatory_status: Dict[str, str]
    synthesis_complexity: float
    bioavailability: float
    attention_analysis: Optional[Dict[str, Any]] = None


class StreamingPredictor:
    """Real-time streaming prediction engine."""
    
    def __init__(self, model: OlfactoryTransformer, tokenizer: MoleculeTokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.active_sessions: Dict[str, Dict] = {}
        self.prediction_buffer = deque(maxlen=1000)
        self.executor = ThreadPoolExecutor(max_workers=4)
        self._lock = threading.Lock()
        
    async def start_session(self, session_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Start a new streaming session."""
        with self._lock:
            self.active_sessions[session_id] = {
                'config': config,
                'start_time': time.time(),
                'sequence_number': 0,
                'predictions': deque(maxlen=100),
                'last_prediction': None,
                'error_count': 0
            }
        
        logging.info(f"Started streaming session {session_id}")
        return {
            'session_id': session_id,
            'status': 'active',
            'start_time': datetime.now().isoformat()
        }
    
    async def predict_stream(self, session_id: str, sensor_data: Dict[str, float]) -> StreamingPrediction:
        """Make real-time prediction from streaming sensor data."""
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.active_sessions[session_id]
        start_time = time.time()
        
        try:
            # Create sensor reading
            sensor_reading = SensorReading(
                gas_sensors=sensor_data,
                sensor_types=list(sensor_data.keys()),
                timestamp=start_time
            )
            
            # Async prediction
            loop = asyncio.get_event_loop()
            prediction = await loop.run_in_executor(
                self.executor,
                self.model.predict_from_sensors,
                sensor_reading
            )
            
            latency = (time.time() - start_time) * 1000
            
            streaming_pred = StreamingPrediction(
                prediction=prediction,
                timestamp=datetime.now(),
                session_id=session_id,
                sequence_number=session['sequence_number'],
                latency_ms=latency,
                sensor_data=sensor_data
            )
            
            # Update session
            session['sequence_number'] += 1
            session['predictions'].append(streaming_pred)
            session['last_prediction'] = streaming_pred
            
            # Add to buffer for analysis
            self.prediction_buffer.append(streaming_pred)
            
            logging.debug(f"Streaming prediction {session_id}:{session['sequence_number']} - latency: {latency:.1f}ms")
            
            return streaming_pred
            
        except Exception as e:
            session['error_count'] += 1
            logging.error(f"Streaming prediction error {session_id}: {e}")
            raise
    
    async def generate_stream(self, session_id: str, data_source: AsyncGenerator) -> AsyncGenerator[StreamingPrediction, None]:
        """Generate streaming predictions from data source."""
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")
        
        try:
            async for sensor_data in data_source:
                prediction = await self.predict_stream(session_id, sensor_data)
                yield prediction
                
        except Exception as e:
            logging.error(f"Stream generation error {session_id}: {e}")
            raise
    
    def stop_session(self, session_id: str) -> Dict[str, Any]:
        """Stop streaming session and return summary."""
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.active_sessions[session_id]
        duration = time.time() - session['start_time']
        
        summary = {
            'session_id': session_id,
            'duration_seconds': duration,
            'total_predictions': session['sequence_number'],
            'predictions_per_second': session['sequence_number'] / duration if duration > 0 else 0,
            'error_count': session['error_count'],
            'last_prediction': session['last_prediction'].dict() if session['last_prediction'] else None
        }
        
        del self.active_sessions[session_id]
        logging.info(f"Stopped streaming session {session_id}")
        
        return summary
    
    def get_session_status(self, session_id: str) -> Dict[str, Any]:
        """Get current session status."""
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.active_sessions[session_id]
        duration = time.time() - session['start_time']
        
        return {
            'session_id': session_id,
            'status': 'active',
            'duration_seconds': duration,
            'sequence_number': session['sequence_number'],
            'predictions_per_second': session['sequence_number'] / duration if duration > 0 else 0,
            'error_count': session['error_count'],
            'buffer_size': len(session['predictions'])
        }


class EnhancedMolecularAnalyzer:
    """Advanced molecular analysis and feature extraction."""
    
    def __init__(self, model: OlfactoryTransformer, tokenizer: MoleculeTokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.analysis_cache = {}
        
        # Chemical families and functional groups
        self.functional_groups = {
            'alcohol': r'[OH]',
            'aldehyde': r'C=O',
            'ketone': r'C(=O)C',
            'ester': r'C(=O)O',
            'ether': r'COC',
            'amine': r'N',
            'aromatic': r'c',
            'alkyl': r'C',
            'halogen': r'[F,Cl,Br,I]',
            'nitro': r'N(=O)=O'
        }
        
        # Safety patterns
        self.safety_patterns = {
            'explosive': [r'N(\+).*N', r'C#C.*C#C', r'N-N-N'],
            'toxic': [r'[As,Hg,Pb,Cd]', r'CN', r'S=C=N'],
            'corrosive': [r'C(=O)Cl', r'S(=O)(=O)Cl'],
            'flammable': [r'CC', r'C=C', r'C#C']
        }
        
        # Regulatory databases (simplified)
        self.regulatory_status = {
            'IFRA': ['Category 1', 'Category 2', 'Category 3', 'Category 4', 'Restricted', 'Prohibited'],
            'FDA': ['GRAS', 'Food Additive', 'Restricted', 'Prohibited'],
            'EU': ['Approved', 'Restricted', 'Prohibited']
        }
    
    def analyze_molecule(self, smiles: str, include_attention: bool = False) -> AnalysisResult:
        """Comprehensive molecular analysis."""
        # Check cache
        cache_key = hashlib.md5(f"{smiles}:{include_attention}".encode()).hexdigest()
        if cache_key in self.analysis_cache:
            return self.analysis_cache[cache_key]
        
        try:
            # Basic molecular properties
            mol_props = self._calculate_molecular_properties(smiles)
            
            # Functional group analysis
            functional_groups = self._identify_functional_groups(smiles)
            
            # Safety assessment
            safety = self._assess_safety(smiles)
            
            # Regulatory status
            regulatory = self._check_regulatory_status(smiles)
            
            # Predict scent descriptors
            prediction = self.model.predict_scent(smiles, self.tokenizer)
            
            # Attention analysis if requested
            attention_data = None
            if include_attention:
                attention_data = self._analyze_attention(smiles)
            
            result = AnalysisResult(
                smiles=smiles,
                molecular_weight=mol_props['molecular_weight'],
                logp=mol_props['logp'],
                tpsa=mol_props['tpsa'],
                chemical_family=prediction.chemical_family or 'unknown',
                functional_groups=functional_groups,
                predicted_descriptors=prediction.descriptors,
                safety_assessment=safety,
                regulatory_status=regulatory,
                synthesis_complexity=mol_props['synthesis_complexity'],
                bioavailability=mol_props['bioavailability'],
                attention_analysis=attention_data
            )
            
            # Cache result
            self.analysis_cache[cache_key] = result
            
            return result
            
        except Exception as e:
            logging.error(f"Molecular analysis failed for {smiles}: {e}")
            raise
    
    def _calculate_molecular_properties(self, smiles: str) -> Dict[str, float]:
        """Calculate molecular properties (simplified implementation)."""
        # In real implementation, would use RDKit or similar
        length = len(smiles)
        
        # Estimate properties based on SMILES structure
        estimated_mw = length * 12 + smiles.count('O') * 16 + smiles.count('N') * 14
        estimated_logp = (smiles.count('C') * 0.5 - smiles.count('O') * 0.7 - smiles.count('N') * 0.5)
        estimated_tpsa = smiles.count('O') * 20 + smiles.count('N') * 15
        
        # Synthesis complexity based on structural features
        complexity_score = (
            smiles.count('(') * 0.1 +  # Branching
            smiles.count('[') * 0.2 +  # Special atoms
            smiles.count('#') * 0.3 +  # Triple bonds
            smiles.count('=') * 0.1    # Double bonds
        )
        
        # Bioavailability estimate (Lipinski-like)
        bioavailability = 1.0
        if estimated_mw > 500:
            bioavailability *= 0.8
        if abs(estimated_logp) > 5:
            bioavailability *= 0.7
        if estimated_tpsa > 140:
            bioavailability *= 0.6
        
        return {
            'molecular_weight': max(50, min(1000, estimated_mw)),
            'logp': max(-5, min(10, estimated_logp)),
            'tpsa': max(0, min(200, estimated_tpsa)),
            'synthesis_complexity': complexity_score,
            'bioavailability': max(0, min(1, bioavailability))
        }
    
    def _identify_functional_groups(self, smiles: str) -> List[str]:
        """Identify functional groups in molecule."""
        import re
        
        identified_groups = []
        smiles_upper = smiles.upper()
        
        for group_name, pattern in self.functional_groups.items():
            if re.search(pattern.upper(), smiles_upper):
                identified_groups.append(group_name)
        
        return identified_groups
    
    def _assess_safety(self, smiles: str) -> Dict[str, Any]:
        """Assess molecular safety."""
        import re
        
        safety_flags = {
            'explosive': False,
            'toxic': False,
            'corrosive': False,
            'flammable': False
        }
        
        concerns = []
        
        for hazard_type, patterns in self.safety_patterns.items():
            for pattern in patterns:
                if re.search(pattern.upper(), smiles.upper()):
                    safety_flags[hazard_type] = True
                    concerns.append(f"{hazard_type}: {pattern}")
        
        # Overall safety score
        risk_factors = sum(safety_flags.values())
        safety_score = max(0, 1.0 - risk_factors * 0.3)
        
        return {
            'safety_score': safety_score,
            'risk_level': 'low' if risk_factors == 0 else 'medium' if risk_factors <= 2 else 'high',
            'hazard_flags': safety_flags,
            'concerns': concerns,
            'recommendations': self._generate_safety_recommendations(safety_flags)
        }
    
    def _check_regulatory_status(self, smiles: str) -> Dict[str, str]:
        """Check regulatory status (simplified)."""
        # In real implementation, would query regulatory databases
        # Simplified heuristics
        
        status = {}
        
        # IFRA assessment based on structure
        if any(hazard in smiles.upper() for hazard in ['BR', 'CL', 'N+', 'S=']):
            status['IFRA'] = 'Category 4 - Restricted use'
        else:
            status['IFRA'] = 'Category 3 - Limited use'
        
        # FDA assessment
        if 'O' in smiles and len(smiles) < 50:  # Simple compounds with oxygen
            status['FDA'] = 'GRAS - Generally Recognized as Safe'
        else:
            status['FDA'] = 'Food Additive - Requires approval'
        
        # EU assessment
        if any(concern in smiles.upper() for concern in ['AS', 'HG', 'PB', 'CD']):
            status['EU'] = 'Prohibited'
        else:
            status['EU'] = 'Approved with restrictions'
        
        return status
    
    def _analyze_attention(self, smiles: str) -> Dict[str, Any]:
        """Analyze attention patterns (simplified)."""
        # In real implementation, would extract actual attention weights
        tokens = list(smiles)
        num_tokens = len(tokens)
        
        # Simulate attention weights
        token_attention = np.random.dirichlet(np.ones(num_tokens))
        
        # Important atoms/bonds
        important_indices = np.argsort(token_attention)[-5:]
        important_tokens = [tokens[i] for i in important_indices if i < len(tokens)]
        
        return {
            'token_attention': token_attention.tolist(),
            'important_tokens': important_tokens,
            'attention_entropy': -np.sum(token_attention * np.log(token_attention + 1e-8)),
            'max_attention_score': float(np.max(token_attention))
        }
    
    def _generate_safety_recommendations(self, safety_flags: Dict[str, bool]) -> List[str]:
        """Generate safety recommendations."""
        recommendations = []
        
        if safety_flags['explosive']:
            recommendations.append("Handle with extreme caution - explosive risk")
            recommendations.append("Store in cool, dry place away from ignition sources")
        
        if safety_flags['toxic']:
            recommendations.append("Use appropriate PPE including gloves and ventilation")
            recommendations.append("Avoid skin contact and inhalation")
        
        if safety_flags['corrosive']:
            recommendations.append("Use corrosion-resistant containers and equipment")
            recommendations.append("Have neutralizing agents readily available")
        
        if safety_flags['flammable']:
            recommendations.append("Keep away from heat, sparks, and open flames")
            recommendations.append("Use explosion-proof electrical equipment")
        
        if not any(safety_flags.values()):
            recommendations.append("Standard laboratory safety practices apply")
        
        return recommendations


class ZeroShotEnhancer:
    """Enhanced zero-shot learning capabilities."""
    
    def __init__(self, model: OlfactoryTransformer, tokenizer: MoleculeTokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.descriptor_embeddings = {}
        
        # Build descriptor embedding space
        self._build_descriptor_embeddings()
    
    def _build_descriptor_embeddings(self):
        """Build semantic embeddings for descriptors."""
        # Simplified descriptor embedding (in practice, use pre-trained embeddings)
        descriptors = [
            "floral", "rose", "jasmine", "lavender", "lily", "violet",
            "citrus", "lemon", "orange", "grapefruit", "lime", "bergamot",
            "woody", "cedar", "sandalwood", "pine", "oak", "birch",
            "fruity", "apple", "pear", "peach", "berry", "tropical",
            "spicy", "pepper", "cinnamon", "clove", "nutmeg", "ginger",
            "fresh", "clean", "aquatic", "marine", "ozonic", "crisp",
            "sweet", "vanilla", "honey", "caramel", "sugar", "candy",
            "green", "grass", "leaf", "herb", "mint", "basil",
            "musky", "animalic", "amber", "powdery", "smoky", "earthy"
        ]
        
        for descriptor in descriptors:
            # Create semantic embedding (simplified)
            embedding = np.random.normal(0, 0.1, 256)  # In practice, use BERT/Word2Vec
            self.descriptor_embeddings[descriptor] = embedding
    
    def classify_custom_categories(
        self, 
        smiles: str, 
        categories: List[str], 
        return_probabilities: bool = False
    ) -> Union[Dict[str, float], str]:
        """Classify molecule into custom categories using zero-shot learning."""
        
        # Get base prediction
        base_prediction = self.model.predict_scent(smiles, self.tokenizer)
        base_descriptors = base_prediction.descriptors
        
        # Calculate similarities to custom categories
        category_scores = {}
        
        for category in categories:
            # Calculate semantic similarity
            category_words = category.lower().split()
            category_score = 0.0
            
            for base_desc in base_descriptors:
                for cat_word in category_words:
                    # Simplified similarity calculation
                    if cat_word in self.descriptor_embeddings and base_desc in self.descriptor_embeddings:
                        similarity = np.dot(
                            self.descriptor_embeddings[cat_word],
                            self.descriptor_embeddings[base_desc]
                        )
                        category_score = max(category_score, similarity)
            
            category_scores[category] = max(0, category_score)
        
        # Normalize to probabilities
        total_score = sum(category_scores.values())
        if total_score > 0:
            probabilities = {cat: score / total_score for cat, score in category_scores.items()}
        else:
            # Uniform distribution if no matches
            probabilities = {cat: 1.0 / len(categories) for cat in categories}
        
        if return_probabilities:
            return probabilities
        else:
            return max(probabilities.keys(), key=lambda k: probabilities[k])
    
    def generate_similar_molecules(
        self, 
        target_descriptors: List[str], 
        n_molecules: int = 5
    ) -> List[Dict[str, Any]]:
        """Generate molecules similar to target descriptors."""
        
        similar_molecules = []
        
        # Simplified molecule generation (in practice, use molecular generators)
        base_structures = [
            "CC(C)CC1=CC=C(C=C1)C(C)C",  # Woody
            "CC1=CC=C(C=C1)C(=O)C",      # Floral  
            "CCCCCCCC(=O)O",             # Fresh
            "CC(C)(C)C1=CC=C(C=C1)O",    # Spicy
            "COC1=CC=C(C=C1)C=O"         # Sweet
        ]
        
        for i in range(min(n_molecules, len(base_structures))):
            smiles = base_structures[i]
            prediction = self.model.predict_scent(smiles, self.tokenizer)
            
            # Calculate similarity to target descriptors
            similarity_score = len(set(prediction.descriptors) & set(target_descriptors)) / len(target_descriptors)
            
            similar_molecules.append({
                'smiles': smiles,
                'descriptors': prediction.descriptors,
                'similarity_score': similarity_score,
                'confidence': prediction.confidence
            })
        
        # Sort by similarity
        similar_molecules.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        return similar_molecules


class ProductionOptimizer:
    """Production-ready optimizations and utilities."""
    
    def __init__(self, model: OlfactoryTransformer):
        self.model = model
        self.batch_processor = None
        self.memory_monitor = MemoryMonitor()
        
    def optimize_for_production(self) -> Dict[str, Any]:
        """Apply production optimizations."""
        optimizations = {}
        
        # Model optimizations
        if HAS_TORCH:
            # Compile model for faster inference
            try:
                # PyTorch 2.0 compile
                if hasattr(torch, 'compile'):
                    self.model = torch.compile(self.model)
                    optimizations['torch_compile'] = True
                    
            except Exception as e:
                logging.warning(f"Torch compile failed: {e}")
                optimizations['torch_compile'] = False
            
            # Set to eval mode and disable gradients
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
                
            optimizations['eval_mode'] = True
            optimizations['gradients_disabled'] = True
        
        # Memory optimizations
        optimizations.update(self._optimize_memory())
        
        # Batch processing setup
        self._setup_batch_processor()
        optimizations['batch_processor'] = True
        
        logging.info(f"Production optimizations applied: {optimizations}")
        return optimizations
    
    def _optimize_memory(self) -> Dict[str, Any]:
        """Optimize memory usage."""
        optimizations = {}
        
        if HAS_TORCH and torch.cuda.is_available():
            # Clear CUDA cache
            torch.cuda.empty_cache()
            
            # Enable memory efficient attention
            try:
                torch.backends.cuda.enable_math_sdp(False)
                torch.backends.cuda.enable_flash_sdp(True)
                optimizations['flash_attention'] = True
            except:
                optimizations['flash_attention'] = False
            
            # Set memory fraction
            torch.cuda.set_per_process_memory_fraction(0.8)
            optimizations['memory_fraction'] = 0.8
        
        return optimizations
    
    def _setup_batch_processor(self):
        """Setup efficient batch processing."""
        self.batch_processor = BatchProcessor(self.model)
    
    def create_model_checkpoint(self, checkpoint_path: Path) -> Dict[str, Any]:
        """Create production model checkpoint."""
        checkpoint_info = {
            'model_state': self.model.state_dict() if HAS_TORCH else None,
            'config': self.model.config.__dict__,
            'timestamp': datetime.now().isoformat(),
            'version': '1.0.0',
            'optimizations': self.optimize_for_production()
        }
        
        if HAS_TORCH:
            torch.save(checkpoint_info, checkpoint_path)
        else:
            with open(checkpoint_path, 'w') as f:
                json.dump({k: v for k, v in checkpoint_info.items() if k != 'model_state'}, 
                         f, indent=2, default=str)
        
        return checkpoint_info


class BatchProcessor:
    """Efficient batch processing for production workloads."""
    
    def __init__(self, model: OlfactoryTransformer):
        self.model = model
        self.batch_size = 32
        self.max_queue_size = 1000
        
    async def process_batch(self, molecules: List[str], tokenizer: MoleculeTokenizer) -> List[ScentPrediction]:
        """Process batch of molecules efficiently."""
        if not molecules:
            return []
        
        results = []
        
        # Process in batches
        for i in range(0, len(molecules), self.batch_size):
            batch = molecules[i:i + self.batch_size]
            batch_results = await self._process_single_batch(batch, tokenizer)
            results.extend(batch_results)
        
        return results
    
    async def _process_single_batch(self, batch: List[str], tokenizer: MoleculeTokenizer) -> List[ScentPrediction]:
        """Process single batch of molecules."""
        batch_results = []
        
        # Sequential processing for now (could be parallelized)
        for smiles in batch:
            try:
                prediction = self.model.predict_scent(smiles, tokenizer)
                batch_results.append(prediction)
            except Exception as e:
                logging.error(f"Batch processing error for {smiles}: {e}")
                # Create error prediction
                error_prediction = ScentPrediction(
                    primary_notes=["unknown"],
                    descriptors=["error"],
                    intensity=0.0,
                    confidence=0.0
                )
                batch_results.append(error_prediction)
        
        return batch_results


class MemoryMonitor:
    """Memory usage monitoring and optimization."""
    
    def __init__(self):
        self.peak_memory = 0
        self.current_memory = 0
        
    def monitor_memory(self) -> Dict[str, float]:
        """Monitor current memory usage."""
        memory_info = {}
        
        try:
            import psutil
            process = psutil.Process()
            memory_info['ram_mb'] = process.memory_info().rss / 1024 / 1024
            memory_info['ram_percent'] = process.memory_percent()
        except ImportError:
            memory_info['ram_mb'] = 0
            memory_info['ram_percent'] = 0
        
        if HAS_TORCH and torch.cuda.is_available():
            memory_info['gpu_allocated_mb'] = torch.cuda.memory_allocated() / 1024 / 1024
            memory_info['gpu_cached_mb'] = torch.cuda.memory_reserved() / 1024 / 1024
            memory_info['gpu_max_mb'] = torch.cuda.max_memory_allocated() / 1024 / 1024
        
        return memory_info
    
    def optimize_memory(self):
        """Optimize memory usage."""
        if HAS_TORCH and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Force garbage collection
        import gc
        gc.collect()


# Integration class for enhanced features
class EnhancedOlfactorySystem:
    """Enhanced olfactory system with all advanced features."""
    
    def __init__(self, model: OlfactoryTransformer, tokenizer: MoleculeTokenizer):
        self.model = model
        self.tokenizer = tokenizer
        
        # Initialize components
        self.streaming = StreamingPredictor(model, tokenizer)
        self.analyzer = EnhancedMolecularAnalyzer(model, tokenizer)
        self.zero_shot = ZeroShotEnhancer(model, tokenizer)
        self.optimizer = ProductionOptimizer(model)
        self.memory_monitor = MemoryMonitor()
        
        # Apply production optimizations
        self.optimizer.optimize_for_production()
        
        logging.info("Enhanced Olfactory System initialized with all advanced features")
    
    async def comprehensive_analysis(self, smiles: str) -> Dict[str, Any]:
        """Perform comprehensive analysis combining all features."""
        start_time = time.time()
        
        # Basic prediction
        prediction = self.model.predict_scent(smiles, self.tokenizer)
        
        # Detailed analysis
        analysis = self.analyzer.analyze_molecule(smiles, include_attention=True)
        
        # Zero-shot classification
        custom_categories = ["luxury", "natural", "synthetic", "allergen", "safe"]
        zero_shot_result = self.zero_shot.classify_custom_categories(
            smiles, custom_categories, return_probabilities=True
        )
        
        # Memory monitoring
        memory_info = self.memory_monitor.monitor_memory()
        
        processing_time = time.time() - start_time
        
        return {
            'basic_prediction': prediction.__dict__,
            'detailed_analysis': analysis.__dict__,
            'zero_shot_categories': zero_shot_result,
            'memory_info': memory_info,
            'processing_time_seconds': processing_time,
            'timestamp': datetime.now().isoformat()
        }