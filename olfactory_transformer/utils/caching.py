"""Advanced caching systems for performance optimization and scaling."""

from typing import Dict, Any, Optional, Union, Tuple, List, Callable
import hashlib
import pickle
import time
import logging
import threading
from pathlib import Path
from dataclasses import dataclass
from collections import OrderedDict, defaultdict
from abc import ABC, abstractmethod

try:
    import redis
    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False

try:
    import lz4.frame
    HAS_LZ4 = True
except ImportError:
    HAS_LZ4 = False
import threading
import json

import numpy as np

from ..core.config import ScentPrediction


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    data: Any
    timestamp: float
    access_count: int = 0
    last_accessed: float = 0.0
    size_bytes: int = 0
    
    def __post_init__(self):
        self.last_accessed = self.timestamp
        if hasattr(self.data, '__sizeof__'):
            self.size_bytes = self.data.__sizeof__()
        else:
            self.size_bytes = len(pickle.dumps(self.data))


class LRUCache:
    """Thread-safe LRU cache with size limits."""
    
    def __init__(self, max_size: int = 1000, max_memory_mb: int = 100):
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.lock = threading.RLock()
        self.current_memory = 0
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        with self.lock:
            if key in self.cache:
                # Move to end (most recently used)
                entry = self.cache.pop(key)
                entry.access_count += 1
                entry.last_accessed = time.time()
                self.cache[key] = entry
                self.hits += 1
                return entry.data
            else:
                self.misses += 1
                return None
    
    def put(self, key: str, value: Any) -> None:
        """Put item in cache."""
        with self.lock:
            # Remove existing entry if present
            if key in self.cache:
                old_entry = self.cache.pop(key)
                self.current_memory -= old_entry.size_bytes
            
            # Create new entry
            entry = CacheEntry(data=value, timestamp=time.time())
            
            # Check if we need to evict
            while (len(self.cache) >= self.max_size or 
                   self.current_memory + entry.size_bytes > self.max_memory_bytes):
                if not self.cache:
                    break
                self._evict_lru()
            
            # Add new entry
            self.cache[key] = entry
            self.current_memory += entry.size_bytes
    
    def _evict_lru(self) -> None:
        """Evict least recently used item."""
        if self.cache:
            key, entry = self.cache.popitem(last=False)  # Remove oldest
            self.current_memory -= entry.size_bytes
            self.evictions += 1
    
    def clear(self) -> None:
        """Clear cache."""
        with self.lock:
            self.cache.clear()
            self.current_memory = 0
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            hit_rate = self.hits / max(1, self.hits + self.misses)
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "memory_mb": self.current_memory / (1024 * 1024),
                "max_memory_mb": self.max_memory_bytes / (1024 * 1024),
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": hit_rate,
                "evictions": self.evictions,
            }


class ModelCache:
    """Caching system for model components and intermediate results."""
    
    def __init__(
        self, 
        cache_dir: Optional[Path] = None,
        memory_cache_size: int = 1000,
        disk_cache_size_gb: int = 1
    ):
        self.cache_dir = cache_dir or Path.home() / ".olfactory_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Memory cache for fast access
        self.memory_cache = LRUCache(
            max_size=memory_cache_size,
            max_memory_mb=100
        )
        
        # Disk cache settings
        self.disk_cache_size_bytes = disk_cache_size_gb * 1024 * 1024 * 1024
        
        # Cache organization
        self.embeddings_cache = self.cache_dir / "embeddings"
        self.predictions_cache = self.cache_dir / "predictions"
        self.molecular_features_cache = self.cache_dir / "molecular_features"
        
        for cache_path in [self.embeddings_cache, self.predictions_cache, self.molecular_features_cache]:
            cache_path.mkdir(exist_ok=True)
        
        # Metadata tracking
        self.metadata_file = self.cache_dir / "metadata.json"
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load cache metadata."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logging.warning(f"Failed to load cache metadata: {e}")
        
        return {"created": time.time(), "entries": {}}
    
    def _save_metadata(self) -> None:
        """Save cache metadata."""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            logging.warning(f"Failed to save cache metadata: {e}")
    
    def _get_cache_key(self, identifier: str, cache_type: str = "general") -> str:
        """Generate cache key."""
        key_data = f"{cache_type}:{identifier}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def cache_molecular_features(self, smiles: str, features: Dict[str, float]) -> None:
        """Cache molecular features."""
        key = self._get_cache_key(smiles, "molecular_features")
        
        # Memory cache
        self.memory_cache.put(f"mol_feat_{key}", features)
        
        # Disk cache
        cache_file = self.molecular_features_cache / f"{key}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(features, f)
            
            # Update metadata
            self.metadata["entries"][key] = {
                "type": "molecular_features",
                "smiles": smiles,
                "created": time.time(),
                "size": cache_file.stat().st_size
            }
            self._save_metadata()
            
        except Exception as e:
            logging.warning(f"Failed to cache molecular features: {e}")
    
    def get_molecular_features(self, smiles: str) -> Optional[Dict[str, float]]:
        """Get cached molecular features."""
        key = self._get_cache_key(smiles, "molecular_features")
        
        # Check memory cache first
        features = self.memory_cache.get(f"mol_feat_{key}")
        if features is not None:
            return features
        
        # Check disk cache
        cache_file = self.molecular_features_cache / f"{key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    features = pickle.load(f)
                
                # Add back to memory cache
                self.memory_cache.put(f"mol_feat_{key}", features)
                return features
                
            except Exception as e:
                logging.warning(f"Failed to load cached molecular features: {e}")
        
        return None
    
    def cache_prediction(self, smiles: str, prediction: ScentPrediction) -> None:
        """Cache scent prediction."""
        key = self._get_cache_key(smiles, "prediction")
        
        # Memory cache
        self.memory_cache.put(f"pred_{key}", prediction)
        
        # Disk cache
        cache_file = self.predictions_cache / f"{key}.json"
        try:
            with open(cache_file, 'w') as f:
                json.dump(prediction.to_dict(), f)
            
            # Update metadata
            self.metadata["entries"][key] = {
                "type": "prediction",
                "smiles": smiles,
                "created": time.time(),
                "size": cache_file.stat().st_size
            }
            self._save_metadata()
            
        except Exception as e:
            logging.warning(f"Failed to cache prediction: {e}")
    
    def get_prediction(self, smiles: str) -> Optional[ScentPrediction]:
        """Get cached prediction."""
        key = self._get_cache_key(smiles, "prediction")
        
        # Check memory cache first
        prediction = self.memory_cache.get(f"pred_{key}")
        if prediction is not None:
            return prediction
        
        # Check disk cache
        cache_file = self.predictions_cache / f"{key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    pred_dict = json.load(f)
                
                # Convert back to ScentPrediction
                prediction = ScentPrediction(**pred_dict)
                
                # Add back to memory cache
                self.memory_cache.put(f"pred_{key}", prediction)
                return prediction
                
            except Exception as e:
                logging.warning(f"Failed to load cached prediction: {e}")
        
        return None
    
    def cache_embeddings(self, identifier: str, embeddings: np.ndarray) -> None:
        """Cache molecular embeddings."""
        key = self._get_cache_key(identifier, "embeddings")
        
        # Memory cache
        self.memory_cache.put(f"emb_{key}", embeddings)
        
        # Disk cache
        cache_file = self.embeddings_cache / f"{key}.npy"
        try:
            np.save(cache_file, embeddings)
            
            # Update metadata
            self.metadata["entries"][key] = {
                "type": "embeddings",
                "identifier": identifier,
                "created": time.time(),
                "size": cache_file.stat().st_size,
                "shape": embeddings.shape
            }
            self._save_metadata()
            
        except Exception as e:
            logging.warning(f"Failed to cache embeddings: {e}")
    
    def get_embeddings(self, identifier: str) -> Optional[np.ndarray]:
        """Get cached embeddings."""
        key = self._get_cache_key(identifier, "embeddings")
        
        # Check memory cache first
        embeddings = self.memory_cache.get(f"emb_{key}")
        if embeddings is not None:
            return embeddings
        
        # Check disk cache
        cache_file = self.embeddings_cache / f"{key}.npy"
        if cache_file.exists():
            try:
                embeddings = np.load(cache_file)
                
                # Add back to memory cache
                self.memory_cache.put(f"emb_{key}", embeddings)
                return embeddings
                
            except Exception as e:
                logging.warning(f"Failed to load cached embeddings: {e}")
        
        return None
    
    def cleanup_old_cache(self, max_age_days: int = 30) -> None:
        """Clean up old cache entries."""
        current_time = time.time()
        max_age_seconds = max_age_days * 24 * 3600
        
        entries_to_remove = []
        total_size_freed = 0
        
        for key, entry_info in self.metadata["entries"].items():
            if current_time - entry_info["created"] > max_age_seconds:
                entries_to_remove.append(key)
                
                # Remove file
                cache_type = entry_info["type"]
                if cache_type == "molecular_features":
                    cache_file = self.molecular_features_cache / f"{key}.pkl"
                elif cache_type == "prediction":
                    cache_file = self.predictions_cache / f"{key}.json"
                elif cache_type == "embeddings":
                    cache_file = self.embeddings_cache / f"{key}.npy"
                else:
                    continue
                
                if cache_file.exists():
                    total_size_freed += cache_file.stat().st_size
                    cache_file.unlink()
        
        # Update metadata
        for key in entries_to_remove:
            del self.metadata["entries"][key]
        
        self._save_metadata()
        
        logging.info(f"Cleaned up {len(entries_to_remove)} cache entries, "
                    f"freed {total_size_freed / (1024*1024):.1f} MB")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        # Memory cache stats
        memory_stats = self.memory_cache.stats()
        
        # Disk cache stats
        total_entries = len(self.metadata["entries"])
        total_size = sum(entry.get("size", 0) for entry in self.metadata["entries"].values())
        
        # Type breakdown
        type_counts = {}
        for entry in self.metadata["entries"].values():
            entry_type = entry.get("type", "unknown")
            type_counts[entry_type] = type_counts.get(entry_type, 0) + 1
        
        return {
            "memory_cache": memory_stats,
            "disk_cache": {
                "entries": total_entries,
                "size_mb": total_size / (1024 * 1024),
                "type_breakdown": type_counts
            },
            "cache_dir": str(self.cache_dir),
            "created": self.metadata.get("created", 0)
        }


class PredictionCache:
    """Specialized cache for prediction results with batch processing."""
    
    def __init__(self, model_cache: Optional[ModelCache] = None):
        self.model_cache = model_cache or ModelCache()
        
        # Batch prediction cache
        self.batch_cache = {}
        self.batch_lock = threading.Lock()
        
        # Prediction patterns for optimization
        self.prediction_patterns = {}
        self.pattern_lock = threading.Lock()
    
    def get_or_predict(
        self, 
        smiles_list: List[str], 
        predict_fn: callable,
        batch_size: int = 32
    ) -> List[ScentPrediction]:
        """Get predictions from cache or compute in batches."""
        results = []
        to_compute = []
        to_compute_indices = []
        
        # Check cache for each SMILES
        for i, smiles in enumerate(smiles_list):
            cached_prediction = self.model_cache.get_prediction(smiles)
            if cached_prediction:
                results.append((i, cached_prediction))
            else:
                to_compute.append(smiles)
                to_compute_indices.append(i)
        
        # Compute missing predictions in batches
        if to_compute:
            computed_predictions = []
            
            for i in range(0, len(to_compute), batch_size):
                batch = to_compute[i:i+batch_size]
                batch_predictions = []
                
                for smiles in batch:
                    prediction = predict_fn(smiles)
                    batch_predictions.append(prediction)
                    
                    # Cache the result
                    self.model_cache.cache_prediction(smiles, prediction)
                
                computed_predictions.extend(batch_predictions)
            
            # Add computed results
            for i, prediction in enumerate(computed_predictions):
                results.append((to_compute_indices[i], prediction))
        
        # Sort by original order and return predictions only
        results.sort(key=lambda x: x[0])
        return [pred for _, pred in results]
    
    def analyze_prediction_patterns(self, smiles_list: List[str]) -> Dict[str, Any]:
        """Analyze prediction patterns for cache optimization."""
        with self.pattern_lock:
            patterns = {
                "molecular_weight_ranges": {},
                "functional_groups": {},
                "ring_systems": {},
                "common_substructures": {},
            }
            
            # Analyze patterns (simplified implementation)
            for smiles in smiles_list:
                # Molecular weight pattern
                mw_range = f"{len(smiles)//10*10}-{(len(smiles)//10+1)*10}"  # Rough estimate
                patterns["molecular_weight_ranges"][mw_range] = \
                    patterns["molecular_weight_ranges"].get(mw_range, 0) + 1
                
                # Functional group patterns
                if "C=O" in smiles:
                    patterns["functional_groups"]["carbonyl"] = \
                        patterns["functional_groups"].get("carbonyl", 0) + 1
                if "OH" in smiles:
                    patterns["functional_groups"]["alcohol"] = \
                        patterns["functional_groups"].get("alcohol", 0) + 1
            
            return patterns
    
    def precompute_similar_molecules(
        self, 
        target_smiles: str, 
        candidate_smiles: List[str],
        predict_fn: callable,
        similarity_threshold: float = 0.8
    ) -> None:
        """Precompute predictions for similar molecules."""
        # Simple similarity check (would use proper molecular similarity in production)
        similar_molecules = []
        
        for smiles in candidate_smiles:
            # Simplified similarity: length and character overlap
            if len(smiles) == len(target_smiles):
                overlap = sum(1 for a, b in zip(smiles, target_smiles) if a == b)
                similarity = overlap / len(smiles)
                
                if similarity >= similarity_threshold:
                    similar_molecules.append(smiles)
        
        # Precompute predictions for similar molecules
        if similar_molecules:
            logging.info(f"Precomputing {len(similar_molecules)} similar molecules")
            self.get_or_predict(similar_molecules, predict_fn)
    
    def get_cache_recommendations(self) -> List[str]:
        """Get recommendations for cache optimization."""
        recommendations = []
        
        stats = self.model_cache.get_cache_stats()
        memory_hit_rate = stats["memory_cache"]["hit_rate"]
        
        if memory_hit_rate < 0.5:
            recommendations.append("Consider increasing memory cache size")
        
        if stats["disk_cache"]["entries"] > 10000:
            recommendations.append("Consider running cache cleanup")
        
        if stats["disk_cache"]["size_mb"] > 500:
            recommendations.append("Disk cache is large - consider archiving old entries")
        
        return recommendations