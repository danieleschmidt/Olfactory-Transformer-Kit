"""
Dependency Manager for Graceful Degradation.

Manages optional dependencies and provides graceful fallbacks:
- Dynamic import handling
- Feature detection and graceful degradation
- Mock implementations for missing dependencies
- Dependency health monitoring
"""

import logging
import warnings
from typing import Dict, List, Optional, Any, Callable, Type, Union
from dataclasses import dataclass
from functools import wraps
import importlib
import sys
from pathlib import Path


@dataclass
class DependencyInfo:
    """Information about a dependency."""
    name: str
    installed: bool
    version: Optional[str] = None
    import_path: str = None
    fallback_available: bool = False
    critical: bool = False
    features_enabled: List[str] = None


class DependencyManager:
    """Manages optional dependencies and graceful degradation."""
    
    def __init__(self):
        self.dependencies: Dict[str, DependencyInfo] = {}
        self.mock_implementations: Dict[str, Any] = {}
        self.feature_flags: Dict[str, bool] = {}
        self.fallback_warnings_shown: set = set()
        
        # Initialize core dependencies
        self._check_core_dependencies()
        self._setup_mock_implementations()
    
    def _check_core_dependencies(self):
        """Check status of core dependencies."""
        core_deps = {
            'numpy': {
                'import_path': 'numpy',
                'features': ['numerical_computation', 'array_operations'],
                'critical': False
            },
            'torch': {
                'import_path': 'torch',
                'features': ['deep_learning', 'gpu_acceleration'],
                'critical': False
            },
            'transformers': {
                'import_path': 'transformers',
                'features': ['pretrained_models', 'tokenizers'],
                'critical': False
            },
            'rdkit': {
                'import_path': 'rdkit.Chem',
                'features': ['molecular_analysis', 'chemical_properties'],
                'critical': False
            },
            'fastapi': {
                'import_path': 'fastapi',
                'features': ['api_server', 'async_endpoints'],
                'critical': False
            },
            'redis': {
                'import_path': 'redis',
                'features': ['caching', 'session_storage'],
                'critical': False
            },
            'scipy': {
                'import_path': 'scipy',
                'features': ['statistical_tests', 'signal_processing'],
                'critical': False
            },
            'pandas': {
                'import_path': 'pandas',
                'features': ['data_manipulation', 'csv_export'],
                'critical': False
            },
            'matplotlib': {
                'import_path': 'matplotlib.pyplot',
                'features': ['visualization', 'plotting'],
                'critical': False
            },
            'seaborn': {
                'import_path': 'seaborn',
                'features': ['statistical_plots', 'heatmaps'],
                'critical': False
            }
        }
        
        for dep_name, dep_config in core_deps.items():
            self.dependencies[dep_name] = self._check_dependency(
                dep_name, 
                dep_config['import_path'],
                dep_config.get('features', []),
                dep_config.get('critical', False)
            )
    
    def _check_dependency(self, name: str, import_path: str, 
                         features: List[str], critical: bool = False) -> DependencyInfo:
        """Check if a dependency is available."""
        try:
            module = importlib.import_module(import_path)
            version = getattr(module, '__version__', 'unknown')
            
            # Enable features for this dependency
            for feature in features:
                self.feature_flags[feature] = True
            
            logging.debug(f"✅ {name} v{version} available")
            
            return DependencyInfo(
                name=name,
                installed=True,
                version=version,
                import_path=import_path,
                fallback_available=name in self._get_fallback_modules(),
                critical=critical,
                features_enabled=features
            )
            
        except ImportError as e:
            logging.info(f"⚠️ {name} not available: {e}")
            
            # Check if fallback available
            has_fallback = name in self._get_fallback_modules()
            
            # Disable features for this dependency
            for feature in features:
                self.feature_flags[feature] = False
            
            if critical and not has_fallback:
                raise ImportError(f"Critical dependency {name} not available and no fallback exists")
            
            return DependencyInfo(
                name=name,
                installed=False,
                version=None,
                import_path=import_path,
                fallback_available=has_fallback,
                critical=critical,
                features_enabled=[]
            )
    
    def _get_fallback_modules(self) -> List[str]:
        """Get list of modules with fallback implementations."""
        return ['numpy', 'torch', 'redis', 'matplotlib']
    
    def _setup_mock_implementations(self):
        """Setup mock implementations for missing dependencies."""
        
        # Mock NumPy
        if not self.is_available('numpy'):
            self.mock_implementations['numpy'] = self._create_numpy_mock()
        
        # Mock PyTorch
        if not self.is_available('torch'):
            self.mock_implementations['torch'] = self._create_torch_mock()
        
        # Mock Redis
        if not self.is_available('redis'):
            self.mock_implementations['redis'] = self._create_redis_mock()
        
        # Mock Matplotlib
        if not self.is_available('matplotlib'):
            self.mock_implementations['matplotlib'] = self._create_matplotlib_mock()
    
    def _create_numpy_mock(self) -> Any:
        """Create mock NumPy implementation."""
        class MockNumPy:
            """Mock NumPy for basic array operations."""
            
            @staticmethod
            def array(data, dtype=None):
                if isinstance(data, (list, tuple)):
                    return list(data)
                return [data]
            
            @staticmethod
            def zeros(shape):
                if isinstance(shape, int):
                    return [0.0] * shape
                elif isinstance(shape, (list, tuple)):
                    size = 1
                    for dim in shape:
                        size *= dim
                    return [0.0] * size
                return []
            
            @staticmethod
            def ones(shape):
                if isinstance(shape, int):
                    return [1.0] * shape
                elif isinstance(shape, (list, tuple)):
                    size = 1
                    for dim in shape:
                        size *= dim
                    return [1.0] * size
                return []
            
            @staticmethod
            def mean(data, axis=None):
                if isinstance(data, (list, tuple)) and data:
                    return sum(data) / len(data)
                return 0.0
            
            @staticmethod
            def std(data, axis=None):
                if isinstance(data, (list, tuple)) and len(data) > 1:
                    mean_val = sum(data) / len(data)
                    variance = sum((x - mean_val) ** 2 for x in data) / len(data)
                    return variance ** 0.5
                return 0.0
            
            @staticmethod
            def max(data, axis=None):
                if isinstance(data, (list, tuple)) and data:
                    return max(data)
                return 0.0
            
            @staticmethod
            def min(data, axis=None):
                if isinstance(data, (list, tuple)) and data:
                    return min(data)
                return 0.0
            
            @staticmethod
            def sum(data, axis=None):
                if isinstance(data, (list, tuple)):
                    return sum(data)
                return data
            
            @staticmethod
            def dot(a, b):
                if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
                    if len(a) == len(b):
                        return sum(x * y for x, y in zip(a, b))
                return 0.0
            
            @staticmethod
            def random():
                class Random:
                    @staticmethod
                    def random(size=None):
                        import random
                        if size is None:
                            return random.random()
                        elif isinstance(size, int):
                            return [random.random() for _ in range(size)]
                        return random.random()
                    
                    @staticmethod
                    def normal(loc=0.0, scale=1.0, size=None):
                        import random
                        if size is None:
                            return random.gauss(loc, scale)
                        elif isinstance(size, int):
                            return [random.gauss(loc, scale) for _ in range(size)]
                        return random.gauss(loc, scale)
                
                return Random()
            
            inf = float('inf')
            nan = float('nan')
        
        return MockNumPy()
    
    def _create_torch_mock(self) -> Any:
        """Create mock PyTorch implementation."""
        class MockTorch:
            """Mock PyTorch for basic tensor operations."""
            
            class Tensor:
                def __init__(self, data, dtype=None):
                    if isinstance(data, (list, tuple)):
                        self.data = list(data)
                    else:
                        self.data = [data]
                    self.shape = (len(self.data),)
                
                def __getitem__(self, idx):
                    return self.data[idx]
                
                def __len__(self):
                    return len(self.data)
                
                def size(self, dim=None):
                    if dim is not None:
                        return self.shape[dim] if dim < len(self.shape) else 1
                    return self.shape
                
                def item(self):
                    return self.data[0] if self.data else 0.0
                
                def tolist(self):
                    return self.data
                
                def cuda(self):
                    return self  # No-op
                
                def cpu(self):
                    return self  # No-op
            
            @staticmethod
            def tensor(data, dtype=None):
                return MockTorch.Tensor(data, dtype)
            
            @staticmethod
            def zeros(shape, dtype=None):
                if isinstance(shape, int):
                    return MockTorch.Tensor([0.0] * shape, dtype)
                elif isinstance(shape, (list, tuple)):
                    size = 1
                    for dim in shape:
                        size *= dim
                    return MockTorch.Tensor([0.0] * size, dtype)
                return MockTorch.Tensor([0.0], dtype)
            
            @staticmethod
            def ones(shape, dtype=None):
                if isinstance(shape, int):
                    return MockTorch.Tensor([1.0] * shape, dtype)
                elif isinstance(shape, (list, tuple)):
                    size = 1
                    for dim in shape:
                        size *= dim
                    return MockTorch.Tensor([1.0] * size, dtype)
                return MockTorch.Tensor([1.0], dtype)
            
            @staticmethod
            def randn(*shape):
                import random
                if len(shape) == 1:
                    return MockTorch.Tensor([random.gauss(0, 1) for _ in range(shape[0])])
                else:
                    size = 1
                    for dim in shape:
                        size *= dim
                    return MockTorch.Tensor([random.gauss(0, 1) for _ in range(size)])
            
            @staticmethod
            def save(obj, path):
                # Mock save - would normally serialize
                logging.warning("Mock torch.save - not actually saving")
            
            @staticmethod
            def load(path, map_location=None):
                # Mock load - return empty dict
                logging.warning("Mock torch.load - returning empty dict")
                return {}
            
            class cuda:
                @staticmethod
                def is_available():
                    return False
                
                @staticmethod
                def empty_cache():
                    pass  # No-op
                
                @staticmethod
                def memory_allocated():
                    return 0
                
                @staticmethod
                def max_memory_allocated():
                    return 0
            
            class nn:
                class Module:
                    def __init__(self):
                        pass
                    
                    def eval(self):
                        return self
                    
                    def parameters(self):
                        return []
                
                class Linear(Module):
                    def __init__(self, in_features, out_features):
                        super().__init__()
                        self.in_features = in_features
                        self.out_features = out_features
                
                class Embedding(Module):
                    def __init__(self, num_embeddings, embedding_dim):
                        super().__init__()
                        self.num_embeddings = num_embeddings
                        self.embedding_dim = embedding_dim
        
        return MockTorch()
    
    def _create_redis_mock(self) -> Any:
        """Create mock Redis implementation."""
        class MockRedis:
            """Mock Redis for caching operations."""
            
            def __init__(self):
                self._cache = {}
            
            def get(self, key):
                return self._cache.get(key)
            
            def set(self, key, value, ex=None):
                self._cache[key] = value
                return True
            
            def delete(self, key):
                return self._cache.pop(key, None) is not None
            
            def exists(self, key):
                return key in self._cache
            
            def flushall(self):
                self._cache.clear()
                return True
            
            def ping(self):
                return True
        
        return MockRedis()
    
    def _create_matplotlib_mock(self) -> Any:
        """Create mock Matplotlib implementation."""
        class MockMatplotlib:
            """Mock Matplotlib for plotting operations."""
            
            @staticmethod
            def plot(*args, **kwargs):
                logging.info("Mock plot - visualization would be displayed here")
            
            @staticmethod
            def show():
                logging.info("Mock show - plot would be displayed here")
            
            @staticmethod
            def savefig(filename, **kwargs):
                logging.info(f"Mock savefig - plot would be saved to {filename}")
            
            @staticmethod
            def figure(**kwargs):
                return MockMatplotlib()
            
            @staticmethod
            def subplot(*args):
                return MockMatplotlib()
            
            @staticmethod
            def xlabel(label):
                pass
            
            @staticmethod
            def ylabel(label):
                pass
            
            @staticmethod
            def title(title):
                pass
            
            @staticmethod
            def legend():
                pass
        
        return MockMatplotlib()
    
    def is_available(self, dependency_name: str) -> bool:
        """Check if a dependency is available."""
        return self.dependencies.get(dependency_name, DependencyInfo(
            dependency_name, False
        )).installed
    
    def is_feature_enabled(self, feature_name: str) -> bool:
        """Check if a feature is enabled."""
        return self.feature_flags.get(feature_name, False)
    
    def get_dependency_info(self, dependency_name: str) -> Optional[DependencyInfo]:
        """Get information about a dependency."""
        return self.dependencies.get(dependency_name)
    
    def get_all_dependencies(self) -> Dict[str, DependencyInfo]:
        """Get information about all dependencies."""
        return self.dependencies.copy()
    
    def require_dependency(self, dependency_name: str, feature_description: str = "this feature"):
        """Decorator to require a dependency for a function."""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                if not self.is_available(dependency_name):
                    fallback_key = f"{dependency_name}_fallback_warning"
                    if fallback_key not in self.fallback_warnings_shown:
                        logging.warning(
                            f"{dependency_name} not available - {feature_description} will use fallback implementation"
                        )
                        self.fallback_warnings_shown.add(fallback_key)
                    
                    # Check if mock implementation available
                    if dependency_name in self.mock_implementations:
                        # Inject mock into globals for the function
                        original_globals = func.__globals__.copy()
                        func.__globals__[dependency_name] = self.mock_implementations[dependency_name]
                        
                        try:
                            result = func(*args, **kwargs)
                            return result
                        finally:
                            # Restore original globals
                            func.__globals__.clear()
                            func.__globals__.update(original_globals)
                    else:
                        raise ImportError(f"{dependency_name} required for {feature_description}")
                
                return func(*args, **kwargs)
            return wrapper
        return decorator
    
    def safe_import(self, module_name: str, fallback_name: Optional[str] = None):
        """Safely import a module with optional fallback."""
        try:
            return importlib.import_module(module_name)
        except ImportError:
            if fallback_name and fallback_name in self.mock_implementations:
                logging.warning(f"Using mock implementation for {module_name}")
                return self.mock_implementations[fallback_name]
            else:
                raise ImportError(f"Could not import {module_name} and no fallback available")
    
    def generate_dependency_report(self) -> str:
        """Generate a comprehensive dependency report."""
        report_lines = [
            "# Dependency Report",
            "",
            "## Core Dependencies",
            ""
        ]
        
        installed_count = 0
        total_count = len(self.dependencies)
        
        for name, info in self.dependencies.items():
            status_icon = "✅" if info.installed else "❌" if info.critical else "⚠️"
            version_str = f" (v{info.version})" if info.version else ""
            fallback_str = " [Fallback Available]" if info.fallback_available else ""
            
            report_lines.append(f"- **{name}**{version_str}: {status_icon}{fallback_str}")
            
            if info.features_enabled:
                report_lines.append(f"  - Features: {', '.join(info.features_enabled)}")
            
            if info.installed:
                installed_count += 1
        
        report_lines.extend([
            "",
            f"## Summary",
            f"- Installed: {installed_count}/{total_count} ({installed_count/total_count*100:.1f}%)",
            "",
            "## Feature Status",
            ""
        ])
        
        enabled_features = [feature for feature, enabled in self.feature_flags.items() if enabled]
        disabled_features = [feature for feature, enabled in self.feature_flags.items() if not enabled]
        
        if enabled_features:
            report_lines.append("### Enabled Features")
            for feature in enabled_features:
                report_lines.append(f"- ✅ {feature}")
        
        if disabled_features:
            report_lines.append("### Disabled Features")
            for feature in disabled_features:
                report_lines.append(f"- ❌ {feature}")
        
        report_lines.extend([
            "",
            "## Fallback Status",
            ""
        ])
        
        for name, info in self.dependencies.items():
            if not info.installed and info.fallback_available:
                report_lines.append(f"- {name}: Mock implementation active")
        
        return "\n".join(report_lines)
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on all dependencies."""
        health_status = {
            'overall_status': 'healthy',
            'dependencies': {},
            'features': self.feature_flags.copy(),
            'warnings': [],
            'errors': []
        }
        
        critical_missing = []
        warnings = []
        
        for name, info in self.dependencies.items():
            dep_status = {
                'installed': info.installed,
                'version': info.version,
                'features': info.features_enabled,
                'fallback_available': info.fallback_available
            }
            
            if not info.installed:
                if info.critical:
                    critical_missing.append(name)
                    dep_status['status'] = 'critical'
                elif info.fallback_available:
                    warnings.append(f"{name} using fallback implementation")
                    dep_status['status'] = 'degraded'
                else:
                    warnings.append(f"{name} not available - some features disabled")
                    dep_status['status'] = 'missing'
            else:
                dep_status['status'] = 'healthy'
            
            health_status['dependencies'][name] = dep_status
        
        # Overall status determination
        if critical_missing:
            health_status['overall_status'] = 'critical'
            health_status['errors'] = [f"Critical dependencies missing: {', '.join(critical_missing)}"]
        elif warnings:
            health_status['overall_status'] = 'degraded'
            health_status['warnings'] = warnings
        
        return health_status


# Global dependency manager instance
dependency_manager = DependencyManager()