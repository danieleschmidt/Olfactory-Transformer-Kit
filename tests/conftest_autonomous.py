"""
Autonomous Test Configuration with Dependency Isolation.

Provides robust testing infrastructure that works with or without optional dependencies.
"""

import pytest
import warnings
from pathlib import Path
import sys
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import our dependency isolation system
from olfactory_transformer.utils.dependency_isolation import (
    dependency_manager, 
    get_torch, 
    get_transformers, 
    get_rdkit,
    get_numpy,
    print_system_status
)

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Print system status at test startup
print_system_status()

@pytest.fixture(scope="session")
def torch_available():
    """Check if PyTorch is available."""
    try:
        torch = get_torch()
        return hasattr(torch, 'tensor')
    except ImportError:
        return False

@pytest.fixture(scope="session") 
def transformers_available():
    """Check if Transformers is available."""
    try:
        transformers = get_transformers()
        return hasattr(transformers, 'PreTrainedModel')
    except ImportError:
        return False

@pytest.fixture(scope="session")
def rdkit_available():
    """Check if RDKit is available."""
    try:
        rdkit = get_rdkit()
        return hasattr(rdkit, 'Chem')
    except ImportError:
        return False

@pytest.fixture(scope="session")
def numpy_available():
    """Numpy should always be available (critical dependency)."""
    try:
        np = get_numpy()
        return True
    except ImportError:
        pytest.fail("NumPy is required but not available")

@pytest.fixture
def sample_molecules():
    """Sample molecular data for testing."""
    return [
        "CC(C)CC1=CC=C(C=C1)C(C)C",  # Lily of the valley
        "CCOC(=O)C1=CC=CC=C1",       # Ethyl benzoate  
        "CC1=CC=C(C=C1)C(C)(C)C",    # tert-Butylbenzene
        "CCO",                        # Ethanol
        "CC(=O)OC1=CC=CC=C1C(=O)O"  # Aspirin
    ]

@pytest.fixture
def sample_scent_descriptions():
    """Sample scent descriptions for testing."""
    return [
        ["floral", "fresh", "sweet"],
        ["fruity", "sweet", "wintergreen"],
        ["woody", "aromatic"],
        ["alcoholic", "sharp"],
        ["medicinal", "bitter"]
    ]

@pytest.fixture  
def mock_sensor_data():
    """Mock sensor array data."""
    np = get_numpy()
    return {
        'timestamp': np.arange(0, 10, 0.1),
        'sensors': {
            'TGS2600': np.random.randn(100) + 2.5,
            'TGS2602': np.random.randn(100) + 1.8,
            'TGS2610': np.random.randn(100) + 3.2,
            'TGS2620': np.random.randn(100) + 2.1
        },
        'temperature': 23.5,
        'humidity': 45.2,
        'pressure': 1013.25
    }

@pytest.fixture
def config_basic():
    """Basic configuration for testing."""
    return {
        'model_name': 'olfactory-base-test',
        'vocab_size': 1000,
        'hidden_size': 256,
        'num_layers': 4,
        'num_attention_heads': 8,
        'max_position_embeddings': 512,
        'dropout': 0.1
    }

class TestEnvironment:
    """Test environment manager with dependency awareness."""
    
    def __init__(self):
        self.health = dependency_manager.check_system_health()
        
    def skip_if_missing(self, dependency: str):
        """Decorator to skip tests if dependency is missing."""
        def decorator(test_func):
            def wrapper(*args, **kwargs):
                if dependency not in dependency_manager._dependencies:
                    pytest.skip(f"Dependency {dependency} not registered")
                if not dependency_manager._dependencies[dependency].available:
                    pytest.skip(f"Dependency {dependency} not available")
                return test_func(*args, **kwargs)
            return wrapper
        return decorator
        
    def require_mock_ok(self, dependency: str):
        """Decorator for tests that work with mocks.""" 
        def decorator(test_func):
            def wrapper(*args, **kwargs):
                if dependency not in dependency_manager._dependencies:
                    pytest.skip(f"Dependency {dependency} not registered")
                # Test runs whether real or mock dependency is available
                return test_func(*args, **kwargs)
            return wrapper
        return decorator

# Global test environment
test_env = TestEnvironment()

# Suppress warnings for cleaner test output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)