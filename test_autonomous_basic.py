"""
Autonomous Basic Functionality Tests.

Tests core functionality with graceful dependency handling.
"""

import pytest
from pathlib import Path
import sys

# Add project root
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from olfactory_transformer.utils.dependency_isolation import (
    dependency_manager,
    get_torch, 
    get_transformers,
    get_rdkit,
    get_numpy,
    print_system_status
)

class TestAutonomousCore:
    """Test autonomous core functionality."""
    
    def test_dependency_manager_health(self):
        """Test dependency manager provides health status."""
        health = dependency_manager.check_system_health()
        
        assert 'total_dependencies' in health
        assert 'available' in health
        assert 'missing' in health
        assert 'performance_impact' in health
        assert 'recommendations' in health
        
        print(f"System Health: {health}")
        
    def test_numpy_availability(self):
        """Test numpy is available (critical dependency)."""
        np = get_numpy()
        assert np is not None
        
        # Test basic numpy functionality
        arr = np.array([1, 2, 3, 4, 5])
        assert arr.shape == (5,)
        assert np.sum(arr) == 15
        
    def test_torch_graceful_degradation(self):
        """Test torch works with real or mock implementation."""
        torch = get_torch()
        assert torch is not None
        
        # Test basic torch-like functionality
        tensor_data = [1.0, 2.0, 3.0]
        t = torch.tensor(tensor_data)
        assert t is not None
        
        # Test zeros creation
        zeros = torch.zeros(2, 3)
        assert zeros is not None
        
    def test_transformers_graceful_degradation(self):
        """Test transformers works with real or mock implementation."""
        transformers = get_transformers()
        assert transformers is not None
        
        # Test model creation
        model = transformers.PreTrainedModel()
        assert model is not None
        
    def test_rdkit_graceful_degradation(self):
        """Test rdkit works with real or mock implementation."""
        rdkit = get_rdkit()
        assert rdkit is not None
        
        # Test molecule creation
        mol = rdkit.Chem.MolFromSmiles("CCO")  # Ethanol
        assert mol is not None
        
        # Test SMILES conversion
        smiles = rdkit.Chem.MolToSmiles(mol)
        assert isinstance(smiles, str)
        assert len(smiles) > 0

class TestBasicOlfactoryFunctionality:
    """Test basic olfactory functionality."""
    
    def test_basic_imports(self):
        """Test core olfactory imports work."""
        from olfactory_transformer import OlfactoryTransformer
        from olfactory_transformer.core.tokenizer import MoleculeTokenizer
        from olfactory_transformer.sensors.enose import ENoseInterface
        
        # Test classes are importable
        assert OlfactoryTransformer is not None
        assert MoleculeTokenizer is not None
        assert ENoseInterface is not None
        
    def test_tokenizer_basic(self):
        """Test molecule tokenizer basic functionality."""
        from olfactory_transformer.core.tokenizer import MoleculeTokenizer
        
        tokenizer = MoleculeTokenizer()
        assert tokenizer is not None
        
        # Test basic tokenization
        smiles = "CCO"  # Ethanol
        tokens = tokenizer.tokenize(smiles)
        assert tokens is not None
        assert isinstance(tokens, list)
        
    def test_config_loading(self):
        """Test configuration loading."""
        from olfactory_transformer.core.config import OlfactoryConfig
        
        config = OlfactoryConfig()
        assert config is not None
        assert hasattr(config, 'model_name')
        assert hasattr(config, 'vocab_size')

class TestAutonomousCapabilities:
    """Test autonomous SDLC capabilities."""
    
    def test_system_status_reporting(self):
        """Test system can report its own status."""
        from olfactory_transformer.utils.dependency_isolation import print_system_status
        
        # Should not raise any exceptions
        print_system_status()
        
    def test_performance_impact_calculation(self):
        """Test performance impact is calculated correctly."""
        health = dependency_manager.check_system_health()
        impact = health['performance_impact']
        
        assert isinstance(impact, float)
        assert 0.0 <= impact <= 1.0
        
        print(f"Performance impact: {impact:.1%}")
        
    def test_recommendations_provided(self):
        """Test system provides helpful recommendations."""
        health = dependency_manager.check_system_health()
        recommendations = health['recommendations']
        
        assert isinstance(recommendations, list)
        # Recommendations might be empty if all deps are available
        print(f"Recommendations: {recommendations}")

if __name__ == "__main__":
    print("🧪 Running Autonomous Basic Tests")
    print("=" * 50)
    
    # Print system status first
    print_system_status()
    print()
    
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])