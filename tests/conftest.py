"""Pytest configuration and shared fixtures."""

import pytest
import torch
import logging
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock

# Configure test logging
logging.basicConfig(level=logging.WARNING)

# Set test environment variables
os.environ['TESTING'] = 'true'
os.environ['LOG_LEVEL'] = 'WARNING'


@pytest.fixture(scope="session")
def torch_device():
    """Get the best available torch device for testing."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


@pytest.fixture
def temp_dir():
    """Create temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def mock_rdkit():
    """Mock RDKit module for tests that don't require actual molecular computations."""
    mock_chem = Mock()
    mock_mol = Mock()
    mock_mol.GetNumAtoms.return_value = 3
    mock_mol.GetNumBonds.return_value = 2
    mock_chem.MolFromSmiles.return_value = mock_mol
    mock_chem.MolToSmiles.return_value = "CCO"
    
    mock_descriptors = Mock()
    mock_descriptors.MolWt.return_value = 46.07
    mock_descriptors.MolLogP.return_value = -0.31
    mock_descriptors.NumRotatableBonds.return_value = 0
    mock_descriptors.NumAromaticRings.return_value = 0
    
    return {
        'Chem': mock_chem,
        'Descriptors': mock_descriptors,
        'mol': mock_mol
    }


@pytest.fixture
def sample_smiles():
    """Sample SMILES strings for testing."""
    return [
        "CCO",                      # Ethanol
        "CC(C)O",                   # Isopropanol
        "C1=CC=CC=C1",             # Benzene
        "CC(=O)OCC",               # Ethyl acetate
        "C1=CC=C(C=C1)C=O",        # Benzaldehyde
        "COC1=CC(=CC=C1O)C=O",     # Vanillin
        "CC1=CC=C(C=C1)C=O",       # p-Tolualdehyde
        "C1=CC=C2C(=C1)C=CC=C2",   # Naphthalene
        "CC(C)(C)C1=CC=C(C=C1)O",  # 4-tert-Butylphenol
        "CCCCCCCCCCCCCCCCCO",       # 1-Octadecanol
    ]


@pytest.fixture
def sample_scent_data():
    """Sample scent evaluation data."""
    return [
        {
            "smiles": "CCO",
            "primary_notes": ["alcohol", "solvent"],
            "intensity": 3.5,
            "character": "neutral",
            "family": "alcohol"
        },
        {
            "smiles": "C1=CC=C(C=C1)C=O",
            "primary_notes": ["almond", "sweet"],
            "intensity": 7.2,
            "character": "pleasant",
            "family": "aldehyde"
        },
        {
            "smiles": "COC1=CC(=CC=C1O)C=O",
            "primary_notes": ["vanilla", "sweet", "creamy"],
            "intensity": 8.1,
            "character": "pleasant",
            "family": "phenolic_aldehyde"
        }
    ]


@pytest.fixture
def mock_sensor_reading():
    """Mock sensor reading for testing."""
    from olfactory_transformer.sensors.enose import SensorReading
    
    return SensorReading(
        timestamp=1234567890.0,
        gas_sensors={
            "TGS2600": 2.5,
            "TGS2602": 1.8,
            "TGS2610": 3.2,
            "TGS2620": 0.9,
            "TGS2611": 1.5,
        },
        temperature=25.0,
        humidity=60.0,
        pressure=1013.25,
        metadata={"location": "lab", "experiment": "test"}
    )


@pytest.fixture
def mock_human_panel_data():
    """Mock human panel evaluation data."""
    import pandas as pd
    import numpy as np
    
    np.random.seed(42)
    
    compounds = ["CCO", "C1=CC=C(C=C1)C=O", "COC1=CC(=CC=C1O)C=O"]
    panelists = list(range(1, 11))  # 10 panelists
    
    data = []
    for compound in compounds:
        for panelist in panelists:
            data.append({
                "compound": compound,
                "panelist": panelist,
                "intensity": np.random.beta(2, 2) * 10,
                "primary_notes": ",".join(np.random.choice(
                    ["floral", "woody", "citrus", "fresh", "sweet"], 
                    size=np.random.randint(1, 3), 
                    replace=False
                )),
                "character": np.random.choice(["pleasant", "neutral", "unpleasant"]),
                "overall_rating": np.random.beta(3, 2) * 10,
            })
    
    return pd.DataFrame(data)


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Setup test environment before each test."""
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Clear any cached models or data
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    yield
    
    # Cleanup after test
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


@pytest.fixture
def mock_config():
    """Mock configuration for testing."""
    from olfactory_transformer.core.config import OlfactoryConfig
    
    return OlfactoryConfig(
        vocab_size=100,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=128,
        max_position_embeddings=128,
        sensor_channels=16,
        molecular_feature_dim=32,
        dropout=0.1,
        layer_norm_eps=1e-5,
    )


# Performance test markers
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU"
    )
    config.addinivalue_line(
        "markers", "security: marks tests that test security features"
    )


@pytest.fixture
def benchmark_data():
    """Generate benchmark data for performance tests."""
    import numpy as np
    
    # Generate synthetic molecular data
    np.random.seed(42)
    
    smiles_chars = list("CHNOPS()=+-#[]@/\\")
    benchmark_smiles = []
    
    for _ in range(100):
        length = np.random.randint(5, 50)
        smiles = ''.join(np.random.choice(smiles_chars, length))
        benchmark_smiles.append(smiles)
    
    return benchmark_smiles


@pytest.fixture
def performance_threshold():
    """Performance thresholds for benchmark tests."""
    return {
        "tokenization_time_ms": 100,      # Max 100ms per SMILES
        "inference_time_ms": 500,         # Max 500ms per prediction
        "batch_inference_time_ms": 1000,  # Max 1s per batch
        "memory_usage_mb": 1000,          # Max 1GB memory usage
    }


# Skip tests based on available hardware/software
def pytest_collection_modifyitems(config, items):
    """Modify test collection based on available resources."""
    skip_gpu = pytest.mark.skip(reason="GPU not available")
    skip_slow = pytest.mark.skip(reason="Slow test skipped")
    
    for item in items:
        # Skip GPU tests if CUDA not available
        if "gpu" in item.keywords and not torch.cuda.is_available():
            item.add_marker(skip_gpu)
        
        # Skip slow tests if --runslow not given
        if "slow" in item.keywords and not config.getoption("--runslow"):
            item.add_marker(skip_slow)


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )
    parser.addoption(
        "--rungpu", action="store_true", default=False, help="run GPU tests"
    )


# Custom assertions for testing
class CustomAssertions:
    """Custom assertions for olfactory transformer testing."""
    
    @staticmethod
    def assert_valid_scent_prediction(prediction):
        """Assert that a scent prediction is valid."""
        from olfactory_transformer.core.config import ScentPrediction
        
        assert isinstance(prediction, ScentPrediction)
        assert isinstance(prediction.primary_notes, list)
        assert len(prediction.primary_notes) > 0
        assert all(isinstance(note, str) for note in prediction.primary_notes)
        assert 0 <= prediction.intensity <= 10
        assert 0 <= prediction.confidence <= 1
        assert isinstance(prediction.chemical_family, str)
    
    @staticmethod
    def assert_tensor_properties(tensor, expected_shape=None, device=None, dtype=None):
        """Assert tensor properties."""
        assert isinstance(tensor, torch.Tensor)
        
        if expected_shape:
            assert tensor.shape == expected_shape
        
        if device:
            assert tensor.device == device
        
        if dtype:
            assert tensor.dtype == dtype
        
        # Check for NaN/Inf values
        assert not torch.any(torch.isnan(tensor))
        assert not torch.any(torch.isinf(tensor))
    
    @staticmethod
    def assert_model_outputs(outputs, batch_size=None):
        """Assert model outputs are valid."""
        assert isinstance(outputs, dict)
        
        required_keys = ["scent_logits", "intensity", "chemical_family_logits"]
        for key in required_keys:
            assert key in outputs
            assert isinstance(outputs[key], torch.Tensor)
            
            if batch_size:
                assert outputs[key].shape[0] == batch_size


@pytest.fixture
def assert_helper():
    """Provide custom assertions as fixture."""
    return CustomAssertions