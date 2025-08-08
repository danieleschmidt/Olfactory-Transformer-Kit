#!/usr/bin/env python3
"""Health check script for Docker container."""

import sys
import time
import logging
from pathlib import Path

try:
    import requests
    import torch
    from olfactory_transformer.core.model import OlfactoryTransformer
    from olfactory_transformer.core.config import OlfactoryConfig
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.WARNING)


def check_api_health(host="localhost", port=8000, timeout=5):
    """Check if API server is responding."""
    try:
        url = f"http://{host}:{port}/health"
        response = requests.get(url, timeout=timeout)
        return response.status_code == 200
    except Exception as e:
        print(f"API health check failed: {e}")
        return False


def check_model_load():
    """Check if model can be loaded."""
    try:
        config = OlfactoryConfig(
            vocab_size=100,
            hidden_size=64,
            num_hidden_layers=1,
            num_attention_heads=2
        )
        model = OlfactoryTransformer(config)
        
        # Test forward pass with dummy data
        dummy_input = torch.zeros((1, 10), dtype=torch.long)
        with torch.no_grad():
            outputs = model.forward(input_ids=dummy_input)
        
        return isinstance(outputs, dict) and len(outputs) > 0
    except Exception as e:
        print(f"Model health check failed: {e}")
        return False


def check_dependencies():
    """Check critical dependencies."""
    try:
        import numpy as np
        import torch
        import transformers
        return True
    except ImportError as e:
        print(f"Dependency check failed: {e}")
        return False


def check_disk_space(min_free_gb=1):
    """Check available disk space."""
    try:
        import shutil
        total, used, free = shutil.disk_usage("/app")
        free_gb = free // (1024**3)
        return free_gb >= min_free_gb
    except Exception as e:
        print(f"Disk space check failed: {e}")
        return False


def check_memory(min_free_mb=500):
    """Check available memory."""
    try:
        import psutil
        memory = psutil.virtual_memory()
        available_mb = memory.available // (1024**2)
        return available_mb >= min_free_mb
    except Exception as e:
        print(f"Memory check failed: {e}")
        return False


def main():
    """Run comprehensive health checks."""
    checks = [
        ("Dependencies", check_dependencies),
        ("Model Loading", check_model_load),
        ("Disk Space", lambda: check_disk_space(1)),
        ("Memory", lambda: check_memory(500)),
        ("API Health", check_api_health),
    ]
    
    failed_checks = []
    
    for check_name, check_func in checks:
        try:
            if not check_func():
                failed_checks.append(check_name)
                print(f"❌ {check_name} check failed")
            else:
                print(f"✅ {check_name} check passed")
        except Exception as e:
            failed_checks.append(check_name)
            print(f"❌ {check_name} check error: {e}")
    
    if failed_checks:
        print(f"Health check failed: {', '.join(failed_checks)}")
        sys.exit(1)
    else:
        print("All health checks passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()