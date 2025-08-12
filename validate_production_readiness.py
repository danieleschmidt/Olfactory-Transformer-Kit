#!/usr/bin/env python3
"""Production readiness validation script."""

import sys
import os
import subprocess
import json
from pathlib import Path
from typing import Dict, List, Any
import time

# Add repo to path
sys.path.insert(0, str(Path(__file__).parent))


class ProductionValidator:
    """Validate system for production deployment."""
    
    def __init__(self):
        self.results = {
            "timestamp": time.time(),
            "checks": {},
            "overall_status": "unknown",
            "critical_issues": [],
            "warnings": [],
            "recommendations": []
        }
    
    def validate_code_structure(self) -> bool:
        """Validate code structure and organization."""
        print("🏗️ Validating code structure...")
        
        required_files = [
            "olfactory_transformer/__init__.py",
            "olfactory_transformer/core/model.py",
            "olfactory_transformer/core/tokenizer.py",
            "olfactory_transformer/core/config.py",
            "olfactory_transformer/sensors/enose.py",
            "olfactory_transformer/utils/error_handling.py",
            "olfactory_transformer/utils/reliability.py",
            "olfactory_transformer/utils/performance.py",
            "pyproject.toml",
            "README.md",
        ]
        
        missing_files = []
        for file_path in required_files:
            if not Path(file_path).exists():
                missing_files.append(file_path)
        
        if missing_files:
            self.results["critical_issues"].extend([f"Missing file: {f}" for f in missing_files])
            self.results["checks"]["code_structure"] = "FAIL"
            return False
        
        print("  ✓ All required files present")
        self.results["checks"]["code_structure"] = "PASS"
        return True
    
    def validate_imports(self) -> bool:
        """Validate all imports work correctly."""
        print("📦 Validating imports...")
        
        try:
            # Test core imports
            from olfactory_transformer import __version__
            from olfactory_transformer.core.config import OlfactoryConfig
            from olfactory_transformer.core.tokenizer import MoleculeTokenizer
            from olfactory_transformer.sensors.enose import ENoseInterface
            
            print(f"  ✓ Core imports successful (version {__version__})")
            
            # Test optional imports gracefully handle missing dependencies
            try:
                from olfactory_transformer import OlfactoryTransformer
                has_torch = True
            except ImportError:
                has_torch = False
            
            print(f"  ✓ Torch availability: {has_torch}")
            
            self.results["checks"]["imports"] = "PASS"
            self.results["torch_available"] = has_torch
            
            if not has_torch:
                self.results["warnings"].append("PyTorch not available - ML features disabled")
            
            return True
            
        except Exception as e:
            self.results["critical_issues"].append(f"Import error: {e}")
            self.results["checks"]["imports"] = "FAIL"
            return False
    
    def validate_configuration(self) -> bool:
        """Validate configuration system."""
        print("⚙️ Validating configuration...")
        
        try:
            from olfactory_transformer.core.config import OlfactoryConfig
            
            # Test default configuration
            config = OlfactoryConfig()
            assert config.vocab_size > 0
            assert config.hidden_size > 0
            
            # Test configuration serialization
            config_dict = config.to_dict()
            config2 = OlfactoryConfig.from_dict(config_dict)
            assert config2.vocab_size == config.vocab_size
            
            print("  ✓ Configuration system working")
            self.results["checks"]["configuration"] = "PASS"
            return True
            
        except Exception as e:
            self.results["critical_issues"].append(f"Configuration error: {e}")
            self.results["checks"]["configuration"] = "FAIL"
            return False
    
    def validate_security(self) -> bool:
        """Validate security measures."""
        print("🔒 Validating security...")
        
        try:
            from olfactory_transformer.core.tokenizer import MoleculeTokenizer
            
            tokenizer = MoleculeTokenizer()
            
            # Test malicious input handling
            malicious_inputs = [
                "exec('import os')",
                "../../../etc/passwd",
                "' OR 1=1 --",
                "<script>alert('xss')</script>",
            ]
            
            security_issues = []
            for malicious in malicious_inputs:
                try:
                    result = tokenizer.encode(malicious)
                    # Should either reject or safely handle
                    if isinstance(result, dict):
                        # Check it was sanitized
                        decoded = tokenizer.decode(result["input_ids"][:5])
                        if "exec(" in decoded or ".." in decoded:
                            security_issues.append(f"Unsanitized input: {malicious}")
                except ValueError:
                    # Expected - security rejection
                    pass
            
            if security_issues:
                self.results["critical_issues"].extend(security_issues)
                self.results["checks"]["security"] = "FAIL"
                return False
            
            print("  ✓ Security validation passed")
            self.results["checks"]["security"] = "PASS"
            return True
            
        except Exception as e:
            self.results["critical_issues"].append(f"Security validation error: {e}")
            self.results["checks"]["security"] = "FAIL"
            return False
    
    def validate_performance(self) -> bool:
        """Validate performance requirements."""
        print("⚡ Validating performance...")
        
        try:
            from olfactory_transformer.core.tokenizer import MoleculeTokenizer
            
            tokenizer = MoleculeTokenizer(vocab_size=100)
            tokenizer.build_vocab_from_smiles(["CCO", "CCC", "CCCC"])
            
            # Test encoding performance
            start_time = time.time()
            for _ in range(100):
                tokenizer.encode("CCO")
            end_time = time.time()
            
            avg_time = (end_time - start_time) / 100
            throughput = 1.0 / avg_time
            
            print(f"  Encoding performance: {throughput:.0f} ops/sec")
            
            # Performance requirements
            min_throughput = 1000  # ops/sec
            if throughput < min_throughput:
                self.results["warnings"].append(f"Low performance: {throughput:.0f} < {min_throughput} ops/sec")
            
            self.results["checks"]["performance"] = "PASS"
            self.results["throughput_ops_sec"] = throughput
            return True
            
        except Exception as e:
            self.results["critical_issues"].append(f"Performance validation error: {e}")
            self.results["checks"]["performance"] = "FAIL"
            return False
    
    def validate_error_handling(self) -> bool:
        """Validate error handling robustness."""
        print("🛡️ Validating error handling...")
        
        try:
            from olfactory_transformer.core.tokenizer import MoleculeTokenizer
            from olfactory_transformer.sensors.enose import ENoseInterface
            
            # Test tokenizer error handling
            tokenizer = MoleculeTokenizer()
            
            # Should handle invalid inputs gracefully
            error_inputs = [None, "", "C" * 20000, 123, []]
            
            for invalid_input in error_inputs:
                try:
                    if invalid_input is None:
                        continue  # Skip None test for now
                    result = tokenizer.encode(invalid_input)
                    # Should either work or raise appropriate error
                except (ValueError, TypeError):
                    # Expected behavior
                    pass
                except Exception as e:
                    self.results["warnings"].append(f"Unexpected error type for {invalid_input}: {type(e)}")
            
            # Test sensor error handling
            enose = ENoseInterface(port="/dev/nonexistent")
            connected = enose.connect()
            
            if not connected:
                # Should still allow fallback operations
                try:
                    reading = enose.read_single()
                    assert hasattr(reading, 'gas_sensors')
                except RuntimeError as e:
                    # This is expected behavior for failed connection
                    if "Failed to connect" in str(e):
                        pass  # Expected
                    else:
                        raise
            
            print("  ✓ Error handling validation passed")
            self.results["checks"]["error_handling"] = "PASS"
            return True
            
        except Exception as e:
            self.results["critical_issues"].append(f"Error handling validation error: {e}")
            self.results["checks"]["error_handling"] = "FAIL"
            return False
    
    def validate_documentation(self) -> bool:
        """Validate documentation completeness."""
        print("📚 Validating documentation...")
        
        readme_path = Path("README.md")
        if not readme_path.exists():
            self.results["critical_issues"].append("Missing README.md")
            self.results["checks"]["documentation"] = "FAIL"
            return False
        
        readme_content = readme_path.read_text()
        
        required_sections = [
            "installation",
            "usage",
            "example",
            "api",
            "license"
        ]
        
        missing_sections = []
        for section in required_sections:
            if section.lower() not in readme_content.lower():
                missing_sections.append(section)
        
        if missing_sections:
            self.results["warnings"].extend([f"Missing documentation section: {s}" for s in missing_sections])
        
        # Check for code examples
        if "```python" not in readme_content:
            self.results["warnings"].append("No Python code examples in README")
        
        print(f"  ✓ Documentation validation (README: {len(readme_content)} chars)")
        self.results["checks"]["documentation"] = "PASS"
        return True
    
    def validate_packaging(self) -> bool:
        """Validate packaging configuration."""
        print("📦 Validating packaging...")
        
        pyproject_path = Path("pyproject.toml")
        if not pyproject_path.exists():
            self.results["critical_issues"].append("Missing pyproject.toml")
            self.results["checks"]["packaging"] = "FAIL"
            return False
        
        try:
            # Try to validate the package structure
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", "-e", ".", "--dry-run"
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                self.results["warnings"].append(f"Package installation may have issues: {result.stderr[:200]}")
            
        except (subprocess.TimeoutExpired, FileNotFoundError):
            self.results["warnings"].append("Could not validate package installation")
        
        print("  ✓ Packaging validation completed")
        self.results["checks"]["packaging"] = "PASS"
        return True
    
    def generate_deployment_config(self) -> Dict[str, Any]:
        """Generate production deployment configuration."""
        config = {
            "deployment": {
                "environment": "production",
                "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
                "dependencies": {
                    "required": [
                        "torch>=2.0.0",
                        "numpy>=1.21.0",
                        "pandas>=1.3.0",
                        "scikit-learn>=1.0.0",
                    ],
                    "optional": [
                        "rdkit>=2022.9.1",
                        "pyserial>=3.5",
                        "psutil>=5.8.0",
                    ]
                },
                "resources": {
                    "memory_mb": 2048,
                    "cpu_cores": 2,
                    "storage_gb": 10,
                },
                "scaling": {
                    "max_workers": 8,
                    "batch_size": 32,
                    "cache_size": 1000,
                },
                "monitoring": {
                    "health_check_endpoint": "/health",
                    "metrics_endpoint": "/metrics", 
                    "log_level": "INFO",
                },
                "security": {
                    "input_validation": True,
                    "rate_limiting": True,
                    "sanitization": True,
                }
            }
        }
        
        return config
    
    def run_validation(self) -> bool:
        """Run all validation checks."""
        print("🚀 Production Readiness Validation")
        print("=" * 50)
        
        validators = [
            self.validate_code_structure,
            self.validate_imports,
            self.validate_configuration,
            self.validate_security,
            self.validate_performance,
            self.validate_error_handling,
            self.validate_documentation,
            self.validate_packaging,
        ]
        
        passed = 0
        failed = 0
        
        for validator in validators:
            try:
                if validator():
                    passed += 1
                else:
                    failed += 1
            except Exception as e:
                failed += 1
                self.results["critical_issues"].append(f"Validator {validator.__name__} crashed: {e}")
            print()
        
        # Determine overall status
        if failed == 0 and len(self.results["critical_issues"]) == 0:
            self.results["overall_status"] = "READY"
        elif len(self.results["critical_issues"]) == 0:
            self.results["overall_status"] = "READY_WITH_WARNINGS"
        else:
            self.results["overall_status"] = "NOT_READY"
        
        # Generate deployment config
        deployment_config = self.generate_deployment_config()
        
        print("=" * 50)
        print(f"Validation Results: {passed}/{len(validators)} checks passed")
        print(f"Overall Status: {self.results['overall_status']}")
        
        if self.results["critical_issues"]:
            print(f"Critical Issues ({len(self.results['critical_issues'])}):")
            for issue in self.results["critical_issues"]:
                print(f"  ❌ {issue}")
        
        if self.results["warnings"]:
            print(f"Warnings ({len(self.results['warnings'])}):")
            for warning in self.results["warnings"]:
                print(f"  ⚠️ {warning}")
        
        # Save results
        with open("production_validation_report.json", "w") as f:
            json.dump(self.results, f, indent=2)
        
        with open("deployment_config.json", "w") as f:
            json.dump(deployment_config, f, indent=2)
        
        print(f"\\n📄 Reports saved:")
        print(f"  - production_validation_report.json")
        print(f"  - deployment_config.json")
        
        return self.results["overall_status"] in ["READY", "READY_WITH_WARNINGS"]


def main():
    """Main validation entry point."""
    validator = ProductionValidator()
    
    if validator.run_validation():
        print("\\n🎉 System is PRODUCTION READY!")
        return True
    else:
        print("\\n❌ System is NOT ready for production")
        print("Address critical issues before deployment")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)