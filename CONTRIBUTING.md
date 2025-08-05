# Contributing to Olfactory Transformer

Welcome to the Olfactory Transformer project! We're excited to have you contribute to the first open-source foundation model for computational olfaction.

## 🌟 Ways to Contribute

### Priority Areas
- **Additional sensor array integrations** - Support for new gas sensors, environmental sensors
- **Fragrance industry datasets** - Curated datasets with proper licensing
- **Multi-lingual scent descriptions** - Translations and cultural scent perception data
- **Safety and toxicity prediction** - Integration with toxicology databases
- **Mobile and edge deployment** - Optimization for resource-constrained devices
- **Real-world validation studies** - Collaboration with perfumers and flavor houses

### Types of Contributions
- 🐛 **Bug Reports** - Found an issue? Let us know!
- 🚀 **Feature Requests** - Ideas for new functionality
- 📝 **Documentation** - Improve guides, examples, and API docs
- 🧪 **Code Contributions** - New features, bug fixes, optimizations
- 📊 **Datasets** - High-quality molecular-scent paired data
- 🏭 **Industry Integration** - Real-world deployment examples
- 🌍 **Translations** - Multi-language support for scent descriptors

## 🚀 Getting Started

### Development Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/danieleschmidt/quantum-inspired-task-planner.git
   cd quantum-inspired-task-planner
   ```

2. **Set Up Development Environment**
   ```bash
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install development dependencies
   pip install -e ".[dev,sensors]"
   
   # Install pre-commit hooks
   pre-commit install
   ```

3. **Verify Installation**
   ```bash
   # Run tests
   pytest tests/ -v
   
   # Run basic example
   python examples/basic_usage.py
   ```

### Development Dependencies
```bash
# Core dependencies
pip install torch>=2.0.0 transformers>=4.30.0 numpy pandas scikit-learn

# Development tools
pip install pytest>=7.0.0 pytest-cov black isort flake8 mypy pre-commit

# Optional sensor support
pip install pyserial smbus2 adafruit-circuitpython-bme680

# Optional optimization
pip install onnx onnxruntime tensorrt pynvml
```

## 📋 Development Workflow

### 1. Issue First
- **Check existing issues** before creating new ones
- **Use issue templates** for bug reports and feature requests
- **Discuss major changes** in issues before implementing

### 2. Branch Naming
- `feature/descriptive-name` - New features
- `fix/descriptive-name` - Bug fixes
- `docs/descriptive-name` - Documentation updates
- `perf/descriptive-name` - Performance improvements

### 3. Commit Messages
Follow conventional commits format:
```
type(scope): description

Examples:
feat(core): add intensity prediction head to transformer
fix(sensors): handle serial connection timeout gracefully
docs(api): update molecular design examples
perf(cache): implement LRU cache for embeddings
test(integration): add end-to-end sensor pipeline tests
```

### 4. Code Quality

#### Code Style
- **Black** for code formatting: `black olfactory_transformer/`
- **isort** for import sorting: `isort olfactory_transformer/`
- **flake8** for linting: `flake8 olfactory_transformer/`
- **mypy** for type checking: `mypy olfactory_transformer/`

#### Testing Requirements
- **Minimum 85% test coverage** for new code
- **Unit tests** for all public functions
- **Integration tests** for complete workflows
- **Docstring examples** that can be tested with doctest

#### Documentation Standards
- **Google-style docstrings** for all public APIs
- **Type hints** for all function parameters and returns
- **Usage examples** in docstrings for complex functions
- **Update README** for user-facing changes

### 5. Pull Request Process

#### Before Submitting
```bash
# Run full test suite
pytest tests/ --cov=olfactory_transformer --cov-report=term-missing

# Check code quality
black --check olfactory_transformer/
isort --check-only olfactory_transformer/
flake8 olfactory_transformer/
mypy olfactory_transformer/

# Ensure examples still work
python examples/basic_usage.py
```

#### PR Template
- **Clear description** of changes and motivation
- **Link to related issues** using "Fixes #123" or "Relates to #456"
- **Test plan** describing how changes were validated
- **Screenshots/outputs** for user-facing changes
- **Breaking changes** clearly documented
- **Performance impact** analysis for optimizations

#### Review Process
1. **Automated checks** must pass (CI/CD pipeline)
2. **Code review** by at least one maintainer
3. **Documentation review** for user-facing changes
4. **Performance testing** for optimization PRs
5. **Security review** for sensor/network code

## 🧪 Testing Guidelines

### Test Structure
```
tests/
├── test_core.py           # Core model and tokenizer tests
├── test_sensors.py        # Sensor integration tests
├── test_utils.py          # Utility function tests
├── test_integration.py    # End-to-end workflow tests
├── fixtures/              # Test data and fixtures
└── conftest.py           # Pytest configuration
```

### Test Categories

#### Unit Tests
- Test individual functions and classes in isolation
- Use mocking for external dependencies
- Fast execution (< 1 second per test)
- High coverage of edge cases and error conditions

#### Integration Tests
- Test complete workflows and component interactions
- Use real data when possible, mocks when necessary
- Moderate execution time (< 30 seconds per test)
- Focus on realistic usage scenarios

#### Performance Tests
- Benchmark critical operations (inference, training)
- Memory usage and resource utilization tests
- Scalability tests with varying input sizes
- Regression tests for optimization changes

### Mock Data Guidelines
- **Realistic molecular structures** (valid SMILES strings)
- **Diverse scent descriptors** covering all major families
- **Sensor readings** within typical operating ranges
- **Temporal data** with appropriate time series patterns

## 🏗️ Architecture Guidelines

### Code Organization
```
olfactory_transformer/
├── core/              # Core model and tokenizer
├── sensors/           # Sensor integration
├── design/            # Molecular design tools
├── evaluation/        # Evaluation metrics and validation
├── training/          # Training infrastructure
├── utils/             # Optimization and monitoring utilities
├── data/              # Data loading and processing
└── cli.py            # Command-line interface
```

### Design Principles
- **Modularity** - Components should be loosely coupled and easily testable
- **Extensibility** - Easy to add new sensor types, model architectures
- **Performance** - Optimize for inference speed and memory efficiency
- **Reliability** - Graceful error handling and recovery
- **Usability** - Clear APIs with comprehensive documentation

### Adding New Features

#### New Sensor Types
1. Create sensor class inheriting from base sensor interface
2. Implement required methods: `connect()`, `read_single()`, `calibrate()`
3. Add configuration options and error handling
4. Include comprehensive tests with hardware mocking
5. Update documentation and examples

#### New Model Architectures
1. Inherit from `OlfactoryTransformer` or create new base class
2. Implement forward pass and prediction methods
3. Add configuration options for hyperparameters
4. Include training and evaluation support
5. Benchmark against existing models

#### New Evaluation Metrics
1. Add metric computation to `evaluation/metrics.py`
2. Include statistical validation and confidence intervals
3. Compare against human panel data when possible
4. Document interpretation and usage guidelines
5. Add visualization tools for results

## 📊 Data Contributions

### Dataset Requirements
- **Proper licensing** - Ensure data can be shared openly
- **Quality validation** - Verified molecular structures and scent descriptions
- **Diversity** - Cover multiple chemical families and scent categories
- **Metadata** - Include source, collection method, validation status
- **Format consistency** - Follow established data schemas

### Data Formats
```python
# Molecular-scent pairs
{
    "smiles": "CCO",
    "scent_descriptors": ["alcoholic", "pungent", "sweet"],
    "intensity": 6.5,
    "chemical_family": "alcohol",
    "source": "GoodScents Database",
    "validated": true
}

# Sensor readings
{
    "timestamp": 1625097600.0,
    "gas_sensors": {"TGS2600": 2.45, "TGS2602": 1.87},
    "environmental": {"temperature": 23.5, "humidity": 45.2},
    "reference_compound": "ethanol",
    "concentration_ppm": 100.0
}
```

### Data Privacy
- **No personal data** without explicit consent
- **Anonymize** any human panel evaluations
- **Respect proprietary formulations** - no trade secrets
- **Follow GDPR** and other privacy regulations

## 🌍 Internationalization

### Multi-language Support
- **Scent descriptors** in multiple languages
- **Cultural variations** in scent perception
- **Unicode support** for non-Latin scripts
- **Localized examples** and documentation

### Translation Guidelines
- Work with native speakers for accuracy
- Consider cultural context of scent descriptions
- Maintain consistency across languages
- Include phonetic transcriptions when helpful

## 🔒 Security Guidelines

### Code Security
- **No hardcoded secrets** - use environment variables
- **Input validation** for all user-provided data
- **Sanitize SMILES** strings before processing
- **Rate limiting** for API endpoints
- **Secure sensor communications** when possible

### Model Security
- **Validate model inputs** to prevent adversarial examples
- **Monitor for anomalous predictions** in production
- **Secure model checkpoints** and training data
- **Document known limitations** and failure modes

## 📝 Documentation Standards

### API Documentation
- **Comprehensive docstrings** with examples
- **Parameter descriptions** including types and ranges
- **Return value documentation** with expected formats
- **Exception documentation** for error conditions
- **Usage examples** for common scenarios

### User Guides
- **Step-by-step tutorials** for common workflows  
- **Installation guides** for different platforms
- **Troubleshooting sections** for common issues
- **Performance optimization tips**
- **Integration examples** for real applications

### Release Documentation
- **Changelog** with clear categorization of changes
- **Migration guides** for breaking changes
- **Performance benchmarks** comparing versions
- **Known issues** and workarounds

## 🤝 Community Guidelines

### Communication
- **Be respectful** and inclusive in all interactions
- **Assume good intent** when reviewing contributions
- **Provide constructive feedback** with specific suggestions
- **Help newcomers** get started with the project
- **Share knowledge** through documentation and examples

### Collaboration
- **Credit contributors** appropriately in commits and releases
- **Acknowledge** data sources and inspiration
- **Collaborate openly** on design decisions
- **Resolve conflicts** through discussion and compromise
- **Celebrate successes** and learn from failures

## 🚀 Release Process

### Version Management
- **Semantic versioning** (MAJOR.MINOR.PATCH)
- **Feature branches** merged to main via PR
- **Release branches** for stabilization
- **Hotfix branches** for critical bug fixes

### Release Checklist
- [ ] All tests passing on CI/CD
- [ ] Documentation updated for changes
- [ ] Performance benchmarks validated
- [ ] Security review completed
- [ ] Changelog updated with changes
- [ ] Version numbers bumped appropriately
- [ ] Release notes prepared
- [ ] PyPI package built and tested

## 📞 Getting Help

### Documentation
- **README** - Start here for basic usage
- **API Reference** - Detailed function documentation
- **Examples** - Real-world usage scenarios
- **FAQ** - Common questions and solutions

### Community Support
- **GitHub Issues** - Bug reports and feature requests
- **GitHub Discussions** - General questions and community chat
- **Discord Server** - Real-time community support (coming soon)
- **Email** - Direct contact for sensitive issues

### Professional Support
- **Consulting** - Custom integration and optimization
- **Training** - Workshops and educational programs
- **Enterprise** - Commercial licensing and support

## 🎯 Roadmap Priorities

### Short Term (Q3 2025)
- [ ] Mobile/edge optimization
- [ ] Additional sensor integrations
- [ ] Multilingual scent descriptors
- [ ] Performance benchmarking suite

### Medium Term (Q4 2025)
- [ ] Federated learning platform
- [ ] Safety/toxicity prediction
- [ ] Industry partnership program
- [ ] Cloud deployment tools

### Long Term (2026+)
- [ ] Multimodal integration (visual, textual)
- [ ] Generative molecular design
- [ ] Real-time optimization
- [ ] Global scent perception studies

## 🏆 Recognition

### Contributors
All contributors will be acknowledged in:
- **README** contributor section
- **Release notes** for their contributions
- **Academic papers** when contributions are significant
- **Conference presentations** and talks

### Types of Recognition
- **Code Contributors** - Direct code contributions
- **Data Contributors** - Dataset provision and curation
- **Documentation Contributors** - Guides, examples, translations
- **Community Contributors** - Support, moderation, outreach
- **Industry Contributors** - Real-world validation and feedback

Thank you for contributing to the future of computational olfaction! 🌸🤖

---

For questions about contributing, please reach out through GitHub Issues or email the maintainers directly.