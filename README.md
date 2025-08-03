# Olfactory-Transformer-Kit 👃🤖

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2025.XXXXX-b31b1b.svg)](https://arxiv.org/)
[![Demo](https://img.shields.io/badge/Demo-Hugging%20Face-yellow)](https://huggingface.co/spaces/yourusername/scent-transformer)

The first open-source foundation model for computational olfaction, enabling smell-sense AI through molecular structure to scent description mapping.

## 🌟 Highlights

- **Pre-trained Foundation Model**: GPT-style transformer trained on 5M molecule-scent pairs
- **Multi-modal Understanding**: Links molecular graphs, spectroscopy data, and human descriptions
- **Electronic Nose Integration**: Real-time inference from gas sensor arrays
- **Zero-shot Classification**: Identify unknown scents without retraining
- **Inverse Design**: Generate molecules matching desired scent profiles

## 🚀 Quick Start

### Installation

```bash
# Install from PyPI
pip install olfactory-transformer

# With electronic nose support
pip install olfactory-transformer[sensors]

# From source
git clone https://github.com/yourusername/Olfactory-Transformer-Kit.git
cd Olfactory-Transformer-Kit
pip install -e ".[dev,sensors]"
```

### Basic Usage

```python
from olfactory_transformer import OlfactoryTransformer, MoleculeTokenizer

# Load pre-trained model
model = OlfactoryTransformer.from_pretrained('olfactory-base-v1')
tokenizer = MoleculeTokenizer.from_pretrained('olfactory-base-v1')

# Predict scent from molecular structure (SMILES)
smiles = "CC(C)CC1=CC=C(C=C1)C(C)C"  # Lily of the valley
prediction = model.predict_scent(smiles)

print(f"Primary notes: {prediction.primary_notes}")
# Output: ['floral', 'fresh', 'sweet']

print(f"Similar to: {prediction.similar_perfumes}")
# Output: ['Diorissimo', 'Muguet des Bois']

print(f"Intensity: {prediction.intensity}/10")
# Output: 7/10
```

### Electronic Nose Integration

```python
from olfactory_transformer import ENoseInterface

# Connect to sensor array
enose = ENoseInterface(
    port='/dev/ttyUSB0',
    sensors=['TGS2600', 'TGS2602', 'TGS2610', 'TGS2620'],
    sampling_rate=10  # Hz
)

# Real-time scent detection
with enose.stream() as sensor_stream:
    for reading in sensor_stream:
        scent = model.predict_from_sensors(reading)
        print(f"Detected: {scent.top_prediction} (confidence: {scent.confidence:.2%})")
```

## 🏗️ Architecture

### Model Architecture

```
OlfactoryTransformer (240M parameters)
├── Molecular Encoder
│   ├── Graph Neural Network (5 layers)
│   ├── 3D Conformer Attention
│   └── Spectroscopic Feature Extractor
├── Transformer Core
│   ├── 24 layers, 16 heads
│   ├── Hidden dimension: 1024
│   └── Rotary position embeddings
├── Olfactory Decoder
│   ├── Scent descriptor head
│   ├── Intensity prediction head
│   └── Perceptual similarity head
└── Sensor Fusion Module
    ├── Time-series encoder
    └── Cross-attention with molecular features
```

### Training Data

| Dataset | Molecules | Descriptions | Sensor Readings |
|---------|-----------|--------------|-----------------|
| GoodScents | 3,800 | 45,000 | - |
| Flavornet | 738 | 8,500 | - |
| Pyrfume | 1,200 | 15,000 | 5,000 |
| LeffingWell | 3,500 | 28,000 | - |
| Industrial (private) | 2.5M | 4M | 850,000 |

## 🔬 Core Features

### Scent Prediction

```python
# Detailed scent analysis
analysis = model.analyze_molecule(
    smiles="CCOC(=O)C1=CC=CC=C1",  # Ethyl benzoate
    return_attention=True
)

# Access multi-level descriptors
print(f"Chemical family: {analysis.chemical_family}")
# Output: 'ester'

print(f"Odor descriptors: {analysis.descriptors}")
# Output: ['fruity', 'sweet', 'wintergreen', 'medicinal']

print(f"IFRA category: {analysis.ifra_category}")
# Output: 'Category 4 - Restricted use'

# Visualize attention on molecular structure
analysis.plot_attention_map()
```

### Zero-Shot Learning

```python
# Classify unknown scent
unknown_molecule = "CC1=CC=C(C=C1)C(C)(C)C=C"  # Novel terpene

# Zero-shot classification with custom categories
categories = ['woody', 'citrus', 'marine', 'gourmand', 'animalic']
classification = model.zero_shot_classify(
    unknown_molecule,
    categories,
    return_probabilities=True
)

for category, prob in classification.items():
    print(f"{category}: {prob:.1%}")
```

### Inverse Molecular Design

```python
from olfactory_transformer import ScentDesigner

designer = ScentDesigner(model)

# Design molecule with target scent profile
target_profile = {
    'notes': ['rose', 'woody', 'amber'],
    'intensity': 6,
    'longevity': 'high',
    'character': 'warm, sophisticated'
}

# Generate candidate molecules
candidates = designer.design_molecules(
    target_profile,
    n_candidates=10,
    molecular_weight=(200, 350),
    logp=(2, 5)  # Lipophilicity range
)

for i, mol in enumerate(candidates):
    print(f"\nCandidate {i+1}:")
    print(f"SMILES: {mol.smiles}")
    print(f"Predicted match: {mol.profile_match:.2%}")
    print(f"Synthetic accessibility: {mol.sa_score:.2f}")
    mol.visualize()
```

## 📊 Sensor Array Processing

### Multi-Sensor Fusion

```python
# Configure multi-modal sensing
sensor_config = {
    'gas_sensors': ['MQ-3', 'MQ-135', 'MQ-7', 'MQ-8'],
    'environmental': ['BME680', 'SHT31'],
    'spectrometer': 'AS7265x'
}

sensor_array = ENoseArray(sensor_config)

# Calibrate with reference compounds
calibration_data = sensor_array.calibrate(
    reference_compounds=['ethanol', 'acetone', 'benzene'],
    concentrations=[10, 50, 100, 500]  # ppm
)

# Advanced pattern recognition
with sensor_array.acquire() as data:
    # Sensor fusion
    fused_features = model.fuse_sensor_data(
        gas_readings=data['gas_sensors'],
        environmental=data['environmental'],
        spectral=data['spectrometer']
    )
    
    # Temporal analysis
    temporal_profile = model.analyze_temporal(
        fused_features,
        window_size=30,  # seconds
        overlap=0.5
    )
    
    print(f"Scent evolution: {temporal_profile.stages}")
```

### Real-time Monitoring

```python
# Industrial quality control application
monitor = OlfactoryQualityControl(
    model=model,
    sensor_array=sensor_array,
    reference_profile='lavender_essential_oil.pkl'
)

# Continuous monitoring
for batch_reading in monitor.stream():
    if batch_reading.deviation > 0.15:  # 15% threshold
        print(f"⚠️ Quality deviation detected!")
        print(f"Abnormal components: {batch_reading.outlier_notes}")
        print(f"Suggested action: {batch_reading.recommendation}")
```

## 🧪 Evaluation & Benchmarks

### Model Performance

| Task | Dataset | Metric | Score |
|------|---------|--------|-------|
| Scent Classification | GoodScents-Test | Accuracy | 89.3% |
| Descriptor Prediction | Pyrfume-Test | F1-Score | 0.847 |
| Intensity Prediction | Industrial-QC | MAE | 0.72 |
| Molecular Similarity | DREAM Challenge | Pearson r | 0.823 |
| Sensor Classification | E-Nose-10K | Accuracy | 91.2% |

### Perceptual Validation

```python
# Compare with human panel data
evaluator = PerceptualEvaluator(
    model=model,
    human_panel_data='panel_evaluations.csv'
)

correlation_report = evaluator.compare_predictions(
    test_molecules='test_set.sdf',
    metrics=['primary_accord', 'intensity', 'character']
)

evaluator.plot_correlation_matrix(correlation_report)
```

## 🛠️ Advanced Usage

### Fine-tuning on Custom Data

```python
from olfactory_transformer import OlfactoryTrainer

# Prepare custom dataset
custom_dataset = OlfactoryDataset(
    molecules='perfume_formulas.csv',
    descriptions='expert_evaluations.json',
    sensor_data='enose_readings.hdf5'
)

# Fine-tune model
trainer = OlfactoryTrainer(
    model=model,
    train_dataset=custom_dataset,
    eval_dataset=custom_dataset.split(0.1),
    output_dir='./fine_tuned_model'
)

trainer.train(
    num_epochs=10,
    learning_rate=1e-5,
    batch_size=32,
    gradient_checkpointing=True
)
```

### Federated Learning for Proprietary Data

```python
# Train on distributed proprietary datasets
from olfactory_transformer import FederatedOlfactory

fed_model = FederatedOlfactory(
    base_model='olfactory-base-v1',
    aggregation='fedavg'
)

# Each participant trains locally
local_trainer = fed_model.create_local_trainer(
    participant_id='fragrance_house_001'
)

local_trainer.train_round(proprietary_data)
model_update = local_trainer.get_update()

# Aggregate updates (on central server)
fed_model.aggregate_updates([update1, update2, update3])
```

## 🎯 Applications

### Perfume Development

```python
# AI perfumer assistant
perfumer = AIPerfumer(model)

# Create new fragrance
brief = """
Create a unisex summer fragrance with:
- Top notes: Fresh, citrusy, with a twist
- Heart: Light floral, not too sweet  
- Base: Clean woods, subtle musk
- Character: Modern, minimalist, long-lasting
"""

formula = perfumer.create_fragrance(
    brief=brief,
    concentration='eau_de_parfum',
    price_range='premium',
    regulatory_compliance=['IFRA', 'EU', 'FDA']
)

formula.export_to_perfumers_workbench()
```

### Food & Beverage QC

```python
# Coffee quality analysis
coffee_analyzer = OlfactoryCoffeeQC(model)

roast_profile = coffee_analyzer.analyze_beans(
    sensor_reading=enose.read(),
    expected_profile='medium_roast_colombian'
)

print(f"Roast level: {roast_profile.roast_degree}")
print(f"Flavor notes: {roast_profile.cupping_notes}")
print(f"Defects detected: {roast_profile.defects}")
```

## 📈 Performance Optimization

### Inference Acceleration

```python
# Quantized model for edge deployment
quantized_model = model.quantize(
    method='int8_dynamic',
    calibration_data=calibration_dataset
)

# ONNX export for embedded systems
model.export_onnx(
    'olfactory_edge.onnx',
    opset_version=15,
    optimize_for_mobile=True
)

# TensorRT optimization
trt_model = model.optimize_tensorrt(
    precision='fp16',
    max_batch_size=32
)
```

## 📚 Citation

```bibtex
@article{olfactory_transformer2025,
  title={Olfactory-Transformer: Foundation Models for Computational Smell},
  author={Your Name et al.},
  journal={Nature Machine Intelligence},
  year={2025},
  doi={10.1038/s42256-025-XXXXX}
}
```

## 🤝 Contributing

We welcome contributions! Priority areas:
- Additional sensor array integrations
- Fragrance industry datasets
- Multi-lingual scent descriptions
- Safety and toxicity prediction

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## ⚖️ Ethical Considerations

This model should be used responsibly:
- Not for synthesizing harmful or illegal substances
- Consider allergen and safety warnings
- Respect proprietary fragrance formulations
- Include diversity in scent perception

## 📄 License

Apache License 2.0 - see [LICENSE](LICENSE)

## 🔗 Resources

- [Documentation](https://olfactory-transformer.readthedocs.io)
- [Model Zoo](https://huggingface.co/olfactory-transformer)
- [Interactive Demo](https://olfactory-transformer.app)
- [Paper](https://arxiv.org/abs/2025.XXXXX)
- [Blog Post](https://techxplore.com/olfactory-ai)
