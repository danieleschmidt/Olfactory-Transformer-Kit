# Olfactory-Transformer-Kit

Foundation model toolkit for computational olfaction: molecular structure → scent description mapping.

## Architecture

- **MolecularTokenizer**: atom/bond vocabulary tokenization
- **OlfactoryTransformer**: 3-layer encoder-only transformer (numpy)
- **ScentDecoder**: multi-label classifier for 20 GoodScents-derived categories
- **PretrainPipeline**: synthetic data generation and evaluation

## Usage

```python
from olfactory_transformer.inference import InferencePipeline

pipeline = InferencePipeline()
result = pipeline.predict("CCO")  # ethanol
print(result["scents"])  # ["fresh", "sweet", ...]
```

## Scent Categories (20)

floral, citrus, woody, musky, fresh, sweet, spicy, earthy, marine, green,
fruity, herbal, resinous, animalic, powdery, smoky, balsamic, aldehydic, lactonic, waxy

## Components

### MolecularTokenizer

Tokenizes SMILES-like strings using atom and bond vocabularies.

```python
from olfactory_transformer.tokenizer import MolecularTokenizer

tok = MolecularTokenizer()
tokens = tok.tokenize("CCO")         # [1, 1, 2]
padded = tok.pad(tokens, max_len=64) # np.ndarray shape (64,)
decoded = tok.decode(tokens)         # "CCO"
```

**ATOM_VOCAB**: `{PAD:0, C:1, O:2, N:3, S:4, F:5, Cl:6, Br:7, P:8, UNK:9}`  
**BOND_VOCAB**: `{PAD:0, single:1, double:2, aromatic:3}`

### OlfactoryTransformer

Pure numpy 3-layer encoder-only transformer with sinusoidal positional encoding,
multi-head self-attention, feed-forward sublayers, and layer normalization.

```python
import numpy as np
from olfactory_transformer.transformer import OlfactoryTransformer

model = OlfactoryTransformer(vocab_size=10, embed_dim=64, n_heads=4, n_layers=3)
token_ids = np.array([[1, 1, 2, 0, 0]])  # (batch=1, seq=5)
embeddings = model.forward(token_ids)    # (1, 5, 64)
```

### ScentDecoder

Linear decoder (embed_dim → n_categories) with sigmoid activation for multi-label
scent classification.

```python
from olfactory_transformer.scent_decoder import ScentDecoder

decoder = ScentDecoder(embed_dim=64, n_categories=20)
probs = decoder.decode(embeddings)                  # (batch, 20)
labels = decoder.get_labels(embeddings, threshold=0.3)  # ["floral", "fresh", ...]
```

### PretrainPipeline

Generates synthetic (molecule_tokens, scent_labels) pairs and evaluates the model.

```python
from olfactory_transformer.pretrain import PretrainPipeline

pipeline = PretrainPipeline()
pairs = pipeline.generate_synthetic_pairs(n=100)
metrics = pipeline.evaluate(model, decoder, pairs)
# {"avg_loss": 0.69, "accuracy": 0.85, "precision": 0.42, "recall": 0.38}
```

## Installation

```bash
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest tests/ -v
```

## License

MIT
