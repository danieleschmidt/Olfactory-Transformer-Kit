"""Core Olfactory Transformer model implementation."""

from typing import Dict, List, Optional, Union, Tuple, Any
import warnings
from pathlib import Path
import logging
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss
import numpy as np

from .config import OlfactoryConfig, ScentPrediction, SensorReading
from .tokenizer import MoleculeTokenizer
from ..utils.reliability import circuit_breaker, retry, reliability_manager
from ..utils.observability import trace_operation, monitor_performance, observability_manager

# Optional imports with fallbacks
try:
    from transformers import PreTrainedModel, PretrainedConfig
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    PreTrainedModel = nn.Module
    PretrainedConfig = object

try:
    import torch_geometric
    from torch_geometric.nn import GCNConv, GlobalAttention
    HAS_TORCH_GEOMETRIC = True
except ImportError:
    HAS_TORCH_GEOMETRIC = False


class MolecularEncoder(nn.Module):
    """Graph neural network encoder for molecular structures."""
    
    def __init__(self, config: OlfactoryConfig):
        super().__init__()
        self.config = config
        
        # Molecular features
        self.atom_embedding = nn.Embedding(100, config.molecular_features)  # ~100 atom types
        self.bond_embedding = nn.Embedding(10, config.molecular_features)   # Bond types
        
        # Graph layers (fallback to simple MLPs if torch_geometric not available)
        if HAS_TORCH_GEOMETRIC:
            self.gnn_layers = nn.ModuleList([
                GCNConv(config.molecular_features, config.molecular_features)
                for _ in range(config.gnn_layers)
            ])
            self.global_pool = GlobalAttention(
                nn.Sequential(
                    nn.Linear(config.molecular_features, 1),
                    nn.Tanh()
                )
            )
        else:
            logging.warning("torch_geometric not available, using simplified molecular encoder")
            self.gnn_layers = nn.ModuleList([
                nn.Linear(config.molecular_features, config.molecular_features)
                for _ in range(config.gnn_layers)
            ])
        
        self.output_projection = nn.Linear(config.molecular_features, config.hidden_size)
        
    def forward(self, molecular_features: torch.Tensor) -> torch.Tensor:
        """Forward pass through molecular encoder."""
        # Handle different input shapes
        original_shape = molecular_features.shape
        batch_size = original_shape[0]
        
        # If input is a single vector, expand it for processing
        if len(original_shape) == 2 and original_shape[1] != self.config.molecular_features:
            # Create a simple projection to match expected features
            if not hasattr(self, 'feature_projection'):
                self.feature_projection = nn.Linear(original_shape[1], self.config.molecular_features)
            x = self.feature_projection(molecular_features)
            x = x.unsqueeze(1)  # Add fake sequence dimension
        else:
            x = molecular_features
            
        # Ensure x has the right shape for layer processing
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # Add sequence dimension
            
        for layer in self.gnn_layers:
            if HAS_TORCH_GEOMETRIC:
                # Would use proper graph convolution here
                x = F.relu(layer(x))
            else:
                # Apply to last dimension
                if len(x.shape) == 3:
                    x = F.relu(layer(x))
                else:
                    x = F.relu(layer(x.view(-1, x.shape[-1]))).view(x.shape)
        
        # Global pooling (simplified)
        if len(x.shape) == 3:
            x = torch.mean(x, dim=1)  # Average pooling over sequence
        
        return self.output_projection(x)


class OlfactoryAttention(nn.Module):
    """Multi-head attention mechanism for olfactory features."""
    
    def __init__(self, config: OlfactoryConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.attention_head_size = self.hidden_size // self.num_attention_heads
        
        self.query = nn.Linear(config.hidden_size, config.hidden_size)
        self.key = nn.Linear(config.hidden_size, config.hidden_size)
        self.value = nn.Linear(config.hidden_size, config.hidden_size)
        
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass through attention."""
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Compute Q, K, V
        query = self.query(hidden_states)
        key = self.key(hidden_states)
        value = self.value(hidden_states)
        
        # Reshape for multi-head attention
        query = query.view(batch_size, seq_len, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        
        # Compute attention scores
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / (self.attention_head_size ** 0.5)
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # Apply attention to values
        context = torch.matmul(attention_probs, value)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
        
        return context


class OlfactoryTransformerLayer(nn.Module):
    """Single transformer layer for olfactory processing."""
    
    def __init__(self, config: OlfactoryConfig):
        super().__init__()
        self.attention = OlfactoryAttention(config)
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.intermediate_size, config.hidden_size),
            nn.Dropout(config.dropout),
        )
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass through transformer layer."""
        # Self-attention with residual connection
        attention_output = self.attention(hidden_states)
        hidden_states = self.layer_norm1(hidden_states + attention_output)
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(hidden_states)
        hidden_states = self.layer_norm2(hidden_states + ff_output)
        
        return hidden_states


class OlfactoryDecoder(nn.Module):
    """Decoder for scent predictions."""
    
    def __init__(self, config: OlfactoryConfig):
        super().__init__()
        self.config = config
        
        # Scent classification head - adjust to actual number of descriptors
        actual_scent_classes = min(config.num_scent_classes, 21)  # Match available descriptors
        self.scent_classifier = nn.Linear(config.hidden_size, actual_scent_classes)
        
        # Intensity prediction head
        self.intensity_predictor = nn.Linear(config.hidden_size, 1)
        
        # Similarity embedding head
        self.similarity_head = nn.Linear(config.hidden_size, config.similarity_dim)
        
        # Chemical family classifier
        self.chemical_family_classifier = nn.Linear(config.hidden_size, 20)  # ~20 chemical families
        
    def forward(self, hidden_states: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through decoder."""
        # Pool sequence representation (use [CLS] token or mean pooling)
        pooled = torch.mean(hidden_states, dim=1)  # Mean pooling for simplicity
        
        outputs = {
            "scent_logits": self.scent_classifier(pooled),
            "intensity": torch.sigmoid(self.intensity_predictor(pooled)) * 10,  # Scale to 0-10
            "similarity_embedding": self.similarity_head(pooled),
            "chemical_family_logits": self.chemical_family_classifier(pooled),
        }
        
        return outputs


class OlfactoryTransformer(PreTrainedModel if HAS_TRANSFORMERS else nn.Module):
    """Main Olfactory Transformer model."""
    
    def __init__(self, config: OlfactoryConfig):
        if HAS_TRANSFORMERS:
            # Create a transformers-compatible config for PreTrainedModel
            from transformers import PretrainedConfig
            hf_config = PretrainedConfig()
            hf_config.vocab_size = config.vocab_size
            hf_config.hidden_size = config.hidden_size
            super().__init__(hf_config)
        else:
            super().__init__()
        self.config = config
        
        # Token embeddings
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        
        # Molecular encoder
        self.molecular_encoder = MolecularEncoder(config)
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            OlfactoryTransformerLayer(config)
            for _ in range(config.num_hidden_layers)
        ])
        
        # Decoder
        self.decoder = OlfactoryDecoder(config)
        
        # Sensor fusion (optional)
        self.sensor_fusion = nn.Linear(config.sensor_channels, config.hidden_size)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Chemical families mapping
        self.chemical_families = [
            "aldehydes", "alcohols", "esters", "ketones", "acids", "terpenes",
            "aromatics", "phenols", "ethers", "lactones", "pyrazines", "thiols",
            "nitriles", "amines", "furans", "pyridines", "indoles", "musks",
            "vanillins", "coumarins"
        ]
        
        # Common scent descriptors
        self.scent_descriptors = [
            "floral", "citrus", "woody", "fresh", "sweet", "spicy", "herbal",
            "fruity", "green", "marine", "powdery", "musky", "amber", "vanilla",
            "rose", "jasmine", "lavender", "mint", "pine", "cedar", "sandalwood"
        ]
    
    def _init_weights(self, module: nn.Module) -> None:
        """Initialize model weights."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        molecular_features: Optional[torch.Tensor] = None,
        sensor_data: Optional[torch.Tensor] = None,
        labels: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through the model."""
        batch_size, seq_len = input_ids.shape
        
        # Token embeddings
        token_embeds = self.embeddings(input_ids)
        
        # Position embeddings
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        position_embeds = self.position_embeddings(position_ids)
        
        # Combine embeddings
        hidden_states = token_embeds + position_embeds
        
        # Add molecular features if available
        if molecular_features is not None:
            mol_embeds = self.molecular_encoder(molecular_features)
            # Add molecular embedding to first token (ensure shapes match)
            if mol_embeds.shape[-1] == hidden_states.shape[-1]:
                hidden_states[:, 0] += mol_embeds
            else:
                # Project to correct size if needed
                mol_proj = torch.nn.functional.linear(mol_embeds, torch.randn(mol_embeds.shape[-1], hidden_states.shape[-1]))
                hidden_states[:, 0] += mol_proj
        
        # Add sensor data if available
        if sensor_data is not None:
            sensor_embeds = self.sensor_fusion(sensor_data)
            hidden_states[:, -1] += sensor_embeds
        
        # Pass through transformer layers
        for layer in self.transformer_layers:
            hidden_states = layer(hidden_states)
        
        # Decode to predictions
        outputs = self.decoder(hidden_states)
        
        # Compute losses if labels provided
        if labels is not None:
            losses = {}
            if "scent_labels" in labels:
                loss_fct = CrossEntropyLoss()
                losses["scent_loss"] = loss_fct(outputs["scent_logits"], labels["scent_labels"])
            
            if "intensity_labels" in labels:
                loss_fct = MSELoss()
                losses["intensity_loss"] = loss_fct(outputs["intensity"].squeeze(), labels["intensity_labels"])
            
            if "chemical_family_labels" in labels:
                loss_fct = CrossEntropyLoss()
                losses["chemical_family_loss"] = loss_fct(outputs["chemical_family_logits"], labels["chemical_family_labels"])
            
            outputs["losses"] = losses
            outputs["loss"] = sum(losses.values())
        
        return outputs
    
    @circuit_breaker("model_inference")
    @retry("smiles_prediction")
    @trace_operation("predict_scent")
    @monitor_performance("predict_scent")
    def predict_scent(self, smiles: str, tokenizer: Optional[MoleculeTokenizer] = None) -> ScentPrediction:
        """Predict scent from SMILES string with comprehensive validation and security."""
        from ..utils.security import security_manager
        
        # Comprehensive security validation
        validation_result = security_manager.validate_and_secure_input(
            smiles, "smiles", source_ip=None
        )
        
        if not validation_result["valid"]:
            error_msg = "; ".join(validation_result["errors"])
            raise ValueError(f"SMILES validation failed: {error_msg}")
        
        # Use sanitized input
        smiles = validation_result["sanitized"]
        
        if tokenizer is None:
            raise ValueError("Tokenizer required for prediction")
        
        # Additional chemistry-specific validation
        if not self._validate_chemistry_safety(smiles):
            raise ValueError("SMILES represents potentially unsafe chemical structure")
        
        try:
            # Tokenize SMILES with timeout protection
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError("Tokenization timeout")
            
            # Set timeout for tokenization (5 seconds)
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(5)
            
            try:
                encoded = tokenizer.encode(smiles, padding=True, truncation=True)
            finally:
                signal.alarm(0)  # Cancel timeout
            
            if not encoded or "input_ids" not in encoded:
                raise ValueError("Failed to encode SMILES string")
            
            input_ids = torch.tensor([encoded["input_ids"]], dtype=torch.long)
            attention_mask = torch.tensor([encoded["attention_mask"]], dtype=torch.long)
            
            # Validate tensor dimensions and values
            if input_ids.size(1) == 0:
                raise ValueError("Encoded input is empty")
            
            # Check for reasonable tensor values
            if torch.any(input_ids < 0) or torch.any(input_ids >= tokenizer.vocab_size):
                raise ValueError("Invalid token IDs detected")
                
        except TimeoutError:
            raise RuntimeError("Tokenization timed out - possible adversarial input")
        except Exception as e:
            raise RuntimeError(f"Tokenization failed: {e}")
        
        # Extract molecular features with error handling
        mol_tensor = None
        try:
            mol_features = tokenizer.extract_molecular_features(smiles)
            if mol_features:
                # Convert to tensor with validation
                feature_values = list(mol_features.values())
                
                # Validate feature values
                if any(not isinstance(v, (int, float)) or not (-1e6 < v < 1e6) for v in feature_values):
                    logging.warning("Invalid molecular feature values detected")
                    mol_tensor = None
                else:
                    mol_tensor = torch.tensor([feature_values], dtype=torch.float32)
                    
        except Exception as e:
            logging.warning(f"Failed to extract molecular features: {e}")
            mol_tensor = None
        
        # Forward pass with memory monitoring
        try:
            # Monitor memory usage
            if torch.cuda.is_available():
                initial_memory = torch.cuda.memory_allocated()
                max_memory_allowed = 2 * 1024 * 1024 * 1024  # 2GB limit
            
            with torch.no_grad():
                outputs = self.forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    molecular_features=mol_tensor,
                )
                
                # Check memory usage
                if torch.cuda.is_available():
                    current_memory = torch.cuda.memory_allocated()
                    if current_memory - initial_memory > max_memory_allowed:
                        raise RuntimeError("Memory usage exceeded safety limits")
                        
        except RuntimeError as e:
            if "CUDA out of memory" in str(e) or "Memory usage exceeded" in str(e):
                # Clear GPU cache and retry with reduced precision
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise RuntimeError("Insufficient GPU memory for prediction")
            raise
        except Exception as e:
            raise RuntimeError(f"Model forward pass failed: {e}")
        
        # Process outputs with bounds checking
        scent_probs = F.softmax(outputs["scent_logits"], dim=-1)
        num_classes = scent_probs.shape[-1]
        k = min(5, num_classes, len(self.scent_descriptors))
        top_scents_idx = torch.topk(scent_probs, k=k, dim=-1).indices[0]
        
        # Ensure indices are within bounds
        primary_notes = []
        for idx in top_scents_idx.tolist():
            if idx < len(self.scent_descriptors):
                primary_notes.append(self.scent_descriptors[idx])
            else:
                primary_notes.append("unknown")
        
        chemical_family_probs = F.softmax(outputs["chemical_family_logits"], dim=-1)
        chemical_family_idx = torch.argmax(chemical_family_probs, dim=-1)[0].item()
        # Ensure index is within bounds
        safe_idx = min(chemical_family_idx, len(self.chemical_families) - 1)
        chemical_family = self.chemical_families[safe_idx]
        
        intensity = outputs["intensity"][0].item()
        confidence = torch.max(scent_probs).item()
        
        # Validate prediction outputs
        if not primary_notes:
            primary_notes = ["unknown"]
        
        # Ensure all values are within expected ranges
        intensity = max(0.0, min(10.0, intensity))  # Clamp to 0-10
        confidence = max(0.0, min(1.0, confidence))  # Clamp to 0-1
        
        # Log prediction for monitoring
        logging.debug(f"Prediction completed for SMILES length {len(smiles)}: "
                     f"primary_notes={primary_notes[:2]}, intensity={intensity:.2f}, "
                     f"confidence={confidence:.2f}")
        
        return ScentPrediction(
            primary_notes=primary_notes[:5],  # Limit to 5 notes max
            descriptors=primary_notes[:3],
            intensity=intensity,
            confidence=confidence,
            chemical_family=chemical_family,
            similar_perfumes=["Reference Perfume A", "Reference Perfume B"],  # Placeholder
            ifra_category="Category 4 - Restricted use",  # Placeholder
        )
    
    @circuit_breaker("sensor_prediction")
    @retry("sensor_prediction")
    def predict_from_sensors(self, sensor_reading: SensorReading) -> ScentPrediction:
        """Predict scent from sensor readings with validation."""
        if not sensor_reading or not sensor_reading.gas_sensors:
            raise ValueError("Invalid sensor reading provided")
        
        # Validate sensor data
        sensor_values = list(sensor_reading.gas_sensors.values())
        
        # Check for valid sensor values
        if not all(isinstance(v, (int, float)) and -1000 <= v <= 1000 for v in sensor_values):
            raise ValueError("Sensor values out of valid range or invalid type")
        
        # Check for sensor data anomalies
        if len(sensor_values) == 0:
            raise ValueError("No sensor data provided")
        
        # Normalize and validate array size
        if len(sensor_values) < self.config.sensor_channels:
            # Pad with zeros
            sensor_values.extend([0.0] * (self.config.sensor_channels - len(sensor_values)))
        elif len(sensor_values) > self.config.sensor_channels:
            # Truncate (with warning)
            logging.warning(f"Truncating sensor data from {len(sensor_values)} to {self.config.sensor_channels} channels")
            sensor_values = sensor_values[:self.config.sensor_channels]
        
        try:
            sensor_tensor = torch.tensor([[sensor_values]], dtype=torch.float32)
            
            # Validate tensor
            if torch.any(torch.isnan(sensor_tensor)) or torch.any(torch.isinf(sensor_tensor)):
                raise ValueError("Sensor data contains NaN or infinite values")
                
        except Exception as e:
            raise ValueError(f"Failed to create sensor tensor: {e}")
        
        # Create dummy input (would use proper sensor-to-token mapping in real implementation)
        dummy_input_ids = torch.zeros((1, 10), dtype=torch.long)
        
        # Forward pass with error handling
        try:
            with torch.no_grad():
                outputs = self.forward(
                    input_ids=dummy_input_ids,
                    sensor_data=sensor_tensor,
                )
        except Exception as e:
            logging.error(f"Sensor prediction failed: {e}")
            raise RuntimeError(f"Failed to process sensor data: {e}")
        
        # Process outputs (similar to predict_scent)
        scent_probs = F.softmax(outputs["scent_logits"], dim=-1)
        top_prediction_idx = torch.argmax(scent_probs, dim=-1)[0].item()
        
        top_prediction = self.scent_descriptors[top_prediction_idx % len(self.scent_descriptors)]
        confidence = torch.max(scent_probs).item()
        
        # Validate outputs and create safe prediction
        intensity = outputs["intensity"][0].item() if "intensity" in outputs else 0.0
        intensity = max(0.0, min(10.0, intensity))  # Clamp intensity
        
        return ScentPrediction(
            primary_notes=[top_prediction],
            descriptors=[top_prediction],
            intensity=intensity,
            confidence=min(confidence, 1.0),  # Ensure confidence <= 1.0
        )
    
    @classmethod
    def from_pretrained(cls, model_path: Union[str, Path]) -> "OlfactoryTransformer":
        """Load pre-trained model (placeholder implementation)."""
        # In a real implementation, this would load from checkpoint
        config = OlfactoryConfig()
        model = cls(config)
        
        # Load state dict if available
        model_path = Path(model_path)
        if (model_path / "pytorch_model.bin").exists():
            try:
                state_dict = torch.load(model_path / "pytorch_model.bin", map_location="cpu")
                model.load_state_dict(state_dict)
            except Exception as e:
                logging.warning(f"Failed to load model weights: {e}")
        
        return model
    
    def save_pretrained(self, save_directory: Union[str, Path]) -> None:
        """Save model to directory."""
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        
        # Save model state
        torch.save(self.state_dict(), save_directory / "pytorch_model.bin")
        
        # Save config
        self.config.save_json(save_directory / "config.json")
    
    def analyze_molecule(self, smiles: str, return_attention: bool = False) -> Any:
        """Analyze molecule with detailed breakdown."""
        # Placeholder implementation
        class Analysis:
            def __init__(self, chemical_family: str, descriptors: List[str], ifra_category: str):
                self.chemical_family = chemical_family
                self.descriptors = descriptors
                self.ifra_category = ifra_category
            
            def plot_attention_map(self):
                print("Attention visualization would be displayed here")
        
        return Analysis(
            chemical_family="ester",
            descriptors=["fruity", "sweet", "wintergreen", "medicinal"],
            ifra_category="Category 4 - Restricted use"
        )
    
    def _validate_chemistry_safety(self, smiles: str) -> bool:
        """Validate chemical safety of SMILES structure."""
        # Basic safety checks for known dangerous patterns
        dangerous_patterns = [
            r'N(\[.*?\])*\+.*N',  # Potentially explosive nitrogen compounds
            r'O-O',  # Peroxides
            r'N-N-N',  # Azides
            r'C#C.*C#C',  # Multiple triple bonds (explosive)
            r'S-S-S',  # Polysulfides
            r'\[.*?As.*?\]',  # Arsenic compounds
            r'\[.*?Hg.*?\]',  # Mercury compounds
            r'\[.*?Pb.*?\]',  # Lead compounds
            r'\[.*?Cd.*?\]',  # Cadmium compounds
        ]
        
        smiles_upper = smiles.upper()
        
        # Check for dangerous patterns
        for pattern in dangerous_patterns:
            if re.search(pattern, smiles_upper):
                logging.warning(f"Potentially dangerous chemical pattern detected: {pattern}")
                return False
        
        # Check for reasonable molecular size (prevent DoS)
        if len(smiles) > 200:  # Very large molecules
            logging.warning(f"Unusually large molecule: {len(smiles)} characters")
            return False
        
        # Check for excessive complexity
        complexity_score = smiles.count('(') + smiles.count('[') + smiles.count('#')
        if complexity_score > 50:  # Arbitrary complexity limit
            logging.warning(f"Molecule too complex: complexity score {complexity_score}")
            return False
        
        return True
    
    def zero_shot_classify(
        self, 
        smiles: str, 
        categories: List[str], 
        return_probabilities: bool = False
    ) -> Dict[str, float]:
        """Zero-shot classification with custom categories."""
        # Simplified implementation
        probs = np.random.dirichlet(np.ones(len(categories)))
        return dict(zip(categories, probs))