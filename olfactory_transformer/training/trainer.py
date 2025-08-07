"""Training infrastructure for olfactory transformer models."""

from typing import Dict, List, Optional, Any, Union, Callable, Tuple
import logging
import time
from pathlib import Path
import json
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm

# Optional imports
try:
    from transformers import AdamW, get_linear_schedule_with_warmup
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    from torch.optim import AdamW

from ..core.config import OlfactoryConfig
from ..core.model import OlfactoryTransformer
from ..core.tokenizer import MoleculeTokenizer


@dataclass
class TrainingArguments:
    """Training configuration arguments."""
    output_dir: str = "./checkpoints"
    num_epochs: int = 10
    learning_rate: float = 1e-5
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    warmup_steps: int = 500
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # Evaluation
    eval_steps: int = 500
    save_steps: int = 1000
    logging_steps: int = 100
    
    # Advanced options
    gradient_checkpointing: bool = True
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True
    
    # Loss weights
    scent_loss_weight: float = 1.0
    intensity_loss_weight: float = 0.5
    chemical_family_loss_weight: float = 0.3
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class OlfactoryDataset(Dataset):
    """Dataset class for olfactory training data."""
    
    def __init__(
        self,
        molecules: Union[str, Path, List[str]],
        descriptions: Union[str, Path, List[Dict[str, Any]]],
        sensor_data: Optional[Union[str, Path, List[Dict[str, Any]]]] = None,
        tokenizer: Optional[MoleculeTokenizer] = None,
        max_length: int = 512,
    ):
        self.tokenizer = tokenizer or MoleculeTokenizer()
        self.max_length = max_length
        
        # Load data
        self.molecules = self._load_molecules(molecules)
        self.descriptions = self._load_descriptions(descriptions)
        self.sensor_data = self._load_sensor_data(sensor_data) if sensor_data else None
        
        # Validate data consistency
        assert len(self.molecules) == len(self.descriptions), \
            f"Molecules ({len(self.molecules)}) and descriptions ({len(self.descriptions)}) must have same length"
        
        if self.sensor_data:
            assert len(self.molecules) == len(self.sensor_data), \
                f"Molecules ({len(self.molecules)}) and sensor data ({len(self.sensor_data)}) must have same length"
        
        logging.info(f"Loaded dataset with {len(self.molecules)} samples")
    
    def _load_molecules(self, molecules: Union[str, Path, List[str]]) -> List[str]:
        """Load molecule SMILES from various sources."""
        if isinstance(molecules, list):
            return molecules
        
        molecules_path = Path(molecules)
        if molecules_path.suffix == '.csv':
            import pandas as pd
            df = pd.read_csv(molecules_path)
            if 'smiles' in df.columns:
                return df['smiles'].tolist()
            elif 'SMILES' in df.columns:
                return df['SMILES'].tolist()
        
        # Generate mock data for development
        return [
            "CCO",  # Ethanol
            "CC(C)O",  # Isopropanol
            "C1=CC=CC=C1",  # Benzene
            "CC(=O)OCC",  # Ethyl acetate
            "C1=CC=C(C=C1)C=O",  # Benzaldehyde
        ] * 20  # Repeat for demo
    
    def _load_descriptions(self, descriptions: Union[str, Path, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Load scent descriptions."""
        if isinstance(descriptions, list):
            return descriptions
        
        desc_path = Path(descriptions)
        if desc_path.suffix == '.json':
            with open(desc_path, 'r') as f:
                return json.load(f)
        elif desc_path.suffix == '.csv':
            import pandas as pd
            df = pd.read_csv(desc_path)
            descriptions = []
            for _, row in df.iterrows():
                desc = {
                    "primary_notes": row.get("primary_notes", "floral,fresh").split(","),
                    "intensity": float(row.get("intensity", 5.0)),
                    "chemical_family": row.get("chemical_family", "alcohol"),
                }
                descriptions.append(desc)
            return descriptions
        
        # Generate mock descriptions
        mock_descriptions = []
        scent_options = ["floral", "citrus", "woody", "fresh", "spicy", "fruity"]
        family_options = ["alcohol", "ester", "aldehyde", "ketone", "ether"]
        
        for i in range(100):  # Match molecule count
            desc = {
                "primary_notes": np.random.choice(scent_options, size=2, replace=False).tolist(),
                "intensity": np.random.uniform(1, 10),
                "chemical_family": np.random.choice(family_options),
            }
            mock_descriptions.append(desc)
        
        return mock_descriptions
    
    def _load_sensor_data(self, sensor_data: Union[str, Path, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Load sensor readings."""
        if isinstance(sensor_data, list):
            return sensor_data
        
        # Generate mock sensor data
        mock_sensor_data = []
        for i in range(100):
            data = {
                "gas_sensors": {
                    f"sensor_{j}": np.random.uniform(0, 5) for j in range(8)
                },
                "environmental": {
                    "temperature": np.random.uniform(20, 30),
                    "humidity": np.random.uniform(40, 60),
                }
            }
            mock_sensor_data.append(data)
        
        return mock_sensor_data
    
    def __len__(self) -> int:
        return len(self.molecules)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get training sample."""
        molecule = self.molecules[idx]
        description = self.descriptions[idx]
        
        # Tokenize molecule
        encoded = self.tokenizer.encode(
            molecule,
            max_length=self.max_length,
            padding=True,
            truncation=True
        )
        
        # Extract molecular features
        mol_features = self.tokenizer.extract_molecular_features(molecule)
        mol_feature_vector = list(mol_features.values()) if mol_features else [0.0] * 10
        
        # Prepare labels
        labels = {}
        
        # Scent labels (simplified - would use proper label encoding)
        scent_descriptors = ["floral", "citrus", "woody", "fresh", "spicy", "fruity"]
        primary_note = description["primary_notes"][0] if description["primary_notes"] else "fresh"
        labels["scent_labels"] = torch.tensor(
            scent_descriptors.index(primary_note) if primary_note in scent_descriptors else 0,
            dtype=torch.long
        )
        
        # Intensity labels
        labels["intensity_labels"] = torch.tensor(description["intensity"], dtype=torch.float)
        
        # Chemical family labels
        chemical_families = ["alcohol", "ester", "aldehyde", "ketone", "ether"]
        family = description.get("chemical_family", "alcohol")
        labels["chemical_family_labels"] = torch.tensor(
            chemical_families.index(family) if family in chemical_families else 0,
            dtype=torch.long
        )
        
        sample = {
            "input_ids": torch.tensor(encoded["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(encoded["attention_mask"], dtype=torch.long),
            "molecular_features": torch.tensor(mol_feature_vector, dtype=torch.float),
            **labels
        }
        
        # Add sensor data if available
        if self.sensor_data:
            sensor_values = list(self.sensor_data[idx]["gas_sensors"].values())
            if len(sensor_values) < 64:  # Pad to 64 channels
                sensor_values.extend([0.0] * (64 - len(sensor_values)))
            sample["sensor_data"] = torch.tensor(sensor_values[:64], dtype=torch.float)
        
        return sample
    
    def split(self, test_ratio: float = 0.1) -> Tuple["OlfactoryDataset", "OlfactoryDataset"]:
        """Split dataset into train/test."""
        n_test = int(len(self.molecules) * test_ratio)
        n_train = len(self.molecules) - n_test
        
        # Simple split (would use proper stratification in production)
        train_molecules = self.molecules[:n_train]
        train_descriptions = self.descriptions[:n_train]
        train_sensor_data = self.sensor_data[:n_train] if self.sensor_data else None
        
        test_molecules = self.molecules[n_train:]
        test_descriptions = self.descriptions[n_train:]
        test_sensor_data = self.sensor_data[n_train:] if self.sensor_data else None
        
        train_dataset = OlfactoryDataset(
            train_molecules, train_descriptions, train_sensor_data, self.tokenizer, self.max_length
        )
        test_dataset = OlfactoryDataset(
            test_molecules, test_descriptions, test_sensor_data, self.tokenizer, self.max_length
        )
        
        return train_dataset, test_dataset


class OlfactoryTrainer:
    """Trainer class for olfactory transformer models."""
    
    def __init__(
        self,
        model: OlfactoryTransformer,
        train_dataset: OlfactoryDataset,
        eval_dataset: Optional[OlfactoryDataset] = None,
        args: Optional[TrainingArguments] = None,
        compute_metrics: Optional[Callable] = None,
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.args = args or TrainingArguments()
        self.compute_metrics = compute_metrics
        
        # Setup directories
        self.output_dir = Path(self.args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_eval_loss = float('inf')
        
        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        logging.info(f"Training on device: {self.device}")
    
    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """Create optimizer and learning rate scheduler."""
        # Optimizer
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        
        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.args.learning_rate,
        )
        
        # Scheduler
        if HAS_TRANSFORMERS:
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.args.warmup_steps,
                num_training_steps=num_training_steps,
            )
        else:
            self.scheduler = None
    
    def train(self) -> Dict[str, float]:
        """Main training loop."""
        # Create data loader
        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )
        
        # Calculate total steps
        num_training_steps = len(train_dataloader) * self.args.num_epochs // self.args.gradient_accumulation_steps
        
        # Setup optimizer and scheduler
        self.create_optimizer_and_scheduler(num_training_steps)
        
        # Enable gradient checkpointing if requested
        if self.args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        
        logging.info("***** Running training *****")
        logging.info(f"  Num examples = {len(self.train_dataset)}")
        logging.info(f"  Num epochs = {self.args.num_epochs}")
        logging.info(f"  Batch size = {self.args.batch_size}")
        logging.info(f"  Gradient accumulation steps = {self.args.gradient_accumulation_steps}")
        logging.info(f"  Total optimization steps = {num_training_steps}")
        
        # Training metrics
        total_loss = 0.0
        best_eval_loss = float('inf')
        
        self.model.train()
        
        for epoch in range(self.args.num_epochs):
            self.epoch = epoch
            epoch_loss = 0.0
            
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{self.args.num_epochs}")
            
            for step, batch in enumerate(progress_bar):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(**batch)
                
                # Calculate loss
                loss = self._calculate_loss(outputs, batch)
                
                # Backward pass
                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps
                
                loss.backward()
                
                # Update weights
                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                    
                    self.optimizer.step()
                    if self.scheduler:
                        self.scheduler.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1
                
                # Logging
                epoch_loss += loss.item()
                total_loss += loss.item()
                
                if self.global_step % self.args.logging_steps == 0:
                    avg_loss = total_loss / self.args.logging_steps
                    progress_bar.set_postfix({"loss": f"{avg_loss:.4f}"})
                    total_loss = 0.0
                
                # Evaluation
                if self.eval_dataset and self.global_step % self.args.eval_steps == 0:
                    eval_results = self.evaluate()
                    
                    if eval_results["eval_loss"] < best_eval_loss:
                        best_eval_loss = eval_results["eval_loss"]
                        self.save_model()
                        logging.info(f"New best model saved with eval_loss: {best_eval_loss:.4f}")
                    
                    self.model.train()  # Return to training mode
                
                # Save checkpoint
                if self.global_step % self.args.save_steps == 0:
                    self.save_checkpoint()
            
            # End of epoch evaluation
            if self.eval_dataset:
                eval_results = self.evaluate()
                logging.info(f"Epoch {epoch+1} evaluation: {eval_results}")
        
        # Final save
        self.save_model()
        logging.info("Training completed!")
        
        return {"train_loss": epoch_loss / len(train_dataloader)}
    
    def _calculate_loss(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Calculate weighted multi-task loss."""
        total_loss = 0.0
        
        if "losses" in outputs:
            # Use model's computed losses with weights
            losses = outputs["losses"]
            
            if "scent_loss" in losses:
                total_loss += self.args.scent_loss_weight * losses["scent_loss"]
            
            if "intensity_loss" in losses:
                total_loss += self.args.intensity_loss_weight * losses["intensity_loss"]
            
            if "chemical_family_loss" in losses:
                total_loss += self.args.chemical_family_loss_weight * losses["chemical_family_loss"]
        
        elif "loss" in outputs:
            total_loss = outputs["loss"]
        
        else:
            # Fallback: compute losses manually
            loss_fct = nn.CrossEntropyLoss()
            if "scent_logits" in outputs and "scent_labels" in batch:
                total_loss += loss_fct(outputs["scent_logits"], batch["scent_labels"])
        
        return total_loss
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate model on validation set."""
        if not self.eval_dataset:
            return {}
        
        eval_dataloader = DataLoader(
            self.eval_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )
        
        self.model.eval()
        total_eval_loss = 0.0
        num_eval_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                outputs = self.model(**batch)
                loss = self._calculate_loss(outputs, batch)
                
                total_eval_loss += loss.item()
                num_eval_samples += batch["input_ids"].size(0)
        
        eval_loss = total_eval_loss / len(eval_dataloader)
        
        results = {
            "eval_loss": eval_loss,
            "eval_samples": num_eval_samples,
        }
        
        # Compute additional metrics if provided
        if self.compute_metrics:
            additional_metrics = self.compute_metrics(self.eval_dataset)
            results.update(additional_metrics)
        
        return results
    
    def save_model(self) -> None:
        """Save the trained model."""
        model_path = self.output_dir / "best_model"
        model_path.mkdir(exist_ok=True)
        
        # Save model
        self.model.save_pretrained(model_path)
        
        # Save tokenizer
        if hasattr(self.train_dataset, 'tokenizer'):
            self.train_dataset.tokenizer.save_pretrained(model_path)
        
        # Save training arguments
        with open(model_path / "training_args.json", 'w') as f:
            json.dump(self.args.to_dict(), f, indent=2)
        
        logging.info(f"Model saved to {model_path}")
    
    def save_checkpoint(self) -> None:
        """Save training checkpoint."""
        checkpoint_path = self.output_dir / f"checkpoint-{self.global_step}"
        checkpoint_path.mkdir(exist_ok=True)
        
        # Save model state
        torch.save(self.model.state_dict(), checkpoint_path / "pytorch_model.bin")
        
        # Save optimizer state
        torch.save(self.optimizer.state_dict(), checkpoint_path / "optimizer.pt")
        
        if self.scheduler:
            torch.save(self.scheduler.state_dict(), checkpoint_path / "scheduler.pt")
        
        # Save training state
        training_state = {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_eval_loss": self.best_eval_loss,
        }
        
        with open(checkpoint_path / "trainer_state.json", 'w') as f:
            json.dump(training_state, f, indent=2)
        
        logging.info(f"Checkpoint saved to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: Union[str, Path]) -> None:
        """Load training checkpoint."""
        checkpoint_path = Path(checkpoint_path)
        
        # Load model state
        model_state = torch.load(checkpoint_path / "pytorch_model.bin", map_location=self.device)
        self.model.load_state_dict(model_state)
        
        # Load optimizer state
        optimizer_path = checkpoint_path / "optimizer.pt"
        if optimizer_path.exists():
            optimizer_state = torch.load(optimizer_path, map_location=self.device)
            self.optimizer.load_state_dict(optimizer_state)
        
        # Load scheduler state
        scheduler_path = checkpoint_path / "scheduler.pt"
        if scheduler_path.exists() and self.scheduler:
            scheduler_state = torch.load(scheduler_path, map_location=self.device)
            self.scheduler.load_state_dict(scheduler_state)
        
        # Load training state
        state_path = checkpoint_path / "trainer_state.json"
        if state_path.exists():
            with open(state_path, 'r') as f:
                training_state = json.load(f)
            
            self.global_step = training_state["global_step"]
            self.epoch = training_state["epoch"]
            self.best_eval_loss = training_state["best_eval_loss"]
        
        logging.info(f"Checkpoint loaded from {checkpoint_path}")
    
    def predict(self, test_dataset: OlfactoryDataset) -> List[Dict[str, Any]]:
        """Make predictions on test dataset."""
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
        )
        
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for batch in tqdm(test_dataloader, desc="Predicting"):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                outputs = self.model(**batch)
                
                # Process outputs to predictions
                batch_predictions = self._process_outputs_to_predictions(outputs)
                predictions.extend(batch_predictions)
        
        return predictions
    
    def _process_outputs_to_predictions(self, outputs: Dict[str, torch.Tensor]) -> List[Dict[str, Any]]:
        """Convert model outputs to structured predictions."""
        batch_size = outputs["scent_logits"].size(0)
        predictions = []
        
        for i in range(batch_size):
            pred = {}
            
            # Scent predictions
            if "scent_logits" in outputs:
                scent_probs = torch.softmax(outputs["scent_logits"][i], dim=-1)
                pred["scent_probabilities"] = scent_probs.cpu().numpy().tolist()
                pred["top_scent"] = torch.argmax(scent_probs).item()
            
            # Intensity predictions
            if "intensity" in outputs:
                pred["intensity"] = outputs["intensity"][i].item()
            
            # Chemical family predictions
            if "chemical_family_logits" in outputs:
                family_probs = torch.softmax(outputs["chemical_family_logits"][i], dim=-1)
                pred["chemical_family"] = torch.argmax(family_probs).item()
            
            predictions.append(pred)
        
        return predictions