"""Inverse molecular design for target scent profiles."""

from typing import Dict, List, Optional, Any, Tuple, Union
import logging
import random
from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Optional imports
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Crippen
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False

from ..core.config import OlfactoryConfig


@dataclass
class MoleculeCandidate:
    """Container for generated molecule candidates."""
    smiles: str
    profile_match: float
    sa_score: float  # Synthetic accessibility score
    molecular_weight: float
    logp: float
    properties: Dict[str, Any]
    
    def visualize(self) -> None:
        """Visualize molecule structure."""
        if HAS_RDKIT:
            mol = Chem.MolFromSmiles(self.smiles)
            if mol:
                print(f"Molecule: {self.smiles}")
                print(f"MW: {self.molecular_weight:.2f}, LogP: {self.logp:.2f}")
                # In a real implementation, would display 2D structure
            else:
                print(f"Invalid SMILES: {self.smiles}")
        else:
            print(f"Molecule: {self.smiles} (MW: {self.molecular_weight:.2f})")


@dataclass 
class TargetProfile:
    """Target scent profile for molecular design."""
    notes: List[str]
    intensity: float
    longevity: str
    character: str
    constraints: Optional[Dict[str, Any]] = None


class MolecularGenerator(nn.Module):
    """Neural network for generating molecular structures."""
    
    def __init__(self, vocab_size: int, hidden_size: int = 512, num_layers: int = 3):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.output = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x: torch.Tensor, hidden: Optional[Tuple] = None) -> Tuple[torch.Tensor, Tuple]:
        """Forward pass through generator."""
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)
        
        output, hidden = self.lstm(embedded, hidden)
        output = self.dropout(output)
        logits = self.output(output)
        
        return logits, hidden
    
    def generate_sequence(
        self, 
        start_token: int,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> List[int]:
        """Generate molecular sequence."""
        self.eval()
        generated = [start_token]
        hidden = None
        
        with torch.no_grad():
            for _ in range(max_length - 1):
                input_token = torch.tensor([[generated[-1]]])
                logits, hidden = self.forward(input_token, hidden)
                
                # Apply temperature
                logits = logits[0, -1] / temperature
                
                # Top-k sampling
                if top_k > 0:
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = float('-inf')
                
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
                
                generated.append(next_token)
                
                # Stop at end token (would be defined in vocabulary)
                if next_token == 0:  # Assuming 0 is end token
                    break
        
        return generated


class ScentDesigner:
    """AI-powered molecular designer for target scent profiles."""
    
    def __init__(self, model: Any, config: Optional[OlfactoryConfig] = None):
        self.model = model  # OlfactoryTransformer
        self.config = config or OlfactoryConfig()
        
        # Initialize molecular generator
        self.generator = MolecularGenerator(
            vocab_size=self.config.vocab_size,
            hidden_size=512,
            num_layers=3
        )
        
        # SMILES building blocks for design
        self.functional_groups = {
            'floral': ['C1=CC=C(C=C1)CO', 'CC(C)CC1=CC=C(C=C1)C(C)C'],  # Benzyl alcohol, lily aldehyde
            'citrus': ['CC1=CC(=O)CC(C1)C', 'C=C(C)C1CCC(CC1)C'],  # Limonene variants
            'woody': ['CC1=C(C(CCC1)(C)C)C', 'C1=CC=C2C(=C1)C=CC3=CC=CC=C32'],  # Cedrol, naphthalene
            'rose': ['CCCCCCCC/C=C/C(=O)[O-]', 'CC(C)CCCC(C)CCCC(C)CCCC(C)C'],  # Rose oxide variants
            'vanilla': ['COC1=CC(=CC=C1O)C=O', 'COC1=CC(=CC=C1)C=O'],  # Vanillin variants
            'musky': ['CCCCCCCCCCCCCCCC', 'CC1CCCC2C1CCCC2C'],  # Musk ketone variants
        }
        
        # Molecular constraints
        self.default_constraints = {
            "molecular_weight": (150, 400),
            "logp": (1, 6),
            "num_rings": (0, 4),
            "rotatable_bonds": (0, 10),
        }
    
    def design_molecules(
        self,
        target_profile: Union[Dict[str, Any], TargetProfile],
        n_candidates: int = 10,
        molecular_weight: Tuple[float, float] = (200, 350),
        logp: Tuple[float, float] = (2, 5),
        method: str = "genetic",
    ) -> List[MoleculeCandidate]:
        """Design molecules matching target scent profile."""
        
        if isinstance(target_profile, dict):
            target = TargetProfile(**target_profile)
        else:
            target = target_profile
        
        logging.info(f"Designing molecules for profile: {target.notes}")
        
        candidates = []
        
        if method == "genetic":
            candidates = self._genetic_algorithm_design(target, n_candidates, molecular_weight, logp)
        elif method == "neural":
            candidates = self._neural_generation_design(target, n_candidates, molecular_weight, logp)
        else:
            candidates = self._template_based_design(target, n_candidates, molecular_weight, logp)
        
        # Rank candidates by profile match
        candidates.sort(key=lambda x: x.profile_match, reverse=True)
        
        return candidates[:n_candidates]
    
    def _genetic_algorithm_design(
        self, 
        target: TargetProfile, 
        n_candidates: int,
        mw_range: Tuple[float, float],
        logp_range: Tuple[float, float]
    ) -> List[MoleculeCandidate]:
        """Use genetic algorithm for molecular design."""
        
        # Initialize population with template molecules
        population = self._initialize_population(target, size=50)
        
        for generation in range(20):  # Run for 20 generations
            # Evaluate fitness
            fitness_scores = []
            for smiles in population:
                score = self._evaluate_fitness(smiles, target, mw_range, logp_range)
                fitness_scores.append(score)
            
            # Selection (top 50%)
            sorted_indices = np.argsort(fitness_scores)[::-1]
            population = [population[i] for i in sorted_indices[:25]]
            
            # Crossover and mutation
            new_population = population.copy()
            while len(new_population) < 50:
                parent1, parent2 = random.sample(population[:10], 2)
                child = self._crossover_molecules(parent1, parent2)
                if child:
                    child = self._mutate_molecule(child)
                    new_population.append(child)
            
            population = new_population
        
        # Convert top candidates
        candidates = []
        for smiles in population[:n_candidates]:
            candidate = self._smiles_to_candidate(smiles, target)
            if candidate:
                candidates.append(candidate)
        
        return candidates
    
    def _neural_generation_design(
        self, 
        target: TargetProfile, 
        n_candidates: int,
        mw_range: Tuple[float, float],
        logp_range: Tuple[float, float]
    ) -> List[MoleculeCandidate]:
        """Use neural generation for molecular design."""
        
        candidates = []
        attempts = 0
        max_attempts = n_candidates * 10
        
        while len(candidates) < n_candidates and attempts < max_attempts:
            # Generate molecule using neural model
            generated_tokens = self.generator.generate_sequence(
                start_token=1,  # Start token
                max_length=100,
                temperature=0.8,
                top_k=50
            )
            
            # Convert tokens to SMILES (simplified)
            smiles = self._tokens_to_smiles(generated_tokens)
            
            if smiles and self._is_valid_molecule(smiles):
                candidate = self._smiles_to_candidate(smiles, target)
                if candidate and self._meets_constraints(candidate, mw_range, logp_range):
                    candidates.append(candidate)
            
            attempts += 1
        
        return candidates
    
    def _template_based_design(
        self, 
        target: TargetProfile, 
        n_candidates: int,
        mw_range: Tuple[float, float],
        logp_range: Tuple[float, float]
    ) -> List[MoleculeCandidate]:
        """Use template-based design approach."""
        
        candidates = []
        
        # Select relevant functional groups
        relevant_groups = []
        for note in target.notes:
            if note.lower() in self.functional_groups:
                relevant_groups.extend(self.functional_groups[note.lower()])
        
        if not relevant_groups:
            # Use random functional groups
            all_groups = []
            for groups in self.functional_groups.values():
                all_groups.extend(groups)
            relevant_groups = random.sample(all_groups, min(5, len(all_groups)))
        
        # Generate variations
        for i in range(n_candidates * 2):  # Generate extra to filter
            base_structure = random.choice(relevant_groups)
            
            # Apply modifications
            modified_smiles = self._modify_structure(base_structure)
            
            if modified_smiles and self._is_valid_molecule(modified_smiles):
                candidate = self._smiles_to_candidate(modified_smiles, target)
                if candidate and self._meets_constraints(candidate, mw_range, logp_range):
                    candidates.append(candidate)
        
        return candidates[:n_candidates]
    
    def _initialize_population(self, target: TargetProfile, size: int) -> List[str]:
        """Initialize population for genetic algorithm."""
        population = []
        
        # Start with template molecules
        for note in target.notes:
            if note.lower() in self.functional_groups:
                templates = self.functional_groups[note.lower()]
                population.extend(templates[:2])  # Take first 2 templates per note
        
        # Fill rest with random valid molecules
        common_molecules = [
            "CCO",  # Ethanol
            "CC(C)O",  # Isopropanol  
            "C1=CC=CC=C1",  # Benzene
            "CC(=O)OCC",  # Ethyl acetate
            "C1=CC=C(C=C1)C=O",  # Benzaldehyde
        ]
        
        while len(population) < size:
            population.append(random.choice(common_molecules))
        
        return population[:size]
    
    def _evaluate_fitness(
        self, 
        smiles: str, 
        target: TargetProfile,
        mw_range: Tuple[float, float],
        logp_range: Tuple[float, float]
    ) -> float:
        """Evaluate fitness of molecule for target profile."""
        if not self._is_valid_molecule(smiles):
            return 0.0
        
        fitness = 0.0
        
        # Get predicted scent profile
        try:
            from ..core.tokenizer import MoleculeTokenizer
            tokenizer = MoleculeTokenizer()  # Would use pre-trained tokenizer
            prediction = self.model.predict_scent(smiles, tokenizer)
            
            # Score based on note overlap
            note_overlap = len(set(prediction.primary_notes) & set(target.notes))
            fitness += note_overlap * 20  # 20 points per matching note
            
            # Score based on intensity match
            intensity_diff = abs(prediction.intensity - target.intensity)
            fitness += max(0, 10 - intensity_diff)  # Up to 10 points for intensity
            
        except Exception as e:
            logging.warning(f"Failed to predict scent for {smiles}: {e}")
            fitness = 1.0  # Minimal fitness for valid molecules
        
        # Penalize molecules outside constraints
        if not self._meets_constraints_simple(smiles, mw_range, logp_range):
            fitness *= 0.1  # Heavy penalty
        
        return fitness
    
    def _crossover_molecules(self, parent1: str, parent2: str) -> Optional[str]:
        """Crossover two parent molecules."""
        # Simplified crossover - would use proper molecular crossover in real implementation
        if random.random() < 0.5:
            return parent1
        else:
            return parent2
    
    def _mutate_molecule(self, smiles: str) -> str:
        """Apply random mutation to molecule."""
        # Simplified mutation - would use proper molecular mutations in real implementation
        mutations = [
            "C",   # Add carbon
            "O",   # Add oxygen
            "N",   # Add nitrogen
            "CC",  # Add ethyl
        ]
        
        if random.random() < 0.1:  # 10% chance of mutation
            mutation = random.choice(mutations)
            # Simple concatenation (not chemically valid, but for demo)
            return smiles + mutation
        
        return smiles
    
    def _modify_structure(self, base_smiles: str) -> str:
        """Apply modifications to base structure."""
        # Simplified structure modification
        modifications = ["C", "O", "CC", "(C)"]
        
        if random.random() < 0.3:
            mod = random.choice(modifications)
            return base_smiles + mod
        
        return base_smiles
    
    def _tokens_to_smiles(self, tokens: List[int]) -> str:
        """Convert token sequence to SMILES string."""
        # Simplified conversion - would use proper tokenizer in real implementation
        smiles_chars = "CONSPcnos()=+-#[]@"
        
        smiles = ""
        for token in tokens:
            if 0 < token < len(smiles_chars):
                smiles += smiles_chars[token]
        
        return smiles
    
    def _is_valid_molecule(self, smiles: str) -> bool:
        """Check if SMILES represents valid molecule."""
        if not smiles or len(smiles) < 2:
            return False
        
        if HAS_RDKIT:
            try:
                mol = Chem.MolFromSmiles(smiles)
                return mol is not None
            except:
                return False
        else:
            # Basic validation without RDKit
            return len(smiles) > 2 and not any(char in smiles for char in ["!", "@", "#"] if char not in "C(=O)")
    
    def _smiles_to_candidate(self, smiles: str, target: TargetProfile) -> Optional[MoleculeCandidate]:
        """Convert SMILES to molecule candidate."""
        try:
            # Calculate properties
            if HAS_RDKIT:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    return None
                
                mw = Descriptors.MolWt(mol)
                logp = Descriptors.MolLogP(mol)
                sa_score = 3.0  # Placeholder - would calculate real SA score
            else:
                # Estimate properties without RDKit
                mw = len(smiles) * 12.0  # Rough estimate
                logp = 2.5  # Default value
                sa_score = 3.0
            
            # Calculate profile match
            try:
                from ..core.tokenizer import MoleculeTokenizer
                tokenizer = MoleculeTokenizer()
                prediction = self.model.predict_scent(smiles, tokenizer)
                
                # Calculate match score
                note_overlap = len(set(prediction.primary_notes) & set(target.notes))
                intensity_match = 1.0 - abs(prediction.intensity - target.intensity) / 10.0
                profile_match = (note_overlap / max(1, len(target.notes)) + intensity_match) / 2.0
                
            except Exception:
                profile_match = 0.5  # Default match score
            
            return MoleculeCandidate(
                smiles=smiles,
                profile_match=profile_match,
                sa_score=sa_score,
                molecular_weight=mw,
                logp=logp,
                properties={"valid": True}
            )
            
        except Exception as e:
            logging.warning(f"Failed to create candidate from {smiles}: {e}")
            return None
    
    def _meets_constraints(
        self, 
        candidate: MoleculeCandidate,
        mw_range: Tuple[float, float],
        logp_range: Tuple[float, float]
    ) -> bool:
        """Check if candidate meets constraints."""
        return (
            mw_range[0] <= candidate.molecular_weight <= mw_range[1] and
            logp_range[0] <= candidate.logp <= logp_range[1]
        )
    
    def _meets_constraints_simple(
        self, 
        smiles: str,
        mw_range: Tuple[float, float],
        logp_range: Tuple[float, float]
    ) -> bool:
        """Simple constraint check for SMILES."""
        if HAS_RDKIT:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    return False
                
                mw = Descriptors.MolWt(mol)
                logp = Descriptors.MolLogP(mol)
                
                return (
                    mw_range[0] <= mw <= mw_range[1] and
                    logp_range[0] <= logp <= logp_range[1]
                )
            except:
                return False
        else:
            # Rough estimate without RDKit
            est_mw = len(smiles) * 12.0
            return mw_range[0] <= est_mw <= mw_range[1]
    
    def optimize_formulation(
        self,
        molecules: List[MoleculeCandidate],
        target_profile: TargetProfile,
        max_components: int = 5
    ) -> Dict[str, float]:
        """Optimize blend formulation from molecule candidates."""
        
        # Select top molecules
        top_molecules = sorted(molecules, key=lambda x: x.profile_match, reverse=True)[:max_components]
        
        # Simple blend optimization (would use proper optimization in real implementation)
        total_weight = sum(mol.profile_match for mol in top_molecules)
        
        formulation = {}
        for mol in top_molecules:
            weight_fraction = mol.profile_match / total_weight
            formulation[mol.smiles] = weight_fraction
        
        return formulation
    
    def save_design_session(self, candidates: List[MoleculeCandidate], file_path: Union[str, Path]) -> None:
        """Save design session results."""
        data = {
            "candidates": [
                {
                    "smiles": c.smiles,
                    "profile_match": c.profile_match,
                    "sa_score": c.sa_score,
                    "molecular_weight": c.molecular_weight,
                    "logp": c.logp,
                    "properties": c.properties,
                }
                for c in candidates
            ]
        }
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logging.info(f"Design session saved to {file_path}")


class AIPerfumer:
    """AI perfumer assistant for fragrance development."""
    
    def __init__(self, model: Any):
        self.model = model
        self.designer = ScentDesigner(model)
        
        # Perfumery knowledge base
        self.fragrance_families = {
            "fresh": ["citrus", "marine", "green"],
            "floral": ["rose", "jasmine", "lily", "violet"],
            "oriental": ["vanilla", "amber", "spices", "resin"],
            "woody": ["sandalwood", "cedar", "vetiver", "patchouli"],
        }
        
        self.concentration_types = {
            "parfum": {"dilution": 0.15, "longevity": "8-12h"},
            "eau_de_parfum": {"dilution": 0.08, "longevity": "6-8h"},
            "eau_de_toilette": {"dilution": 0.05, "longevity": "3-5h"},
            "eau_de_cologne": {"dilution": 0.03, "longevity": "2-3h"},
        }
    
    def create_fragrance(
        self,
        brief: str,
        concentration: str = "eau_de_parfum",
        price_range: str = "premium",
        regulatory_compliance: List[str] = None
    ) -> Any:
        """Create new fragrance from brief."""
        
        # Parse brief (simplified NLP)
        notes = self._extract_notes_from_brief(brief)
        character = self._extract_character_from_brief(brief)
        
        # Create target profile
        target_profile = TargetProfile(
            notes=notes,
            intensity=7.0,  # Default intensity
            longevity="long-lasting",
            character=character
        )
        
        # Design molecules
        molecules = self.designer.design_molecules(target_profile, n_candidates=20)
        
        # Create formulation
        formulation = self.designer.optimize_formulation(molecules, target_profile)
        
        # Create fragrance formula
        formula = FragranceFormula(
            name=f"AI Generated Fragrance",
            concentration=concentration,
            formulation=formulation,
            molecules=molecules,
            compliance=regulatory_compliance or [],
        )
        
        return formula
    
    def _extract_notes_from_brief(self, brief: str) -> List[str]:
        """Extract scent notes from text brief."""
        # Simplified keyword extraction
        keywords = ["citrus", "floral", "woody", "fresh", "sweet", "spicy", "rose", "jasmine", "cedar", "vanilla"]
        
        found_notes = []
        brief_lower = brief.lower()
        
        for keyword in keywords:
            if keyword in brief_lower:
                found_notes.append(keyword)
        
        return found_notes or ["fresh", "floral"]  # Default notes
    
    def _extract_character_from_brief(self, brief: str) -> str:
        """Extract character description from brief."""
        # Simplified character extraction
        if "modern" in brief.lower():
            return "modern, minimalist"
        elif "classic" in brief.lower():
            return "classic, elegant"
        else:
            return "contemporary, sophisticated"


@dataclass
class FragranceFormula:
    """Container for fragrance formulation."""
    name: str
    concentration: str
    formulation: Dict[str, float]
    molecules: List[MoleculeCandidate]
    compliance: List[str]
    
    def export_to_perfumers_workbench(self) -> str:
        """Export formula to perfumer's workbench format."""
        output = f"Fragrance: {self.name}\n"
        output += f"Concentration: {self.concentration}\n\n"
        output += "Formulation:\n"
        
        for smiles, percentage in self.formulation.items():
            output += f"  {smiles}: {percentage:.2%}\n"
        
        output += f"\nCompliance: {', '.join(self.compliance)}\n"
        
        return output