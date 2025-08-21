"""
Autonomous SDLC enhancement module for progressive quality gates.

This module implements self-improving development patterns and autonomous 
quality gate management for the Olfactory Transformer system.
"""

from .progressive_gates import (
    ProgressiveQualityGates,
    QualityGateResult, 
    AutomatedQualityValidator,
    QualityEvolutionManager
)

from .sdlc_orchestrator import (
    AutonomousSDLCOrchestrator,
    SDLCStage,
    ImplementationGeneration
)

from .self_improving_patterns import (
    SelfImprovingCodebase,
    AdaptiveOptimizer,
    EvolutionaryEnhancer
)

__all__ = [
    "ProgressiveQualityGates",
    "QualityGateResult", 
    "AutomatedQualityValidator",
    "QualityEvolutionManager",
    "AutonomousSDLCOrchestrator",
    "SDLCStage",
    "ImplementationGeneration",
    "SelfImprovingCodebase",
    "AdaptiveOptimizer",
    "EvolutionaryEnhancer"
]