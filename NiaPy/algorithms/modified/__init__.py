"""Implementation of modified nature-inspired algorithms."""

from NiaPy.algorithms.modified.hba import HybridBatAlgorithm
from NiaPy.algorithms.modified.jde import SelfAdaptiveDifferentialEvolutionAlgorithm
from NiaPy.algorithms.modified.dynnpjde import DynNPSelfAdaptiveDifferentialEvolutionAlgorithm

__all__ = [
    'HybridBatAlgorithm',
    'SelfAdaptiveDifferentialEvolutionAlgorithm',
    'DynNPSelfAdaptiveDifferentialEvolutionAlgorithm'
]
