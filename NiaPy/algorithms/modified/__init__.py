# encoding=utf8
# pylint: disable=line-too-long
"""Implementation of modified nature-inspired algorithms."""

from NiaPy.algorithms.modified.hba import HybridBatAlgorithm
from NiaPy.algorithms.modified.jde import SelfAdaptiveDifferentialEvolutionAlgorithm, DynNPSelfAdaptiveDifferentialEvolutionAlgorithm

__all__ = [
    'HybridBatAlgorithm',
    'SelfAdaptiveDifferentialEvolutionAlgorithm',
    'DynNPSelfAdaptiveDifferentialEvolutionAlgorithm',
]
