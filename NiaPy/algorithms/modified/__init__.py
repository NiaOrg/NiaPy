# encoding=utf8
# pylint: disable=line-too-long
"""Implementation of modified nature-inspired algorithms."""

from NiaPy.algorithms.modified.hba import HybridBatAlgorithm
from NiaPy.algorithms.modified.hde import DifferentialEvolutionMTS, DifferentialEvolutionMTSv1
from NiaPy.algorithms.modified.jde import SelfAdaptiveDifferentialEvolution, DynNPSelfAdaptiveDifferentialEvolutionAlgorithm, MultiStrategySelfAdaptiveDifferentialEvolution, DynNpMultiStrategySelfAdaptiveDifferentialEvolution

__all__ = [
    'HybridBatAlgorithm',
    'DifferentialEvolutionMTS',
    'DifferentialEvolutionMTSv1',
    'SelfAdaptiveDifferentialEvolution',
    'DynNPSelfAdaptiveDifferentialEvolutionAlgorithm',
    'MultiStrategySelfAdaptiveDifferentialEvolution',
    'DynNpMultiStrategySelfAdaptiveDifferentialEvolution'
]
