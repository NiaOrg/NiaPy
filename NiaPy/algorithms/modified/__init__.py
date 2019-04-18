# encoding=utf8
# pylint: disable=line-too-long
"""Implementation of modified nature-inspired algorithms."""

from NiaPy.algorithms.modified.hba import HybridBatAlgorithm
from NiaPy.algorithms.modified.hde import DifferentialEvolutionMTS, DifferentialEvolutionMTSv1, DynNpDifferentialEvolutionMTS, DynNpDifferentialEvolutionMTSv1, MultiStrategyDifferentialEvolutionMTS, DynNpMultiStrategyDifferentialEvolutionMTS, DynNpMultiStrategyDifferentialEvolutionMTSv1, MultiStrategyDifferentialEvolutionMTSv1
from NiaPy.algorithms.modified.jde import SelfAdaptiveDifferentialEvolution, DynNpSelfAdaptiveDifferentialEvolutionAlgorithm, MultiStrategySelfAdaptiveDifferentialEvolution, DynNpMultiStrategySelfAdaptiveDifferentialEvolution, AgingSelfAdaptiveDifferentialEvolution

__all__ = [
    'HybridBatAlgorithm',
    'DifferentialEvolutionMTS',
    'DifferentialEvolutionMTSv1',
    'DynNpDifferentialEvolutionMTS',
    'DynNpDifferentialEvolutionMTSv1',
    'MultiStrategyDifferentialEvolutionMTS',
    'MultiStrategyDifferentialEvolutionMTSv1',
    'DynNpMultiStrategyDifferentialEvolutionMTS',
    'DynNpMultiStrategyDifferentialEvolutionMTSv1',
    'SelfAdaptiveDifferentialEvolution',
    'DynNpSelfAdaptiveDifferentialEvolutionAlgorithm',
    'MultiStrategySelfAdaptiveDifferentialEvolution',
    'DynNpMultiStrategySelfAdaptiveDifferentialEvolution',
    'AgingSelfAdaptiveDifferentialEvolution'
]
