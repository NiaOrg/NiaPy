# encoding=utf8
"""Implementation of modified nature-inspired algorithms."""

from NiaPy.algorithms.modified.hba import HybridBatAlgorithm
from NiaPy.algorithms.modified.jade import (
    AdaptiveArchiveDifferentialEvolution,
    CrossRandCurr2Pbest
)
from NiaPy.algorithms.modified.hde import (
    DifferentialEvolutionMTS,
    DifferentialEvolutionMTSv1,
    DynNpDifferentialEvolutionMTS,
    DynNpDifferentialEvolutionMTSv1,
    MultiStrategyDifferentialEvolutionMTS,
    DynNpMultiStrategyDifferentialEvolutionMTS,
    DynNpMultiStrategyDifferentialEvolutionMTSv1,
    MultiStrategyDifferentialEvolutionMTSv1
)
from NiaPy.algorithms.modified.jde import (
    SelfAdaptiveDifferentialEvolution,
    DynNpSelfAdaptiveDifferentialEvolutionAlgorithm,
    MultiStrategySelfAdaptiveDifferentialEvolution,
    DynNpMultiStrategySelfAdaptiveDifferentialEvolution,
    AgingSelfAdaptiveDifferentialEvolution
)
from NiaPy.algorithms.modified.sade import (
    StrategyAdaptationDifferentialEvolution,
    StrategyAdaptationDifferentialEvolutionV1
)
from NiaPy.algorithms.modified.saba import (
    AdaptiveBatAlgorithm,
    SelfAdaptiveBatAlgorithm
)
from NiaPy.algorithms.modified.hsaba import HybridSelfAdaptiveBatAlgorithm
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
    'AgingSelfAdaptiveDifferentialEvolution',
    'AdaptiveArchiveDifferentialEvolution',
    'CrossRandCurr2Pbest',
    'StrategyAdaptationDifferentialEvolution',
    'StrategyAdaptationDifferentialEvolutionV1',
    'AdaptiveBatAlgorithm',
    'SelfAdaptiveBatAlgorithm',
    'HybridSelfAdaptiveBatAlgorithm'
]
