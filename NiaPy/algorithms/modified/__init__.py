# encoding=utf8
# pylint: disable=line-too-long
"""Implementation of modified nature-inspired algorithms."""

from NiaPy.algorithms.modified.hba import HybridBatAlgorithm
from NiaPy.algorithms.modified.jde import SelfAdaptiveDifferentialEvolutionAlgorithm, DynNPSelfAdaptiveDifferentialEvolutionAlgorithm, SelfAdaptiveDifferentialEvolutionAlgorithmBestMTS1, SelfAdaptiveDifferentialEvolutionAlgorithmBestMTS2, SelfAdaptiveDifferentialEvolutionAlgorithmBestMTS3, SelfAdaptiveDifferentialEvolutionAlgorithmBestSimulatedAnnealing
from NiaPy.algorithms.modified.hde import DifferentialEvolutionBestSimulatedAnnealing, DifferentialEvolutionBestMTS1, DifferentialEvolutionBestMTS2, DifferentialEvolutionBestMTS3

__all__ = [
    'HybridBatAlgorithm',
    'SelfAdaptiveDifferentialEvolutionAlgorithm',
    'DynNPSelfAdaptiveDifferentialEvolutionAlgorithm',
    'SelfAdaptiveDifferentialEvolutionAlgorithmBestSimulatedAnnealing',
    'SelfAdaptiveDifferentialEvolutionAlgorithmBestMTS1',
    'SelfAdaptiveDifferentialEvolutionAlgorithmBestMTS2',
    'SelfAdaptiveDifferentialEvolutionAlgorithmBestMTS3',
    'DifferentialEvolutionBestSimulatedAnnealing',
    'DifferentialEvolutionBestMTS1',
    'DifferentialEvolutionBestMTS2',
    'DifferentialEvolutionBestMTS3'
]
