"""Implementation of modified nature-inspired algorithms."""

from NiaPy.algorithms.modified.hba import HybridBatAlgorithm
from NiaPy.algorithms.modified.jde import SelfAdaptiveDifferentialEvolutionAlgorithm, DynNPSelfAdaptiveDifferentialEvolutionAlgorithm, SelfAdaptiveDifferentialEvolutionAlgorithmBestHarmonySearch, SelfAdaptiveDifferentialEvolutionAlgorithmBestMTS1, SelfAdaptiveDifferentialEvolutionAlgorithmBestMTS2, SelfAdaptiveDifferentialEvolutionAlgorithmBestMTS3
from NiaPy.algorithms.modified.hde import DifferentialEvolutionBestSimulatedAnnealing, DifferentialEvolutionBestHarmonySearch, DifferentialEvolutionPBestHarmonySearch, DifferentialEvolutionBestMTS1, DifferentialEvolutionBestMTS2, DifferentialEvolutionBestMTS3

__all__ = [
    'HybridBatAlgorithm',
    'SelfAdaptiveDifferentialEvolutionAlgorithm',
    'DynNPSelfAdaptiveDifferentialEvolutionAlgorithm',
    'SelfAdaptiveDifferentialEvolutionAlgorithmBestHarmonySearch',
    'SelfAdaptiveDifferentialEvolutionAlgorithmBestMTS1',
    'SelfAdaptiveDifferentialEvolutionAlgorithmBestMTS2',
    'SelfAdaptiveDifferentialEvolutionAlgorithmBestMTS3',
    'DifferentialEvolutionBestSimulatedAnnealing',
    'DifferentialEvolutionBestHarmonySearch',
    'DifferentialEvolutionPBestHarmonySearch',
    'DifferentialEvolutionBestMTS1',
    'DifferentialEvolutionBestMTS2',
    'DifferentialEvolutionBestMTS3'
]
