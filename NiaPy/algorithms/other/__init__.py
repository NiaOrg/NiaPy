"""Implementation of basic nature-inspired algorithms."""

from NiaPy.algorithms.other.nmm import NelderMeadMethod
from NiaPy.algorithms.other.hc import HillClimbAlgorithm
from NiaPy.algorithms.other.sa import SimulatedAnnealing
from NiaPy.algorithms.other.mts import MultipleTrajectorySearch, MultipleTrajectorySearchV1, MTS_LS1, MTS_LS2, MTS_LS3, MTS_LS1v1, MTS_LS3v1
from NiaPy.algorithms.other.aso import AnarchicSocietyOptimization
from NiaPy.algorithms.other.ts import TabuSearch

__all__ = [
    'NelderMeadMethod',
    'HillClimbAlgorithm',
    'SimulatedAnnealing',
    'MultipleTrajectorySearch',
    'MultipleTrajectorySearchV1',
    'MTS_LS1',
    'MTS_LS2',
    'MTS_LS3',
    'MTS_LS1v1',
    'MTS_LS3v1',
    'AnarchicSocietyOptimization',
    'TabuSearch'
]
