"""Implementation of other algorithms."""

from niapy.algorithms.other.nmm import NelderMeadMethod
from niapy.algorithms.other.hc import HillClimbAlgorithm
from niapy.algorithms.other.sa import SimulatedAnnealing
from niapy.algorithms.other.mts import MultipleTrajectorySearch, MultipleTrajectorySearchV1, MTS_LS1, MTS_LS2, MTS_LS3, MTS_LS1v1, MTS_LS3v1
from niapy.algorithms.other.aso import AnarchicSocietyOptimization
from niapy.algorithms.other.rs import RandomSearch

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
    'RandomSearch'
]
