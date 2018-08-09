"""Implementation of basic nature-inspired algorithms."""
# pylint: disable=line-too-long

from NiaPy.algorithms.other.nmm import NelderMeadMethod
from NiaPy.algorithms.other.ihc import HillClimbAlgorithm
from NiaPy.algorithms.other.sa import SimulatedAnnealing

__all__ = [
    'NelderMeadMethod',
    'HillClimbAlgorithm',
    'SimulatedAnnealing'
]
