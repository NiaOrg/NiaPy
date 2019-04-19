# encoding=utf8
# pylint: disable=bad-continuation
"""Module with implementations of basic and hybrid algorithms."""

from NiaPy.algorithms import basic
from NiaPy.algorithms import modified
from NiaPy.algorithms import other
from NiaPy.algorithms.algorithm import Algorithm, Individual, default_numpy_init, default_individual_init
from NiaPy.algorithms.statistics import BasicStatistics

__all__ = [
	'basic',
	'modified',
	'other',
	'Algorithm',
	'default_numpy_init',
	'default_individual_init',
	'Individual',
	'BasicStatistics'
]
