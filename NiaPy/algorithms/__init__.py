# encoding=utf8
# pylint: disable=bad-continuation
"""Module with implementations of basic and hybrid algorithms."""

from NiaPy.algorithms import basic
from NiaPy.algorithms import modified
from NiaPy.algorithms import other
from NiaPy.algorithms.algorithm import Algorithm, Individual, defaultNumPyInit, defaultIndividualInit
from NiaPy.algorithms.statistics import BasicStatistics

__all__ = [
	'basic',
	'modified',
	'other',
	'Algorithm',
	'defaultNumPyInit',
	'defaultIndividualInit',
	'Individual',
	'BasicStatistics'
]
