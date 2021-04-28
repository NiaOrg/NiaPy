# encoding=utf8

"""Module with implementations of basic and hybrid algorithms."""

from niapy.algorithms import basic
from niapy.algorithms import modified
from niapy.algorithms import other
from niapy.algorithms.algorithm import Algorithm, Individual, defaultNumPyInit, defaultIndividualInit
from niapy.algorithms.statistics import BasicStatistics
from niapy.algorithms.utility import AlgorithmUtility

__all__ = [
	'basic',
	'modified',
	'other',
	'Algorithm',
	'defaultNumPyInit',
	'defaultIndividualInit',
	'Individual',
	'BasicStatistics',
	'AlgorithmUtility'
]
