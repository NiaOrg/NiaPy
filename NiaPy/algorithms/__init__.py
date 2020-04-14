# encoding=utf8

"""Module with implementations of basic and hybrid algorithms."""

from NiaPy.algorithms import basic
from NiaPy.algorithms import modified
from NiaPy.algorithms import other
from NiaPy.algorithms.algorithm import Algorithm
from NiaPy.algorithms.individual import (
    Individual,
    defaultNumPyInit,
    defaultIndividualInit
)
from NiaPy.algorithms.statistics import BasicStatistics
from NiaPy.algorithms.utility import AlgorithmUtility

__all__ = [
    'Individual',
    'defaultNumPyInit',
    'defaultIndividualInit',
    'Algorithm',
    'AlgorithmUtility',
    'basic',
    'modified',
    'other',
    'BasicStatistics'
]
