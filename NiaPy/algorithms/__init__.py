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

__all__ = [
    'Individual',
    'defaultNumPyInit',
    'defaultIndividualInit',
    'Algorithm',
    'basic',
    'modified',
    'other'
]
