# encoding=utf8

"""Module with implementations of basic and hybrid algorithms."""

from niapy.algorithms import basic
from niapy.algorithms import modified
from niapy.algorithms import other
from niapy.algorithms.algorithm import Algorithm, Individual, default_numpy_init, default_individual_init

__all__ = [
    'basic',
    'modified',
    'other',
    'Algorithm',
    'default_numpy_init',
    'default_individual_init',
    'Individual',
]
