"""Utilities for benchmarks."""
import inspect

from . import Rastrigin, Rosenbrock, Griewank, Sphere, Ackley

__all__ = ['Utility']


class Utility(object):

    @staticmethod
    def initialize_benchmark(function):
        if isinstance(function, object):
            return function.function()
        else:
            if function == 'rastrigin':
                return Rastrigin.function()
            elif function == 'rosenbrock':
                return Rosenbrock.function()
            elif function == 'griewank':
                return Griewank.function()
            elif function == 'sphere':
                return Sphere.function()
            elif function == 'ackley':
                return Ackley.function()
            else:
                raise TypeError('Passed benchmark is not defined!')
