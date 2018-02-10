"""Utilities for benchmarks."""

from . import Rastrigin, Rosenbrock, Griewank, Sphere

__all__ = ['Utility']


class Utility(object):

    @staticmethod
    def itialize_benchmark(function):
        if callable(function):
            return function
        else:
            if function == 'rastrigin':
                return Rastrigin.function()
            elif function == 'rosenbrock':
                return Rosenbrock.function()
            elif function == 'griewank':
                return Griewank.function()
            elif function == 'sphere':
                return Sphere.function()
            else:
                raise TypeError('Passed benchmark is not defined!')
