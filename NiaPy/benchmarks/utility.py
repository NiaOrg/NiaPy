"""Utilities for benchmarks."""
import inspect

from . import Rastrigin, Rosenbrock, Griewank, Sphere, Ackley

__all__ = ['Utility']


class Utility(object):

    @staticmethod
    def get_benchmark(benchmark):
        if not isinstance(benchmark, basestring):
            return benchmark
        else:
            if benchmark == 'rastrigin':
                return Rastrigin()
            elif benchmark == 'rosenbrock':
                return Rosenbrock()
            elif benchmark == 'griewank':
                return Griewank()
            elif benchmark == 'sphere':
                return Sphere()
            elif benchmark == 'ackley':
                return Ackley()
            else:
                raise TypeError('Passed benchmark is not defined!')
