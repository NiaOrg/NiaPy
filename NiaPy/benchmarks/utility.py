"""Utilities for benchmarks."""

from . import Rastrigin, Rosenbrock, Griewank, Sphere, Ackley, Schwefel, Whitley

__all__ = ['Utility']


class Utility(object):

    # pylint: disable=too-many-return-statements
    @staticmethod
    def get_benchmark(benchmark):
        if not isinstance(benchmark, ''.__class__):
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
            elif benchmark == 'schwefel':
                return Schwefel()
            elif benchmark == 'whitley':
                return Whitley()
            else:
                raise TypeError('Passed benchmark is not defined!')
