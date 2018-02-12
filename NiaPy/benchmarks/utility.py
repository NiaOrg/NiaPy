"""Utilities for benchmarks."""

from . import Rastrigin, Rosenbrock, Griewank, Sphere, Ackley

__all__ = ['Utility']


class Utility(object):

    @staticmethod
    def get_benchmark(benchmark, Upper=None, Lower=None):
        if not isinstance(benchmark, ''.__class__):
            return benchmark
        else:
            if benchmark == 'rastrigin':
                if Upper == None and Lower == None:
                    return Rastrigin()
                elif Upper != None and Lower != None:
                    return Rastrigin(Lower, Upper)
                else:
                    raise TypeError('Upper and Lower value must be defined!')
            elif benchmark == 'rosenbrock':
                if Upper == None and Lower == None:
                    return Rosenbrock()
                elif Upper != None and Lower != None:
                    return Rosenbrock(Lower, Upper)
                else:
                    raise TypeError('Upper and Lower value must be defined!')
            elif benchmark == 'griewank':
                if Upper == None and Lower == None:
                    return Griewank()
                elif Upper != None and Lower != None:
                    return Griewank(Lower, Upper)
                else:
                    raise TypeError('Upper and Lower value must be defined!')
            elif benchmark == 'sphere':
                if Upper == None and Lower == None:
                    return Sphere()
                elif Upper != None and Lower != None:
                    return Sphere(Lower, Upper)
                else:
                    raise TypeError('Upper and Lower value must be defined!')
            elif benchmark == 'ackley':
                if Upper == None and Lower == None:
                    return Ackley()
                elif Upper != None and Lower != None:
                    return Ackley(Lower, Upper)
                else:
                    raise TypeError('Upper and Lower value must be defined!')
            else:
                raise TypeError('Passed benchmark is not defined!')
