"""Implementation of benchmarks utility function."""

from . import Rastrigin, Rosenbrock, Griewank, Sphere, Ackley

__all__ = ['Utility']


# pylint: disable=too-many-return-statements
class Utility:

    @staticmethod
    def get_benchmark(benchmark, LowerBound=None, UpperBound=None):
        if not isinstance(benchmark, ''.__class__):
            return benchmark
        else:
            if benchmark == 'rastrigin':
                if UpperBound is None and LowerBound is None:
                    return Rastrigin()
                elif UpperBound is not None and LowerBound is not None:
                    return Rastrigin(LowerBound, UpperBound)
                else:
                    raise TypeError('Upper and Lower value must be defined!')
            elif benchmark == 'rosenbrock':
                if UpperBound is None and LowerBound is None:
                    return Rosenbrock()
                elif UpperBound is not None and LowerBound is not None:
                    return Rosenbrock(LowerBound, UpperBound)
                else:
                    raise TypeError('Upper and Lower value must be defined!')
            elif benchmark == 'griewank':
                if UpperBound is None and LowerBound is None:
                    return Griewank()
                elif UpperBound is not None and LowerBound is not None:
                    return Griewank(LowerBound, UpperBound)
                else:
                    raise TypeError('Upper and Lower value must be defined!')
            elif benchmark == 'sphere':
                if UpperBound is None and LowerBound is None:
                    return Sphere()
                elif UpperBound is not None and LowerBound is not None:
                    return Sphere(LowerBound, UpperBound)
                else:
                    raise TypeError('Upper and Lower value must be defined!')
            elif benchmark == 'ackley':
                if UpperBound is None and LowerBound is None:
                    return Ackley()
                elif UpperBound is not None and LowerBound is not None:
                    return Ackley(LowerBound, UpperBound)
                else:
                    raise TypeError('Upper and Lower value must be defined!')
            else:
                raise TypeError('Passed benchmark is not defined!')
