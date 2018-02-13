"""Implementation of benchmarks utility function."""

from . import Rastrigin, Rosenbrock, Griewank, Sphere, Ackley

__all__ = ['Utility']


class Utility(object):

    @staticmethod
    def get_benchmark(benchmark, Upper=None, Lower=None):
        if not isinstance(benchmark, ''.__class__):
            return benchmark
        else:
            returnBenchmark = None
            if benchmark == 'rastrigin':
                if Upper is None and Lower is None:
                    returnBenchmark = Rastrigin()
                elif Upper is not None and Lower is not None:
                    returnBenchmark = Rastrigin(Lower, Upper)
                else:
                    raise TypeError('Upper and Lower value must be defined!')
            elif benchmark == 'rosenbrock':
                if Upper is None and Lower is None:
                    returnBenchmark = Rosenbrock()
                elif Upper is not None and Lower is not None:
                    returnBenchmark = Rosenbrock(Lower, Upper)
                else:
                    raise TypeError('Upper and Lower value must be defined!')
            elif benchmark == 'griewank':
                if Upper is None and Lower is None:
                    returnBenchmark = Griewank()
                elif Upper is not None and Lower is not None:
                    returnBenchmark = Griewank(Lower, Upper)
                else:
                    raise TypeError('Upper and Lower value must be defined!')
            elif benchmark == 'sphere':
                if Upper is None and Lower is None:
                    returnBenchmark = Sphere()
                elif Upper is not None and Lower is not None:
                    returnBenchmark = Sphere(Lower, Upper)
                else:
                    raise TypeError('Upper and Lower value must be defined!')
            elif benchmark == 'ackley':
                if Upper is None and Lower is None:
                    returnBenchmark = Ackley()
                elif Upper is not None and Lower is not None:
                    returnBenchmark = Ackley(Lower, Upper)
                else:
                    raise TypeError('Upper and Lower value must be defined!')
            else:
                raise TypeError('Passed benchmark is not defined!')

            return returnBenchmark
