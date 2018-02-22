"""Implementation of benchmarks utility function."""

from . import Rastrigin, Rosenbrock, Griewank, \
    Sphere, Ackley, Schwefel, Schwefel221, \
    Schwefel222, Whitley, Alpine1, Alpine2, HappyCat, \
    Ridge, ChungReynolds, Csendes, Pinter, Qing, Quintic, \
    Salomon, SchumerSteiglitz, Step, Step2, Step3, Stepint, SumSquares, StyblinskiTang


__all__ = ['Utility']


class Utility(object):

    def __init__(self):
        self.classes = {
            'ackley': Ackley,
            'alpine1': Alpine1,
            'alpine2': Alpine2,
            'chungReynolds': ChungReynolds,
            'csendes': Csendes,
            'griewank': Griewank,
            'happyCat': HappyCat,
            'pinter': Pinter,
            'quing': Qing,
            'quintic': Quintic,
            'rastrigin': Rastrigin,
            'ridge': Ridge,
            'rosenbrock': Rosenbrock,
            'salomon': Salomon,
            'schumerSteiglitz': SchumerSteiglitz,
            'schwefel': Schwefel,
            'schwefel221': Schwefel221,
            'schwefel222': Schwefel222,
            'sphere': Sphere,
            'step': Step,
            'step2': Step2,
            'step3': Step3,
            'stepint': Stepint,
            'styblinskiTang': StyblinskiTang,
            'sumSquares': SumSquares,
            'whitley': Whitley
        }

    def get_benchmark(self, benchmark):
        if not isinstance(benchmark, ''.__class__):
            return benchmark
        else:
            if benchmark in self.classes:
                return self.classes[benchmark]()
            else:
                raise TypeError('Passed benchmark is not defined!')

    @classmethod
    def __raiseLowerAndUpperNotDefined(cls):
        raise TypeError('Upper and Lower value must be defined!')
