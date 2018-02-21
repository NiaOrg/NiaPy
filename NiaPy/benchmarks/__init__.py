"""Module with implementations of benchmark functions."""

from NiaPy.benchmarks.rastrigin import Rastrigin
from NiaPy.benchmarks.rosenbrock import Rosenbrock
from NiaPy.benchmarks.griewank import Griewank
from NiaPy.benchmarks.sphere import Sphere
from NiaPy.benchmarks.ackley import Ackley
from NiaPy.benchmarks.schwefel import Schwefel
from NiaPy.benchmarks.schwefel import Schwefel221
from NiaPy.benchmarks.schwefel import Schwefel222
from NiaPy.benchmarks.whitley import Whitley
from NiaPy.benchmarks.alpine import Alpine1
from NiaPy.benchmarks.alpine import Alpine2
from NiaPy.benchmarks.happyCat import HappyCat
from NiaPy.benchmarks.ridge import Ridge
from NiaPy.benchmarks.chungReynolds import ChungReynolds
from NiaPy.benchmarks.csendes import Csendes
from NiaPy.benchmarks.pinter import Pinter
from NiaPy.benchmarks.qing import Qing
from NiaPy.benchmarks.quintic import Quintic
from NiaPy.benchmarks.salomon import Salomon
from NiaPy.benchmarks.schumerSteiglitz import SchumerSteiglitz
from NiaPy.benchmarks.step import Step
from NiaPy.benchmarks.step import Step2
from NiaPy.benchmarks.step import Step3
from NiaPy.benchmarks.stepint import Stepint
from NiaPy.benchmarks.sumSquares import SumSquares
from NiaPy.benchmarks.styblinskiTang import StyblinskiTang


__all__ = [
    'Rastrigin',
    'Rosenbrock',
    'Griewank',
    'Sphere',
    'Ackley',
    'Schwefel',
    'Schwefel221',
    'Schwefel222',
    'Whitley',
    'Alpine1',
    'Alpine2',
    'HappyCat',
    'Ridge',
    'ChungReynolds',
    'Csendes',
    'Pinter',
    'Qing',
    'Quintic',
    'Salomon',
    'SchumerSteiglitz',
    'Step',
    'Step2',
    'Step3',
    'Stepint',
    'SumSquares',
    'StyblinskiTang'
]
