"""Module with implementations of benchmark functions."""

from NiaPy.benchmarks.benchmark import Benchmark
from NiaPy.benchmarks.ackley import Ackley
from NiaPy.benchmarks.alpine import Alpine1, Alpine2
from NiaPy.benchmarks.rastrigin import Rastrigin
from NiaPy.benchmarks.rosenbrock import Rosenbrock
from NiaPy.benchmarks.griewank import Griewank, ExpandedGriewankPlusRosenbrock
from NiaPy.benchmarks.schwefel import Schwefel, Schwefel221, Schwefel222, ExpandedScaffer, ModifiedSchwefel
from NiaPy.benchmarks.whitley import Whitley
from NiaPy.benchmarks.happyCat import HappyCat
from NiaPy.benchmarks.ridge import Ridge
from NiaPy.benchmarks.chungReynolds import ChungReynolds
from NiaPy.benchmarks.csendes import Csendes
from NiaPy.benchmarks.pinter import Pinter
from NiaPy.benchmarks.qing import Qing
from NiaPy.benchmarks.quintic import Quintic
from NiaPy.benchmarks.salomon import Salomon
from NiaPy.benchmarks.schumerSteiglitz import SchumerSteiglitz
from NiaPy.benchmarks.step import Step, Step2, Step3
from NiaPy.benchmarks.stepint import Stepint
from NiaPy.benchmarks.sumSquares import SumSquares
from NiaPy.benchmarks.styblinskiTang import StyblinskiTang
from NiaPy.benchmarks.bentcigar import BentCigar
from NiaPy.benchmarks.weierstrass import Weierstrass
from NiaPy.benchmarks.hgbat import HGBat
from NiaPy.benchmarks.katsuura import Katsuura
from NiaPy.benchmarks.elliptic import Elliptic
from NiaPy.benchmarks.discus import Discus
from NiaPy.benchmarks.michalewicz import Michalewicz
from NiaPy.benchmarks.levy import Levy
from NiaPy.benchmarks.sphere import Sphere, Sphere2, Sphere3
from NiaPy.benchmarks.trid import Trid
from NiaPy.benchmarks.perm import Perm
from NiaPy.benchmarks.zakharov import Zakharov
from NiaPy.benchmarks.dixonprice import DixonPrice
from NiaPy.benchmarks.powell import Powell
from NiaPy.benchmarks.cosinemixture import CosineMixture
from NiaPy.benchmarks.infinity import Infinity

__all__ = [
    'Rastrigin',
    'Rosenbrock',
    'Griewank',
    'ExpandedGriewankPlusRosenbrock',
    'Sphere',
    'Ackley',
    'Schwefel',
    'Schwefel221',
    'Schwefel222',
    'ExpandedScaffer',
    'ModifiedSchwefel',
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
    'StyblinskiTang',
    'BentCigar',
    'Weierstrass',
    'HGBat',
    'Katsuura',
    'Elliptic',
    'Discus',
    'Michalewicz',
    'Levy',
    'Sphere',
    'Sphere2',
    'Sphere3',
    'Trid',
    'Perm',
    'Zakharov',
    'DixonPrice',
    'Powell',
    'CosineMixture',
    'Infinity',
    'Benchmark'
]
