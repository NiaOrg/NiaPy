"""Module with implementations of benchmark functions."""

from niapy.benchmarks.ackley import Ackley
from niapy.benchmarks.alpine import Alpine1, Alpine2
from niapy.benchmarks.benchmark import Benchmark
from niapy.benchmarks.bent_cigar import BentCigar
from niapy.benchmarks.chung_reynolds import ChungReynolds
from niapy.benchmarks.cosine_mixture import CosineMixture
from niapy.benchmarks.csendes import Csendes
from niapy.benchmarks.discus import Discus
from niapy.benchmarks.dixon_price import DixonPrice
from niapy.benchmarks.elliptic import Elliptic
from niapy.benchmarks.griewank import Griewank, ExpandedGriewankPlusRosenbrock
from niapy.benchmarks.happy_cat import HappyCat
from niapy.benchmarks.hgbat import HGBat
from niapy.benchmarks.infinity import Infinity
from niapy.benchmarks.katsuura import Katsuura
from niapy.benchmarks.levy import Levy
from niapy.benchmarks.michalewichz import Michalewichz
from niapy.benchmarks.perm import Perm
from niapy.benchmarks.pinter import Pinter
from niapy.benchmarks.powell import Powell
from niapy.benchmarks.qing import Qing
from niapy.benchmarks.quintic import Quintic
from niapy.benchmarks.rastrigin import Rastrigin
from niapy.benchmarks.ridge import Ridge
from niapy.benchmarks.rosenbrock import Rosenbrock
from niapy.benchmarks.salomon import Salomon
from niapy.benchmarks.schaffer import SchafferN2, SchafferN4, ExpandedSchaffer
from niapy.benchmarks.schumer_steiglitz import SchumerSteiglitz
from niapy.benchmarks.schwefel import Schwefel, Schwefel221, Schwefel222, ModifiedSchwefel
from niapy.benchmarks.sphere import Sphere, Sphere2, Sphere3
from niapy.benchmarks.step import Step, Step2, Step3
from niapy.benchmarks.stepint import Stepint
from niapy.benchmarks.styblinski_tang import StyblinskiTang
from niapy.benchmarks.sum_squares import SumSquares
from niapy.benchmarks.trid import Trid
from niapy.benchmarks.weierstrass import Weierstrass
from niapy.benchmarks.whitley import Whitley
from niapy.benchmarks.zakharov import Zakharov

__all__ = [
    'Benchmark',
    'Rastrigin',
    'Rosenbrock',
    'Griewank',
    'ExpandedGriewankPlusRosenbrock',
    'Sphere',
    'Ackley',
    'Schwefel',
    'Schwefel221',
    'Schwefel222',
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
    'Michalewichz',
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
    'ExpandedSchaffer',
    'SchafferN2',
    'SchafferN4'
]
