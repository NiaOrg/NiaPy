"""Module with implementations of optimization problems."""

from niapy.problems.ackley import Ackley
from niapy.problems.alpine import Alpine1, Alpine2
from niapy.problems.problem import Problem
from niapy.problems.bent_cigar import BentCigar
from niapy.problems.chung_reynolds import ChungReynolds
from niapy.problems.cosine_mixture import CosineMixture
from niapy.problems.csendes import Csendes
from niapy.problems.discus import Discus
from niapy.problems.dixon_price import DixonPrice
from niapy.problems.elliptic import Elliptic
from niapy.problems.griewank import Griewank, ExpandedGriewankPlusRosenbrock
from niapy.problems.happy_cat import HappyCat
from niapy.problems.hgbat import HGBat
from niapy.problems.infinity import Infinity
from niapy.problems.katsuura import Katsuura
from niapy.problems.levy import Levy
from niapy.problems.michalewichz import Michalewichz
from niapy.problems.perm import Perm
from niapy.problems.pinter import Pinter
from niapy.problems.powell import Powell
from niapy.problems.qing import Qing
from niapy.problems.quintic import Quintic
from niapy.problems.rastrigin import Rastrigin
from niapy.problems.ridge import Ridge
from niapy.problems.rosenbrock import Rosenbrock
from niapy.problems.salomon import Salomon
from niapy.problems.schaffer import SchafferN2, SchafferN4, ExpandedSchaffer
from niapy.problems.schumer_steiglitz import SchumerSteiglitz
from niapy.problems.schwefel import Schwefel, Schwefel221, Schwefel222, ModifiedSchwefel
from niapy.problems.sphere import Sphere, Sphere2, Sphere3
from niapy.problems.step import Step, Step2, Step3
from niapy.problems.stepint import Stepint
from niapy.problems.styblinski_tang import StyblinskiTang
from niapy.problems.sum_squares import SumSquares
from niapy.problems.trid import Trid
from niapy.problems.weierstrass import Weierstrass
from niapy.problems.whitley import Whitley
from niapy.problems.zakharov import Zakharov

__all__ = [
    'Problem',
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
