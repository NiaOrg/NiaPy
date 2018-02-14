"""Module with implementations of benchmarks."""

from NiaPy.benchmarks.rastrigin import Rastrigin
from NiaPy.benchmarks.rosenbrock import Rosenbrock
from NiaPy.benchmarks.griewank import Griewank
from NiaPy.benchmarks.sphere import Sphere
from NiaPy.benchmarks.ackley import Ackley
from NiaPy.benchmarks.schwefel import Schwefel
from NiaPy.benchmarks.schwefel import Schwefel221
from NiaPy.benchmarks.schwefel import Schwefel222
from NiaPy.benchmarks.whitley import Whitley

__all__ = [
    'Rastrigin',
    'Rosenbrock',
    'Griewank',
    'Sphere',
    'Ackley',
    'Schwefel',
    'Schwefel221',
    'Schwefel222',
    'Whitley'
]
