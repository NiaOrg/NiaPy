# encoding=utf8
# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys

sys.path.append('../')
# End of fix

import numpy as np
from niapy import Runner
from niapy.algorithms.basic import (
    GreyWolfOptimizer,
    ParticleSwarmAlgorithm
)
from niapy.problems import (
    Problem,
    Ackley,
    Griewank,
    Sphere,
    HappyCat
)

"""Example demonstrating the use of niapy Runner."""


class MyProblem(Problem):
    def __init__(self, dimension, lower=-10, upper=10, *args, **kwargs):
        super().__init__(dimension, lower, upper, *args, **kwargs)

    def _evaluate(self, x):
        return np.sum(x ** 2)


runner = Runner(dimension=40, max_evals=100, runs=2, algorithms=[
    GreyWolfOptimizer(),
    "FlowerPollinationAlgorithm",
    ParticleSwarmAlgorithm(),
    "HybridBatAlgorithm",
    "SimulatedAnnealing",
    "CuckooSearch"], problems=[
    Ackley(dimension=40),
    Griewank(dimension=40),
    Sphere(dimension=40),
    HappyCat(dimension=40),
    "rastrigin",
    MyProblem(dimension=40)
])

runner.run(export='dataframe', verbose=True)
