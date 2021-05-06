# encoding=utf8
# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys

sys.path.append('../')
# End of fix

from niapy import Runner
from niapy.algorithms.basic import (
    GreyWolfOptimizer,
    ParticleSwarmAlgorithm
)
from niapy.benchmarks import (
    Benchmark,
    Ackley,
    Griewank,
    Sphere,
    HappyCat
)

"""Example demonstrating the use of niapy Runner."""


class MyBenchmark(Benchmark):
    def __init__(self):

        Benchmark.__init__(self, -10, 10)

    def function(self):
        def evaluate(D, sol):
            val = 0.0
            for i in range(D): val += sol[i] ** 2
            return val

        return evaluate


runner = Runner(dimension=40, max_evals=100, runs=2, algorithms=[
    GreyWolfOptimizer(),
    "FlowerPollinationAlgorithm",
    ParticleSwarmAlgorithm(),
    "HybridBatAlgorithm",
    "SimulatedAnnealing",
    "CuckooSearch"], benchmarks=[
    Ackley(),
    Griewank(),
    Sphere(),
    HappyCat(),
    "rastrigin",
    MyBenchmark()
])

runner.run(export='dataframe', verbose=True)
