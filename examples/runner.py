# encoding=utf8
# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys
sys.path.append('../')
# End of fix

from NiaPy import Runner
from NiaPy.algorithms.basic import (
    GreyWolfOptimizer,
    ParticleSwarmAlgorithm
)
from NiaPy.benchmarks import (
    Benchmark,
    Ackley,
    Griewank,
    Sphere,
    HappyCat
)


"""Example demonstrating the use of NiaPy Runner."""

class MyBenchmark(Benchmark):
	def __init__(self):
		Benchmark.__init__(self, -10, 10)

	def function(self):
		def evaluate(D, sol):
			val = 0.0
			for i in range(D): val += sol[i] ** 2
			return val
		return evaluate

runner = Runner(
    D=40,
    nFES=100,
    nRuns=2,
    useAlgorithms=[
        GreyWolfOptimizer(),
        "FlowerPollinationAlgorithm",
        ParticleSwarmAlgorithm(),
        "HybridBatAlgorithm",
        "SimulatedAnnealing",
        "CuckooSearch"],
    useBenchmarks=[
        Ackley(),
        Griewank(),
        Sphere(),
        HappyCat(),
        "rastrigin",
        MyBenchmark()
    ]
)

runner.run(export='json', verbose=True)
