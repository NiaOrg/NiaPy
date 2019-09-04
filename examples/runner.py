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
    Ackley,
    Griewank,
    Sphere,
    HappyCat
)


"""Example demonstrating the use of NiaPy Runner."""


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
        "rastrigin"]
)

print(runner.run(verbose=True))
