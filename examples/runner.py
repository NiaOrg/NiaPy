# encoding=utf8
# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
from NiaPy import Runner
from NiaPy.algorithms.modified import SelfAdaptiveDifferentialEvolution
from NiaPy.algorithms.basic import (
    GreyWolfOptimizer,
    ParticleSwarmAlgorithm
)
from NiaPy.benchmarks import Ackley, Griewank, Sphere, HappyCat
import sys
sys.path.append('../')
# End of fix

"""Example demonstrating the use of NiaPy Runner."""


runner = Runner(
    D=40,
    nFES=100,
    nRuns=1,
    useAlgorithms=[
        SelfAdaptiveDifferentialEvolution(),
        "DifferentialEvolution",
        GreyWolfOptimizer(),
        ParticleSwarmAlgorithm(NP=40)],
    useBenchmarks=[
        Ackley(),
        Griewank(),
        Sphere(),
        HappyCat(),
        "rastrigin"]
)

print(runner.run(verbose=True))
