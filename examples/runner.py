# encoding=utf8
# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys
sys.path.append('../')
# End of fix

"""Example demonstrating the use of NiaPy Runner."""

from NiaPy.benchmarks import Ackley, Griewank, Sphere, HappyCat
from NiaPy.algorithms.basic import (
    DifferentialEvolution,
    GreyWolfOptimizer,
    ParticleSwarmAlgorithm
)
from NiaPy.algorithms.modified import SelfAdaptiveDifferentialEvolution
from NiaPy import Runner

runner = Runner(
    D=40,
    nFES=100,
    nRuns=1,
    useAlgorithms=[
        ParticleSwarmAlgorithm(NP=40)],
    useBenchmarks=[
        Ackley(),
        Griewank(),
        Sphere(),
        HappyCat(),
        'rastrigin']
)

print(runner.run(verbose=True, export="xlsx"))
