# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys
sys.path.append('../')
# End of fix

import logging
import NiaPy

from NiaPy.benchmarks import Benchmark

logging.basicConfig()
logger = logging.getLogger('examples')
logger.setLevel('INFO')


class MyBenchmark(Benchmark):
    Name = ['MyBenchmark']

    def __init__(self):
        self.Lower = -5.12
        self.Upper = 5.12

    def function(self):
        def evaluate(D, sol):
            val = 0.0
            for i in range(D):
                val = val + sol[i] * sol[i]
            return val
        return evaluate


algorithms = ['BatAlgorithm',
              'DifferentialEvolution',
              'FireflyAlgorithm',
              'FlowerPollinationAlgorithm',
              'GreyWolfOptimizer',
              'ArtificialBeeColonyAlgorithm',
              'GeneticAlgorithm',
              'ParticleSwarmAlgorithm',
              'HybridBatAlgorithm',
              'SelfAdaptiveDifferentialEvolution']
benchmarks = ['ackley', 'alpine1', 'alpine2', 'chungReynolds',
              'csendes', 'griewank', 'happyCat', 'pinter',
              'qing', 'quintic', 'rastrigin', 'ridge',
              'rosenbrock', 'salomon', 'schumerSteiglitz', 'schwefel',
              'schwefel221', 'schwefel222', 'sphere', 'step',
              'step2', 'step3', 'stepint', 'styblinskiTang',
              'sumSquares', 'whitley', MyBenchmark()]

NiaPy.Runner(D=10, nFES=1000, nRuns=3, useAlgorithms=algorithms, useBenchmarks=benchmarks).run(export='json', verbose=True)
