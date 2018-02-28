# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys
sys.path.append('../')
# End of fix

import logging
import NiaPy

logging.basicConfig()
logger = logging.getLogger('examples')
logger.setLevel('INFO')


class MyBenchmark(object):
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
              'DifferentialEvolutionAlgorithm',
              'FireflyAlgorithm',
              'FlowerPollinationAlgorithm',
              'GreyWolfOptimizer',
              'ArtificialBeeColonyAlgorithm',
              'GeneticAlgorithm',
              'ParticleSwarmAlgorithm',
              'HybridBatAlgorithm',
              'SelfAdaptiveDifferentialEvolutionAlgorithm']
benchmarks = ['ackley', 'alpine1', 'alpine2', 'chungReynolds',
              'csendes', 'griewank', 'happyCat', 'pinter',
              'quing', 'quintic', 'rastrigin', 'ridge',
              'rosenbrock', 'salomon', 'schumerSteiglitz', 'schwefel',
              'schwefel221', 'schwefel222', 'sphere', 'step',
              'step2', 'step3', 'stepint', 'styblinskiTang',
              'sumSquares', 'whitley', MyBenchmark()]

NiaPy.Runner(10, 40, 1000, 3, algorithms, benchmarks).run(export='json', verbose=True)
