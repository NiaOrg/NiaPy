# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys
sys.path.append('../')
# End of fix

import random
import logging
import NiaPy

logging.basicConfig()
logger = logging.getLogger('examples')
logger.setLevel('INFO')


class MyBenchmark(object):
    def __init__(self):
        self.Lower = -5
        self.Upper = 5

    def function(self):
        def evaluate(D, sol):
            val = 0.0
            for i in range(D):
                val = val + sol[i] * sol[i]
            return val
        return evaluate


algorithms = ['BatAlgorithm',
              'DifferentialEvolutionAlgorithm', 'FireflyAlgorithm', 'FlowerPollinationAlgorithm', 'GreyWolfOptimizer', 'HybridBatAlgorithm']
benchmarks = ['sphere', 'ackley', 'rosenbrock', 'griewank', 'rastrigin', MyBenchmark()]

NiaPy.Runner(10, 40, 10000, 10, algorithms, benchmarks).run()
