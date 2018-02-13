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

# For reproducive results
random.seed(1234)


class MyBenchmark(object):
    def __init__(self):
        self.Lower = -11
        self.Upper = 11

    def function(self):
        def evaluate(D, sol):
            val = 0.0
            for i in range(D):
                val = val + sol[i] * sol[i]
            return val
        return evaluate


algorithms = ['BatAlgorithm', 'DifferentialEvolutionAlgorithm']
benchmarks = ['griewank', 'ackley', 'sphere', MyBenchmark()]

NiaPy.Runner(10, 40, 1000, 10, algorithms, benchmarks).run()
