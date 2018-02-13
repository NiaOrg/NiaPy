# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys
sys.path.append('../')
# End of fix

import logging
from NiaPy.algorithms.basic import FlowerPollinationAlgorithm

logging.basicConfig()
logger = logging.getLogger('examples')
logger.setLevel('INFO')


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

for i in range(10):
    Algorithm = FlowerPollinationAlgorithm(10, 20, 10000, 0.5, MyBenchmark())
    Best = Algorithm.run()

    logger.info(Best)
