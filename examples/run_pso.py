# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys
sys.path.append('../')
# End of fix

import random
import logging
from NiaPy.algorithms.basic import ParticleSwarmAlgorithm

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


for i in range(10):
    Algorithm = ParticleSwarmAlgorithm(NP=50, D=40, nFES=40000, C1=2.0, C2=2.0, w=0.5, vMin=-5, vMax=5, benchmark=MyBenchmark())
    Best = Algorithm.run()
    logger.info(Best)
