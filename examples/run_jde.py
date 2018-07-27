# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys
sys.path.append('../')
# End of fix

import random
import logging
from NiaPy.algorithms.modified import SelfAdaptiveDifferentialEvolutionAlgorithm

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
    Algorithm = SelfAdaptiveDifferentialEvolutionAlgorithm(NP=10, D=40, nFES=10000, F=0.5, F_l=-1, F_u=2.0, Tao1=0.1, CR=0.45, Tao2=0.25, benchmark=MyBenchmark())
    Best = Algorithm.run()
    logger.info(Best)
