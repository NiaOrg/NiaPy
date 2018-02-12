# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys
sys.path.append('../')
# End of fix

import random
import logging
from NiaPy.algorithms.basic import DifferentialEvolutionAlgorithm

logging.basicConfig()
logger = logging.getLogger('examples')
logger.setLevel('INFO')

# For reproducive results
random.seed(1234)

def Fun(D, sol):
    val = 0.0
    for i in range(D):
        val = val + sol[i] * sol[i]
    return val


for i in range(10):
    Algorithm = DifferentialEvolutionAlgorithm(10, 40, 10000, 0.5, 0.9, 0.0, 2.0, Fun)
    Best = Algorithm.run()

    logger.info(Best)
