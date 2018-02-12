# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys
sys.path.append('../')
# End of fix

import random
import logging
from NiaPy.algorithms.modified import HybridBatAlgorithm

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
    Algorithm = HybridBatAlgorithm(
        10, 40, 1000, 0.5, 0.5, 0.0, 2.0, -2, 2, Fun)
    Best = Algorithm.move_bat()
    
    logger.info(Best)
