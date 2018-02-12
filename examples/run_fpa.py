# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys
sys.path.append('../')
# End of fix

import logging
from NiaPy.algorithms.basic import FlowerPollinationAlgorithm

def Fun(D, sol):
    val = 0.0
    for i in range(D):
        val = val + sol[i] * sol[i]
    return val

for i in range(10):
    Algorithm = FlowerPollinationAlgorithm(10, 20, 10000, 0.5, -2.0, 2.0, Fun)
    Best = Algorithm.move_flower()

    logging.info(Best)
