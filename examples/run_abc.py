# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys
sys.path.append('../')
# End of fix

import random
from NiaPy.algorithms.basic import ArtificialBeeColonyAlgorithm


def Fun(D, sol):
    val = 0.0
    for i in range(D):
        val = val + sol[i] * sol[i]
    return val


for i in range(10):
    Algorithm = ArtificialBeeColonyAlgorithm(10, 40, 10000, 0.0, 2.0, Fun)
    Best = Algorithm.run()

    print(Best)
