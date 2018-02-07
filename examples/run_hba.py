# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys
sys.path.append('../')
# End of fix

import random
from NiaPy.algorithms.modified import HybridBatAlgorithm

def Fun(D, sol):
    val = 0.0
    for i in range(D):
        val = val + sol[i] * sol[i]
    return val

# For reproducive results
random.seed(1234)

for i in range(10):
    Algorithm = HybridBatAlgorithm(
        10, 40, 1000, 0.5, 0.5, 0.0, 2.0, -2, 2, Fun)
    Best = Algorithm.move_bat()

print(Best)