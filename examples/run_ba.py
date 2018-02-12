# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys
sys.path.append('../')
# End of fix

import random
from NiaPy.algorithms.basic import BatAlgorithm


def Fun(D, sol):
    val = 0.0
    for i in range(D):
        val = val + sol[i] * sol[i]
    return val


# example using custom benchmark function "Fun"
for i in range(10):
    Algorithm = BatAlgorithm(10, 40, 10000, 0.5, 0.5,
                             0.0, 2.0, -10.0, 10.0, Fun)
    Best = Algorithm.move_bat()

    print(Best)

# example using predifined benchmark function
# available benchmarks are:
# - griewank
# - rastrigin
# - rosenbrock
# - sphere
for i in range(10):
    Algorithm = BatAlgorithm(10, 40, 10000, 0.5, 0.5,
                             0.0, 2.0, -10.0, 10.0, 'griewank')
    Best = Algorithm.move_bat()

    print(Best)
