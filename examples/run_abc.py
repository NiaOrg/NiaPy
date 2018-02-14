# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys
sys.path.append('../')
# End of fix

from __future__ import print_function

import random
from NiaPy.algorithms.basic import ArtificialBeeColonyAlgorithm


class MyBenchmark(object):
    def __init__(self):
        self.Lower = -5
        self.Upper = 5

    def function(self):
        def evaluate(D, sol):
            val = 0.0
            for i in range(D):
                val = val + sol[i] * sol[i]
            return val
        return evaluate


for i in range(10):
    Algorithm = ArtificialBeeColonyAlgorithm(10, 40, 10000, MyBenchmark())
    Best = Algorithm.run()

    print(Best)
