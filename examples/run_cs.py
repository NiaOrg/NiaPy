# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys
sys.path.append('../')
# End of fix

import random
from NiaPy.algorithms.basic import CuckooSearchAlgorithm


for i in range(10):
    Algorithm = CuckooSearchAlgorithm(40, 10, 10000, 0.25, 0.01, 'sphere')
    Best = Algorithm.run()

    print(Best)
