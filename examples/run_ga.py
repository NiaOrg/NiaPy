# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys
sys.path.append('../')
# End of fix

import random
from NiaPy.algorithms.basic import GeneticAlgorithm


for i in range(10):
    Algorithm = GeneticAlgorithm(10, 40, 10000, 4, 0.05,0.4, 'sphere')
    Best = Algorithm.run()

    print(Best)
