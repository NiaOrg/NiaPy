# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys
sys.path.append('../')
# End of fix

import random
from NiaPy.algorithms.basic import ParticleSwarmAlgorithm


for i in range(10):
    Algorithm = ParticleSwarmAlgorithm(
        50, 40, 40000, 2.0, 2.0, 0.5, -5, 5, 'sphere')

    Best = Algorithm.run()

    print(Best)
