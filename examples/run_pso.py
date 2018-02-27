# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys
sys.path.append('../')
# End of fix

import random
from NiaPy.algorithms.basic import ParticleSwarmAlgorithm


for i in range(10):
    #def __init__(self, Np, D, nFES, C1, C2, w, vMin, vMax, benchmark):
    Algorithm = ParticleSwarmAlgorithm(40, 10, 10000, 2.0, 2.0, 0.7, -6, 6,0,0, 'sphere')
    Best = Algorithm.run()

    print(Best)
