# encoding=utf8
# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys
sys.path.append('../')
# End of fix

import random
from NiaPy.algorithms.basic import ParticleSwarmAlgorithm
from NiaPy.task.task import StoppingTask, OptimizationType
from NiaPy.benchmarks import Sphere

# we will run ParticleSwarmAlgorithm for 1 independent runs
for i in range(1):
    task = StoppingTask(D=10, nFES=1000, optType=OptimizationType.MINIMIZATION, logger=True, benchmark=Sphere())
    algo = ParticleSwarmAlgorithm(NP=40, C1=2.0, C2=2.0, w=0.7, vMin=-4, vMax=4)
    best = algo.run(task=task)
    print('%s -> %s' % (best[0], best[1]))

    #plot a convergence graph
    task.plot()