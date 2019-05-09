# coding=utf-8
# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
from NiaPy.benchmarks import Sphere
from NiaPy.task.task import StoppingTask, OptimizationType
from NiaPy.algorithms.basic import BeesAlgorithm

import sys
sys.path.append('../')
# End of fix


# we will run Bees Algorithm for 5 independent runs
for i in range(5):
    task = StoppingTask(D=10, nFES=100000, optType=OptimizationType.MINIMIZATION, benchmark=Sphere())
    algo = BeesAlgorithm(NP=30, m=15, e=6, nep=15, nsp=10, ngh=3)
    best = algo.run(task=task)
    print('%s -> %s' % (best[0], best[1]))
