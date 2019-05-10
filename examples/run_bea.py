# encoding=utf8
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
    task = StoppingTask(D=20, nGEN=2, optType=OptimizationType.MINIMIZATION, benchmark=Sphere())
    algo = BeesAlgorithm(NP=50, m=20, e=10, nep=20, nsp=15, ngh=7)
    best = algo.run(task=task)
    print('%s -> %s' % (best[0], best[1]))
