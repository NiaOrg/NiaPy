# encoding=utf8
# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys
sys.path.append('../')
# End of fix

import random
from NiaPy.algorithms.basic import ForestOptimizationAlgorithm
from NiaPy.task.task import StoppingTask, OptimizationType
from NiaPy.benchmarks import Sphere

# we will run Forest Optimization Algorithm for 5 independent runs
for i in range(5):
    task = StoppingTask(D=10, nFES=10000, optType=OptimizationType.MINIMIZATION, benchmark=Sphere())
    algo = ForestOptimizationAlgorithm(NP=20, lt=5, lsc=1, gsc=1, al=20, tr=0.35)
    best = algo.run(task=task)
    print('%s -> %s' % (best[0], best[1]))

