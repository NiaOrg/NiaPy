# encoding=utf8
# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
from NiaPy.algorithms.basic import MonarchButterflyOptimization
from NiaPy.task.task import StoppingTask, OptimizationType
from NiaPy.benchmarks import Sphere
import random
import sys
sys.path.append('../')
# End of fix


# we will run Monarch Butterfly Optimization algorithm for 5 independent runs
for i in range(5):
    task = StoppingTask(D=10, nFES=100000, optType=OptimizationType.MINIMIZATION, benchmark=Sphere())
    algo = MonarchButterflyOptimization(NP=20, PAR=5.0 / 12.0, PER=1.2)
    best = algo.run(task=task)
    print('%s -> %s' % (best[0], best[1]))
