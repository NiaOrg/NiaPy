# encoding=utf8
# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
from NiaPy.benchmarks import Sphere
from NiaPy.util import StoppingTask, OptimizationType
from NiaPy.algorithms.basic import MonarchButterflyOptimization
import random
import sys
sys.path.append('../')
# End of fix


# we will run Forest Optimization Algorithm for 5 independent runs
for i in range(5):
    task = StoppingTask(D=10, nFES=100000, optType=OptimizationType.MINIMIZATION, benchmark=Sphere())
    algo = MonarchButterflyOptimization(NP=50)
    best = algo.run(task=task)
    print(best)
