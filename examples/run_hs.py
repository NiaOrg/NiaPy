# encoding=utf8
# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys
sys.path.append('../')
# End of fix

import random
from NiaPy.algorithms.basic import HarmonySearch
from NiaPy.util import StoppingTask
from NiaPy.task.task import OptimizationType
from NiaPy.benchmarks import Sphere

#we will run Harmony Search for 5 independent runs
for i in range(5):
    task = StoppingTask(D=10, nFES=10000, optType=OptimizationType.MINIMIZATION, benchmark=Sphere())
    algo = HarmonySearch()
    best = algo.run(task=task)
    print(best)

