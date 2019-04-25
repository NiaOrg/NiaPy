# encoding=utf8
# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys
sys.path.append('../')
# End of fix

import random
from NiaPy.algorithms.other import HillClimbAlgorithm
from NiaPy.task.task import StoppingTask, OptimizationType
from NiaPy.benchmarks import Sphere

#we will run Hill Climb algorithm for 5 independent runs
for i in range(5):
    task = StoppingTask(D=10, nGEN=4000, optType=OptimizationType.MINIMIZATION, benchmark=Sphere())
    algo = HillClimbAlgorithm(delta=0.3)
    best = algo.run(task)
    print(best)
