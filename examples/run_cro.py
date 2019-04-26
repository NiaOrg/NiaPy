# encoding=utf8
# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys
sys.path.append('../')
# End of fix

import random
from NiaPy.algorithms.basic import CoralReefsOptimization
from NiaPy.task.task import StoppingTask, OptimizationType
from NiaPy.benchmarks import Sphere

#we will run Coral Reefs Optimization algorithm for 5 independent runs
for i in range(5):
    task = StoppingTask(D=10, nFES=1000, optType=OptimizationType.MINIMIZATION, benchmark=Sphere())
    algo = CoralReefsOptimization(N=60, Fb=0.9, Fa=0.4, Fd=0.3, phi=25)
    best = algo.run(task=task)
    print(best)
 
