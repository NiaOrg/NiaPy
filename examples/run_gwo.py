# encoding=utf8
# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys
sys.path.append('../')
# End of fix

from NiaPy.algorithms.basic import GreyWolfOptimizer
from NiaPy.task import StoppingTask, OptimizationType
from NiaPy.benchmarks import Sphere

# we will run Grey Wolf Optimizer for 5 independent runs looking for minimum
for i in range(5):
    task = StoppingTask(D=10, nFES=10000, benchmark=Sphere())
    algo = GreyWolfOptimizer(NP=40)
    best = algo.run(task)
    print(best)

# we will run Grey Wolf Optimizer for 5 independent runs looking for maximum
for i in range(5):
    task = StoppingTask(D=10, nFES=10000, optType=OptimizationType.MAXIMIZATION, benchmark=Sphere())
    algo = GreyWolfOptimizer(NP=40)
    best = algo.run(task)
    print(best)
