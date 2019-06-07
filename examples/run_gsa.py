# encoding=utf8
# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys
sys.path.append('../')
# End of fix

from NiaPy.algorithms.basic import GravitationalSearchAlgorithm
from NiaPy.task.task import StoppingTask, OptimizationType
from NiaPy.benchmarks import Sphere

# we will run Gravitational Search Algorithm for 5 independent runs
for i in range(5):
    task = StoppingTask(D=10, nFES=1000, optType=OptimizationType.MINIMIZATION, benchmark=Sphere())
    algo = GravitationalSearchAlgorithm(NP=40)
    best = algo.run(task=task)
    print(best)
