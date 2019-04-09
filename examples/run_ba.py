# encoding=utf8
# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys
sys.path.append('../')
# End of fix

from NiaPy.algorithms.basic import BatAlgorithm
from NiaPy.util import StoppingTask, OptimizationType
from NiaPy.benchmarks import Sphere

# we will run Bat Algorithm for 5 independent runs
for i in range(5):
    task = StoppingTask(D=10, nFES=10000, optType=OptimizationType.MINIMIZATION, benchmark=Sphere())
    algo = BatAlgorithm(NP=40, A=0.7, r=0.7, Qmin=0.0, Qmax=2.0)
    best = algo.run(task=task)
    print best
