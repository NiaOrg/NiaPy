# encoding=utf8
# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys
sys.path.append('../')
# End of fix

import random
from NiaPy.algorithms.other import SimulatedAnnealing
from NiaPy.task.task import StoppingTask, OptimizationType
from NiaPy.benchmarks import Sphere
from NiaPy.algorithms.other.sa import coolLinear

# we will run Simulated Annealing for 5 independent runs
for i in range(5):
    task = StoppingTask(D=10, nGEN=10000, optType=OptimizationType.MINIMIZATION, benchmark=Sphere())
    algo = SimulatedAnnealing(coolingMethod=coolLinear)
    best = algo.run(task=task)
    print(best)
