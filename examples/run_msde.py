# encoding=utf8
# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys
sys.path.append('../')
# End of fix

import random
from NiaPy.algorithms.basic import MultiStrategyDifferentialEvolution
from NiaPy.task.task import StoppingTask, OptimizationType
from NiaPy.benchmarks import Sphere

#we will run Differential Evolution for 5 independent runs
for i in range(5):
    task = StoppingTask(D=10, nFES=1000, optType=OptimizationType.MINIMIZATION, benchmark=Sphere())
    algo = MultiStrategyDifferentialEvolution(NP=50, F=0.5, CR=0.9)
    best = algo.run(task=task)
    print('%s -> %s' % (best[0].x, best[1]))
