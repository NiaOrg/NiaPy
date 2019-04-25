# encoding=utf8
# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys
sys.path.append('../')
# End of fix

import random
from NiaPy.algorithms.other import NelderMeadMethod
from NiaPy.task.task import StoppingTask, OptimizationType
from NiaPy.benchmarks import Sphere

# we will run Nelder Mead algorithm for 5 independent runs
for i in range(5):
    task = StoppingTask(D=10, nGEN=10000, optType=OptimizationType.MINIMIZATION, benchmark=Sphere())
    algo = NelderMeadMethod(NP=70, alpha=0.2, gamma=0.1, rho=-0.24, sigma=-0.1)
    best = algo.run(task=task)
    print('%s -> %s' % (best[0], best[1]))
