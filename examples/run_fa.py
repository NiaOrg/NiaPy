# encoding=utf8
# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys
sys.path.append('../')
# End of fix

import random
from NiaPy.algorithms.basic import FireflyAlgorithm
from NiaPy.util import StoppingTask, OptimizationType
from NiaPy.benchmarks import Sphere

#we will run Firefly Algorithm for 5 independent runs
for i in range(5):
    task = StoppingTask(D=10, nFES=10000, optType=OptimizationType.MINIMIZATION, benchmark=Sphere())
    algo = FireflyAlgorithm(NP=20, alpha=0.5, betamin=0.2, gamma=1.0)
    best = algo.run(task=task)
    print(best)
