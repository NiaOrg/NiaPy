# encoding=utf8
# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys
sys.path.append('../')
# End of fix

import random
from NiaPy.algorithms.basic import FlowerPollinationAlgorithm
from NiaPy.task.task import StoppingTask, OptimizationType
from NiaPy.benchmarks import Sphere

#we will run Flower Pollination Algorithm for 5 independent runs
for i in range(5):
    task = StoppingTask(D=10, nFES=10000, optType=OptimizationType.MINIMIZATION, benchmark=Sphere())
    algo = FlowerPollinationAlgorithm(NP=20, p=0.5)
    best = algo.run(task=task)
    print(best)

