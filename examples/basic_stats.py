# encoding=utf8
# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys
sys.path.append('../')
# End of fix

import random
import logging
import numpy as np
from NiaPy import Runner
from NiaPy.util import Task, TaskConvPrint, TaskConvPlot, OptimizationType, getDictArgs
from NiaPy.algorithms.modified import SelfAdaptiveDifferentialEvolution
from NiaPy.algorithms.basic import DifferentialEvolution, MonkeyKingEvolutionV3
from NiaPy.benchmarks import Griewank, Sphere
from NiaPy.algorithms.statistics import BasicStatistics

NUM_RUNS = 10  # define number of runs
stats = np.zeros(NUM_RUNS)

for i in range(NUM_RUNS):
    task = Task(
        D=10,
        nFES=10000,
        optType=OptimizationType.MINIMIZATION,
        benchmark=Sphere())
    print ("Working on run: " + str(i+1))
    algo = DifferentialEvolution(task=task, NP=40, CR=0.9, F=0.5)
    best = algo.run()
    stats[i] = best[1]  # save best


stat = BasicStatistics(stats)
print stat.generate_standard_report()  # show basic stats
