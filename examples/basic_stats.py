# encoding=utf8
# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys
sys.path.append('../')
# End of fix

import numpy as np
from NiaPy.task.task import StoppingTask, OptimizationType
from NiaPy.algorithms.basic import DifferentialEvolution
from NiaPy.benchmarks import Sphere
from NiaPy.algorithms.statistics import BasicStatistics

NUM_RUNS = 10  # define number of runs
stats = np.zeros(NUM_RUNS)

for i in range(NUM_RUNS):
    task = StoppingTask(D=10, nFES=10000, optType=OptimizationType.MINIMIZATION, benchmark=Sphere())
    print ("Working on run: " + str(i+1))
    algo = DifferentialEvolution(NP=40, CR=0.9, F=0.5)
    best = algo.run(task)
    stats[i] = best[1]  # save best

stat = BasicStatistics(stats)
print(stat.generate_standard_report())  # generate report
