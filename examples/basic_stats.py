# encoding=utf8
# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys

sys.path.append('../')
# End of fix

import numpy as np
from niapy.task import StoppingTask, OptimizationType
from niapy.algorithms import BasicStatistics
from niapy.algorithms.basic import DifferentialEvolution
from niapy.benchmarks import Sphere

NUM_RUNS = 10  # define number of runs
stats = np.zeros(NUM_RUNS)

for i in range(NUM_RUNS):
    task = StoppingTask(max_evals=10000, dimension=10, optimization_type=OptimizationType.MINIMIZATION,
                        benchmark=Sphere())
    print("Working on run: " + str(i + 1))
    algo = DifferentialEvolution(population_size=40, CR=0.9, F=0.5)
    best = algo.run(task)
    stats[i] = best[1]  # save best

stat = BasicStatistics(stats)
print(stat.generate_standard_report())  # generate report
