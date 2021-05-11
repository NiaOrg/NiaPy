# encoding=utf8
# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys

sys.path.append('../')

from niapy.algorithms.basic import MonarchButterflyOptimization
from niapy.task import StoppingTask
from niapy.benchmarks import Sphere

# we will run Monarch Butterfly Optimization algorithm for 5 independent runs
for i in range(5):
    task = StoppingTask(max_evals=10000, dimension=10, benchmark=Sphere())
    algo = MonarchButterflyOptimization(population_size=20, partition=5.0 / 12.0, period=1.2)
    best = algo.run(task)
    print('%s -> %s' % (best[0], best[1]))
