# encoding=utf8
# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys

sys.path.append('../')

from niapy.algorithms.basic import MonarchButterflyOptimization
from niapy.task import Task
from niapy.problems import Sphere

# we will run Monarch Butterfly Optimization algorithm for 5 independent runs
for i in range(5):
    task = Task(problem=Sphere(dimension=10), max_evals=10000)
    algo = MonarchButterflyOptimization(population_size=20, partition=5.0 / 12.0, period=1.2)
    best = algo.run(task)
    print('%s -> %s' % (best[0], best[1]))
