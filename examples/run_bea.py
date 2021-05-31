# encoding=utf8
# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys

from niapy.algorithms.basic import BeesAlgorithm
from niapy.problems import Sphere
from niapy.task import Task

sys.path.append('../')

# we will run Bees Algorithm for 5 independent runs
for i in range(5):
    task = Task(problem=Sphere(dimension=10), max_iters=2)
    algo = BeesAlgorithm(population_size=50, m=20, e=10, nep=20, nsp=15, ngh=7)
    best = algo.run(task)
    print('%s -> %s' % (best[0], best[1]))
