# encoding=utf8
# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys

sys.path.append('../')
# End of fix

from niapy.algorithms.basic import KrillHerd
from niapy.task import Task
from niapy.problems import Sphere

# we will run Fireworks Algorithm for 5 independent runs
for i in range(5):
    task = Task(problem=Sphere(dimension=10), max_iters=50)
    algo = KrillHerd(population_size=70)
    best = algo.run(task)
    print('%s -> %s' % (best[0], best[1]))
