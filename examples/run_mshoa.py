# encoding=utf8
# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys

sys.path.append('../')
# End of fix

from niapy.algorithms.basic import MantisSearchAlgorithm
from niapy.task import Task
from niapy.problems import Sphere

# we will run Mantis Shrimp Optimization Algorithm for 20 iterations
task = Task(problem=Sphere(dimension=10), max_iters=20)
algo = MantisSearchAlgorithm(population_size=30, k_value=0.3)
best = algo.run(task)
print('Best solution: %s' % best[0])
print('Best fitness: %f' % best[1])

