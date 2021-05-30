# encoding=utf8
# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys

sys.path.append('../')
# End of fix

from niapy.algorithms.basic import ForestOptimizationAlgorithm
from niapy.task import Task
from niapy.problems import Sphere

# we will run Forest Optimization Algorithm for 5 independent runs
for i in range(5):
    task = Task(problem=Sphere(dimension=10), max_evals=10000)
    algo = ForestOptimizationAlgorithm(population_size=20, lifetime=5, area_limit=20, local_seeding_changes=1,
                                       global_seeding_changes=1, transfer_rate=0.35)
    best = algo.run(task)
    print('%s -> %s' % (best[0], best[1]))
