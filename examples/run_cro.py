# encoding=utf8
# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys

sys.path.append('../')
# End of fix

from niapy.algorithms.basic import CoralReefsOptimization
from niapy.task import Task
from niapy.problems import Sphere

# we will run Coral Reefs Optimization algorithm for 5 independent runs
for i in range(5):
    task = Task(problem=Sphere(dimension=10), max_evals=1000)
    algo = CoralReefsOptimization(population_size=60, broadcast_prob=0.9, asexual_reproduction_prob=0.4, depredation_prob=0.3, phi=25)

    best = algo.run(task)
    print(best)
