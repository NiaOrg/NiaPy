# encoding=utf8
# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys

sys.path.append('../')
# End of fix

from niapy.algorithms.basic import GreyWolfOptimizer
from niapy.task import Task
from niapy.problems import Sphere

# we will run Grey Wolf Optimizer for 5 independent runs
for i in range(5):
    task = Task(problem=Sphere(dimension=10), max_evals=10000)
    algo = GreyWolfOptimizer(population_size=40)
    best = algo.run(task)
    print(best)
