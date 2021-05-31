# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys

sys.path.append('../')
# End of fix

from niapy.algorithms.basic import GreyWolfOptimizer
from niapy.task import Task, OptimizationType
from niapy.problems import Pinter

# initialize Pinter problem with custom bound
pinter = Pinter(20, -5, 5)

# we will run 10 repetitions of Grey Wolf Optimizer against Pinter problem
for i in range(10):
    task = Task(problem=pinter, max_iters=100)

    # parameter is population size
    algo = GreyWolfOptimizer(population_size=20)

    # running algorithm returns best found minimum
    best = algo.run(task)

    # printing best minimum
    print(best[-1])
