# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys

sys.path.append('../')
# End of fix

from niapy.algorithms.basic import GreyWolfOptimizer
from niapy.task import Task

# we will run 10 repetitions of Grey Wolf Optimizer against Pinter problem
for i in range(10):
    task = Task(problem='pinter', dimension=10, max_evals=1000)
    algorithm = GreyWolfOptimizer(population_size=20)
    best = algorithm.run(task)
    print(best[-1])
