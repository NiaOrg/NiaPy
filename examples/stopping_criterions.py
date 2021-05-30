# encoding=utf8
# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys

sys.path.append('../')
# End of fix

from niapy.task import Task
from niapy.algorithms.basic import DifferentialEvolution
from niapy.problems import Sphere

# 1 Number of function evaluations (nFES) as a stopping criteria
for i in range(10):
    task = Task(problem=Sphere(dimension=10), max_evals=10000)
    algo = DifferentialEvolution(population_size=40, crossover_probability=0.9, differential_weight=0.5)
    best = algo.run(task)
    print('%s -> %s' % (best[0], best[1]))

print('---------------------------------------')

# 2 Number of generations (iterations) as a stopping criteria
for i in range(10):
    task = Task(problem=Sphere(dimension=10), max_iters=1000)
    algo = DifferentialEvolution(population_size=40, crossover_probability=0.9, differential_weight=0.5)
    best = algo.run(task)
    print('%s -> %s' % (best[0], best[1]))

print('---------------------------------------')

# 3 Reference value as a stopping criteria
for i in range(10):
    task = Task(problem=Sphere(dimension=10), cutoff_value=50.0)
    algo = DifferentialEvolution(population_size=40, crossover_probability=0.9, differential_weight=0.5)
    best = algo.run(task)
    print('%s -> %s' % (best[0], best[1]))
