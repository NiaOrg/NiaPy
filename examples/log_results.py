# encoding=utf8
# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys

sys.path.append('../')
# End of fix

from niapy.task import Task
from niapy.problems import Sphere
from niapy.algorithms.basic import DifferentialEvolution

# Storing improvements during the evolutionary cycle

task = Task(max_evals=10000, problem=Sphere(dimension=10))
algo = DifferentialEvolution(population_size=40, crossover_probability=0.9, differential_weight=0.5)
best = algo.run(task)
evals, x_f = task.convergence_data(x_axis='evals')
print(evals)  # print function evaluations
print(x_f)  # print values
