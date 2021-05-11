# encoding=utf8
# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys

sys.path.append('../')
# End of fix

from niapy.task import StoppingTask
from niapy.benchmarks import Sphere
from niapy.algorithms.basic import DifferentialEvolution

# Storing improvements during the evolutionary cycle
for i in range(1):
    task = StoppingTask(max_evals=10000, dimension=10, benchmark=Sphere())
    algo = DifferentialEvolution(population_size=40, crossover_probability=0.9, differential_weight=0.5)
    best = algo.run(task)
    evals, x_f = task.return_conv()
    print(evals)  # print function evaluations
    print(x_f)  # print values
