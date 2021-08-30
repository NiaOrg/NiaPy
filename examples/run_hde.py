# encoding=utf8
# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys

sys.path.append('../')
# End of fix

from niapy.algorithms.modified import DifferentialEvolutionMTS
from niapy.task import Task
from niapy.problems import Sphere

# we will run Differential Evolution for 5 independent runs
for i in range(5):
    task = Task(problem=Sphere(dimension=10), max_evals=10000)
    algo = DifferentialEvolutionMTS(population_size=50, differential_weight=0.5, crossover_probability=0.9)
    best = algo.run(task)
    print('%s -> %s' % (best[0], best[1]))
