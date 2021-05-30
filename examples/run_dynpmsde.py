# encoding=utf8
# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys

sys.path.append('../')
# End of fix

from niapy.algorithms.basic import DynNpMultiStrategyDifferentialEvolution
from niapy.algorithms.basic.de import cross_best2, cross_curr2best1
from niapy.task import Task
from niapy.problems import Sphere

# we will run Differential Evolution for 5 independent runs
for i in range(5):
    task = Task(problem=Sphere(dimension=10), max_evals=10000)
    algo = DynNpMultiStrategyDifferentialEvolution(population_size=80, differential_weight=0.2, crossover_probability=0.7,
                                                   strategies=(cross_curr2best1, cross_best2), p_max=5)
    best = algo.run(task)
    print('%s -> %s' % (best[0], best[1]))
