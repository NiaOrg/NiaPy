# encoding=utf8
# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys

sys.path.append('../')
# End of fix

from niapy.algorithms.basic import AgingNpDifferentialEvolution
from niapy.algorithms.basic.de import bilinear
from niapy.task import Task
from niapy.problems import Sphere

# we will run Differential Evolution for 5 independent runs
for i in range(5):
    task = Task(problem=Sphere(dimension=10), max_evals=10000)
    algo = AgingNpDifferentialEvolution(population_size=40, differential_weight=0.63, crossover_probability=0.9, min_lifetime=3, max_lifetime=7, omega=0.2, delta_np=0.1,
                                        age=bilinear)
    best = algo.run(task)
    print('%s -> %s' % (best[0], best[1]))
