# encoding=utf8
# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys

from niapy.algorithms.basic import CovarianceMatrixAdaptionEvolutionStrategy
from niapy.problems import Sphere
from niapy.task import Task

sys.path.append('../')
# End of fix

# we will run CMA-ES for 5 independent runs
for i in range(5):
    task = Task(problem=Sphere(dimension=10), max_evals=1000, enable_logging=True)
    algo = CovarianceMatrixAdaptionEvolutionStrategy(population_size=20)
    best = algo.run(task)
    print('%s -> %s' % (best[0], best[1]))
