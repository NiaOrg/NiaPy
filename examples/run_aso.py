# encoding=utf8
# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys

sys.path.append('../')
# End of fix

from niapy.algorithms.other import AnarchicSocietyOptimization
from niapy.algorithms.other.aso import elitism
from niapy.task import StoppingTask
from niapy.benchmarks import Sphere

# we will run Anarchic Society Optimization for 5 independent runs
for i in range(5):
    task = StoppingTask(max_evals=6500, dimension=10, benchmark=Sphere())
    algo = AnarchicSocietyOptimization(population_size=40, Combination=elitism)
    best = algo.run(task)
    print('%s -> %s' % (best[0], best[1]))
