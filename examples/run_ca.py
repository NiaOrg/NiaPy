# encoding=utf8
# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys

sys.path.append('../')
# End of fix

from niapy.task import StoppingTask
from niapy.benchmarks import Sphere
from niapy.algorithms.basic import CamelAlgorithm

# we will run Camel Algorithm for 5 independent runs
for i in range(5):
    task = StoppingTask(max_evals=1000, dimension=10, benchmark=Sphere())
    algo = CamelAlgorithm(population_size=40)
    best = algo.run(task=task)
    print('%s -> %s' % (best[0], best[1]))
