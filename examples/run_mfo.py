# encoding=utf8
# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys

sys.path.append('../')
# End of fix

from niapy.algorithms.basic import MothFlameOptimizer
from niapy.task import StoppingTask
from niapy.benchmarks import Sphere

# we will run Moth Flame Optimizer for 5 independent runs
for i in range(5):
    task = StoppingTask(max_evals=10000, dimension=10, benchmark=Sphere())
    algo = MothFlameOptimizer(population_size=30)
    best = algo.run(task)
    print('%s -> %s' % (best[0], best[1]))
