# encoding=utf8
# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys

sys.path.append('../')
# End of fix

from niapy.algorithms.other import NelderMeadMethod
from niapy.task import StoppingTask
from niapy.benchmarks import Sphere

# we will run Nelder Mead algorithm for 5 independent runs
for i in range(5):
    task = StoppingTask(max_iters=1000, dimension=10, benchmark=Sphere())
    algo = NelderMeadMethod(population_size=70, alpha=0.2, gamma=0.1, rho=-0.24, sigma=-0.1)
    best = algo.run(task)
    print('%s -> %s' % (best[0], best[1]))
