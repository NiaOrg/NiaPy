# encoding=utf8
# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys

sys.path.append('../')
# End of fix

from niapy.algorithms.basic import BacterialForagingOptimizationAlgorithm
from niapy.task import StoppingTask
from niapy.benchmarks import Griewank

# we will run Bacterial Foraging Optimization Algorithm for 5 independent runs
for i in range(5):
    task = StoppingTask(max_iters=10000, enable_logging=True, dimension=10, benchmark=Griewank(lower=-600, upper=600))
    algo = BacterialForagingOptimizationAlgorithm()
    best = algo.run(task)
    print('%s -> %s' % (best[0], best[1]))
