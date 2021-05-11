# encoding=utf8
# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys

sys.path.append('../')
# End of fix

from niapy.algorithms.modified import HybridSelfAdaptiveBatAlgorithm
from niapy.task import StoppingTask
from niapy.benchmarks import Griewank

# we will run Bat Algorithm for 5 independent runs
algo = HybridSelfAdaptiveBatAlgorithm(population_size=50)
for i in range(5):
    task = StoppingTask(max_iters=10000, dimension=10, benchmark=Griewank(upper=600, lower=-600))
    best = algo.run(task)
    print('%s -> %s' % (best[0], best[1]))
print(algo.get_parameters())

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
