# encoding=utf8
# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys
sys.path.append('../')
# End of fix

from NiaPy.algorithms.modified import SelfAdaptiveBatAlgorithm
from NiaPy.task import StoppingTask
from NiaPy.benchmarks import Griewank

# we will run Bat Algorithm for 5 independent runs
algo = SelfAdaptiveBatAlgorithm()
for i in range(5):
    task = StoppingTask(D=10, nGEN=10000, benchmark=Griewank(Lower=-600, Upper=600))
    best = algo.run(task)
    print('%s -> %s' % (best[0], best[1]))
print(algo.getParameters())

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
