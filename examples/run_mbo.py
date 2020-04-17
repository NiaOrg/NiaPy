# encoding=utf8
# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys
sys.path.append('../')

from NiaPy.algorithms.basic import MonarchButterflyOptimization
from NiaPy.task import StoppingTask
from NiaPy.benchmarks import Sphere

# we will run Monarch Butterfly Optimization algorithm for 5 independent runs
for i in range(5):
    task = StoppingTask(D=10, nFES=10000, benchmark=Sphere())
    algo = MonarchButterflyOptimization(NP=20, PAR=5.0 / 12.0, PER=1.2)
    best = algo.run(task)
    print('%s -> %s' % (best[0], best[1]))
