# encoding=utf8
# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys
sys.path.append('../')
# End of fix

from NiaPy.algorithms.basic import CuckooSearch
from NiaPy.task.task import StoppingTask, OptimizationType
from NiaPy.benchmarks import Sphere

# we will run Cuckoo Search for 5 independent runs
for i in range(5):
    task = StoppingTask(D=10, nFES=10000, optType=OptimizationType.MINIMIZATION, benchmark=Sphere())
    algo = CuckooSearch(N=100, pa=0.95, alpha=1)
    best = algo.run(task=task)
    print(best)
 
