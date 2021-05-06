# encoding=utf8
# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys

sys.path.append('../')
# End of fix

from niapy.algorithms.basic import CuckooSearch
from niapy.benchmarks import Sphere
from niapy.task import StoppingTask

# we will run Cuckoo Search for 5 independent runs
for i in range(5):
    task = StoppingTask(max_evals=10000, dimension=10, benchmark=Sphere())
    algo = CuckooSearch(N=100, pa=0.95, alpha=1)
    best = algo.run(task)
    print(best)
