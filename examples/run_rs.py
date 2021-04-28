# encoding=utf8
# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys
sys.path.append('../')
# End of fix

from niapy.task import StoppingTask
from niapy.benchmarks import Sphere
from niapy.algorithms.other import RandomSearch

for i in range(1):
    task = StoppingTask(D=5, nGEN=5000, benchmark=Sphere())
    algo = RandomSearch()
    best = algo.run(task=task)
    print(best)
