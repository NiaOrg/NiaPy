# encoding=utf8
# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys
sys.path.append('../')
# End of fix

from NiaPy.task import StoppingTask
from NiaPy.benchmarks import Sphere
from NiaPy.algorithms.other import RandomSearch

for i in range(1):
    task = StoppingTask(D=5, nGEN=5000, benchmark=Sphere())
    algo = RandomSearch()
    best = algo.run(task=task)
    print(best)
