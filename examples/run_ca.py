# encoding=utf8
# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys
sys.path.append('../')
# End of fix

from NiaPy.task import StoppingTask
from NiaPy.benchmarks import Sphere
from NiaPy.algorithms.basic import CamelAlgorithm

#we will run Camel Algorithm for 5 independent runs
for i in range(5):
    task = StoppingTask(D=10, nFES=1000, benchmark=Sphere())
    algo = CamelAlgorithm(NP=40)
    best = algo.run(task=task)
    print('%s -> %s' % (best[0], best[1]))
