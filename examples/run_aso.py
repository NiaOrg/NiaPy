# encoding=utf8
# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys
sys.path.append('../')
# End of fix

from NiaPy.algorithms.other import AnarchicSocietyOptimization
from NiaPy.algorithms.other.aso import Elitism, Sequential, Crossover
from NiaPy.util import StoppingTask, OptimizationType
from NiaPy.benchmarks import Sphere

# we will run Anarchic Society Optimization for 5 independent runs
for i in range(5):
    task = StoppingTask(D=10, nFES=10000, optType=OptimizationType.MINIMIZATION, benchmark=Sphere())
    algo = AnarchicSocietyOptimization(NP=40, Combination=Elitism)
    best = algo.run(task=task)
    print best
