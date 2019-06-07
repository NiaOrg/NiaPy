# encoding=utf8
# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys
sys.path.append('../')
# End of fix

from NiaPy.algorithms.basic import CatSwarmOptimization
from NiaPy.util import StoppingTask, OptimizationType, ThrowingTask
from NiaPy.benchmarks import Ackley, Sphere, Griewank

# we will run Cat Swarm Optimization for 5 independent runs
for i in range(5):
    task = StoppingTask(D=10, nFES=10000, optType=OptimizationType.MINIMIZATION, benchmark=Griewank())
    algo = CatSwarmOptimization(vMax = 1.9)
    best = algo.run(task=task)
    #print(best,'\n', Sphere().function()(task.D,best[0]))
    print('%s -> %s' % (best[0], best[1]))
