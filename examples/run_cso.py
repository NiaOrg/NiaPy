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
    task = StoppingTask(D=10, nFES=100000, optType=OptimizationType.MINIMIZATION, benchmark=Sphere())
    algo = CatSwarmOptimization(NP = 50, MR = 0.05)
    best = algo.run(task=task)
    #print(best,'\n', Sphere().function()(task.D,best[0]))
    print('%s -> %s' % (best[0], best[1]))
    sum = 0
    for i in best[0]:
        sum += i**2
    print(sum)