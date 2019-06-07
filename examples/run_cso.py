# encoding=utf8
# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys
sys.path.append('../')
# End of fix

from NiaPy.algorithms.basic import CatSwarmOptimization
from NiaPy.task.task import StoppingTask, OptimizationType
from NiaPy.benchmarks import Sphere

# we will run Cat Swarm Optimization for 5 independent runs
for i in range(1):
    task = StoppingTask(D=10, nFES=1000, optType=OptimizationType.MINIMIZATION, logger=True, benchmark=Sphere())
    algo = CatSwarmOptimization()
    best = algo.run(task=task)
    #print(best,'\n', Sphere().function()(task.D,best[0]))
    print('%s -> %s' % (best[0], best[1]))

    #plot a convergence graph
    task.plot()