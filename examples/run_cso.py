# encoding=utf8
# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys
sys.path.append('../')
# End of fix

from NiaPy.task import StoppingTask
from NiaPy.benchmarks import Sphere
from NiaPy.algorithms.basic import CatSwarmOptimization

task = StoppingTask(D=10, nFES=1000, logger=True, benchmark=Sphere())
algo = CatSwarmOptimization()
best = algo.run(task=task)
print('%s -> %s' % (best[0], best[1]))
#plot a convergence graph
task.plot()