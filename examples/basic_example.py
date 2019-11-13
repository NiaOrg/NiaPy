# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys
sys.path.append('../')
# End of fix

from NiaPy.algorithms.basic import GreyWolfOptimizer
from NiaPy.task import StoppingTask

# we will run 10 repetitions of Grey Wolf Optimizer against Pinter benchmark function
for i in range(10):
    task = StoppingTask(D=10, nFES=1000, benchmark='pinter')
    algorithm = GreyWolfOptimizer(NP=20)
    best = algorithm.run(task)
    print(best)
