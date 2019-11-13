# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys
sys.path.append('../')
# End of fix

from NiaPy.algorithms.basic import GreyWolfOptimizer
from NiaPy.task import StoppingTask, OptimizationType
from NiaPy.benchmarks import Pinter

# initialize Pinter benchamrk with custom bound
pinterCustom = Pinter(-5, 5)

# we will run 10 repetitions of Grey Wolf Optimizer against Pinter benchmark function
for i in range(10):
    # first parameter takes dimension of problem
    # second parameter takes the number of function evaluations
    # third parameter is benchmark optimization type
    # forth parameter is benchmark function
    task = StoppingTask(D=20, nGEN=100, optType=OptimizationType.MINIMIZATION, benchmark=pinterCustom)

    # parameter is population size
    algo = GreyWolfOptimizer(NP=20)

    # running algorithm returns best found minimum
    best = algo.run(task)

    # printing best minimum
    print(best)
