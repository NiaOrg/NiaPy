# encoding=utf8
# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys

sys.path.append('../')
# End of fix

from niapy.algorithms.basic import DynNpMultiStrategyDifferentialEvolution
from niapy.algorithms.basic.de import cross_best2, cross_curr2best1
from niapy.task import StoppingTask
from niapy.benchmarks import Sphere

# we will run Differential Evolution for 5 independent runs
for i in range(5):
    task = StoppingTask(max_evals=10000, dimension=10, benchmark=Sphere())
    algo = DynNpMultiStrategyDifferentialEvolution(population_size=80, F=0.2, CR=0.7,
                                                   strategies=(cross_curr2best1, cross_best2), pmax=5)
    best = algo.run(task)
    print('%s -> %s' % (best[0], best[1]))
