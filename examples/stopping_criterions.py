# encoding=utf8
# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys

sys.path.append('../')
# End of fix

from niapy.task import StoppingTask
from niapy.algorithms.basic import DifferentialEvolution
from niapy.benchmarks import Sphere

# 1 Number of function evaluations (nFES) as a stopping criteria
for i in range(10):
    task = StoppingTask(max_evals=10000, dimension=10, benchmark=Sphere())
    algo = DifferentialEvolution(population_size=40, CR=0.9, F=0.5)
    best = algo.run(task)
    print('%s -> %s' % (best[0], best[1]))

print('---------------------------------------')

# 2 Number of generations (iterations) as a stopping criteria
for i in range(10):
    task = StoppingTask(max_iters=1000, dimension=10, benchmark=Sphere())
    algo = DifferentialEvolution(population_size=40, CR=0.9, F=0.5)
    best = algo.run(task)
    print('%s -> %s' % (best[0], best[1]))

print('---------------------------------------')

# 3 Reference value as a stopping criteria
for i in range(10):
    task = StoppingTask(cutoff_value=50.0, dimension=10, benchmark=Sphere())
    algo = DifferentialEvolution(population_size=40, CR=0.9, F=0.5)
    best = algo.run(task)
    print('%s -> %s' % (best[0], best[1]))
