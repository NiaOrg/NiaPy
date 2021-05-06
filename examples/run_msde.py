# encoding=utf8
# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys

sys.path.append('../')
# End of fix

from niapy.algorithms.basic import MultiStrategyDifferentialEvolution
from niapy.task import StoppingTask
from niapy.benchmarks import Sphere

# we will run Differential Evolution for 5 independent runs
for i in range(5):
    task = StoppingTask(max_evals=5000, dimension=10, benchmark=Sphere())
    algo = MultiStrategyDifferentialEvolution(population_size=50, differential_weight=0.5, crossover_probability=0.9)
    best = algo.run(task)
    print('%s -> %s' % (best[0], best[1]))

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
