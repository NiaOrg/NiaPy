# encoding=utf8
# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys

sys.path.append('../')
# End of fix

from niapy.algorithms.modified import SelfAdaptiveDifferentialEvolution
from niapy.task import StoppingTask
from niapy.benchmarks import Griewank

# we will run jDE algorithm for 5 independent runs
algo = SelfAdaptiveDifferentialEvolution(f_lower=0.0, f_upper=2.0, tao1=0.9, tao2=0.45, population_size=40,
                                         differential_weight=0.5, crossover_probability=0.5)
for i in range(5):
    task = StoppingTask(max_evals=10000, enable_logging=True, dimension=10, benchmark=Griewank(lower=-600, upper=600))
    best = algo.run(task)
    print('%s -> %s' % (best[0], best[1]))
print(algo.get_parameters())

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
