# encoding=utf8
# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys
sys.path.append('../')
# End of fix

from NiaPy.algorithms.modified import SelfAdaptiveDifferentialEvolution
from NiaPy.task import StoppingTask
from NiaPy.benchmarks import Griewank

# we will run jDE algorithm for 5 independent runs
algo = SelfAdaptiveDifferentialEvolution(NP=40, F=0.5, F_l=0.0, F_u=2.0, Tao1=0.9, CR=0.5, Tao2=0.45)
for i in range(5):
	task = StoppingTask(D=10, nFES=10000, benchmark=Griewank(Lower=-600, Upper=600), logger=True)
	best = algo.run(task)
	print('%s -> %s' % (best[0], best[1]))
print(algo.getParameters())

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3

