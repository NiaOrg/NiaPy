# encoding=utf8
# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys
sys.path.append('../')
# End of fix

from NiaPy.algorithms.modified import SelfAdaptiveDifferentialEvolution
from NiaPy.task.task import StoppingTask
from NiaPy.task.task import OptimizationType
from NiaPy.benchmarks import Sphere

# we will run jDE algorithm for 5 independent runs
for i in range(5):
	task = StoppingTask(D=10, nFES=10000, optType=OptimizationType.MINIMIZATION, benchmark=Sphere())
	algo = SelfAdaptiveDifferentialEvolution(NP=40, F=0.5, F_l=0.0, F_u=2.0, Tao1=0.9, CR=0.5, Tao2=0.45)
	best = algo.run(task=task)
	print('%s -> %s' % (best[0].x, best[1]))

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3

