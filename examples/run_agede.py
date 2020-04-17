# encoding=utf8
# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys
sys.path.append('../')
# End of fix

import random
from NiaPy.algorithms.basic import AgingNpDifferentialEvolution
from NiaPy.algorithms.basic.de import bilinear
from NiaPy.task import StoppingTask
from NiaPy.benchmarks import Sphere

# we will run Differential Evolution for 5 independent runs
for i in range(5):
	task = StoppingTask(D=10, nFES=10000, benchmark=Sphere())
	algo = AgingNpDifferentialEvolution(NP=40, F=0.63, CR=0.9, Lt_min=3, Lt_max=7, omega=0.2, delta_np=0.1, age=bilinear)
	best = algo.run(task)
	print('%s -> %s' % (best[0], best[1]))

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
