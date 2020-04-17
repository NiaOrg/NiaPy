# encoding=utf8
# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys
sys.path.append('../')
# End of fix

from NiaPy.task import StoppingTask
from NiaPy.benchmarks import Sphere
from NiaPy.algorithms.basic import DynamicFireworksAlgorithm


# we will run Fireworks Algorithm for 5 independent runs
algo = DynamicFireworksAlgorithm(N=70, Ainit=0.1, Afinal=0.9)
for i in range(5):
	task = StoppingTask(D=10, nGEN=80, benchmark=Sphere())
	best = algo.run(task)
	print('%s -> %s' % (best[0], best[1]))
print(algo.getParameters())

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
