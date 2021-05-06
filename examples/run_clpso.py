# encoding=utf8
# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys

sys.path.append('../')
# End of fix

from niapy.algorithms.basic import ComprehensiveLearningParticleSwarmOptimizer
from niapy.benchmarks import Sphere
from niapy.task import StoppingTask

# we will run ParticleSwarmAlgorithm for 5 independent runs
algo = ComprehensiveLearningParticleSwarmOptimizer(population_size=50, C1=.3, C2=1.0, m=5, w=0.86, vMin=-2, vMax=2)
for i in range(5):
    task = StoppingTask(max_evals=20000, dimension=25, benchmark=Sphere())
    best = algo.run(task=task)
    print('%s -> %f' % (best[0], best[1]))
print(algo.get_parameters())

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
