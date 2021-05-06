# encoding=utf8
# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys

sys.path.append('../')
# End of fix

from niapy.algorithms.basic import OppositionVelocityClampingParticleSwarmOptimization
from niapy.task import StoppingTask
from niapy.benchmarks import Sphere

# we will run ParticleSwarmAlgorithm for 5 independent runs
algo = OppositionVelocityClampingParticleSwarmOptimization(population_size=50, c1=0.83, c2=1.6, w=0.6, min_velocity=-1.5,
                                                           max_velocity=1.5)
for i in range(5):
    task = StoppingTask(max_evals=1000, dimension=10, benchmark=Sphere())
    best = algo.run(task=task)
    print('%s -> %f' % (best[0], best[1]))
# print(algo.get_parameters())

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
