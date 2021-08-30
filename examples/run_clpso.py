# encoding=utf8
# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys

sys.path.append('../')
# End of fix

from niapy.algorithms.basic import ComprehensiveLearningParticleSwarmOptimizer
from niapy.problems import Sphere
from niapy.task import Task

# we will run ParticleSwarmAlgorithm for 5 independent runs
algo = ComprehensiveLearningParticleSwarmOptimizer(population_size=50, c1=.3, c2=1.0, m=5, w=0.86, min_velocity=-2, max_velocity=2)
for i in range(5):
    task = Task(problem=Sphere(dimension=25), max_evals=20000)
    best = algo.run(task=task)
    print('%s -> %f' % (best[0], best[1]))
print(algo.get_parameters())
