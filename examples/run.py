# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys
sys.path.append('../')
# End of fix

import random
import logging
import NiaPy

logging.basicConfig()
logger = logging.getLogger('examples')
logger.setLevel('INFO')

# For reproducive results
random.seed(1234)

algorithms = ['BatAlgorithm', 'DifferentialEvolutionAlgorithm']
benchmarks = ['griewank', 'ackley', 'sphere']

NiaPy.Runner(10, 40, 1000, 10, algorithms, benchmarks).run()