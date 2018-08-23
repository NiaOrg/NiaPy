# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys
sys.path.append('../')
# End of fix

import random
import logging
from NiaPy import Runner
from NiaPy.benchmarks import Griewank

logging.basicConfig()
logger = logging.getLogger('examples')
logger.setLevel('INFO')

# For reproducive results
random.seed(1234)

class MyBenchmark(object):
    def __init__(self):
        self.Lower = -5.12
        self.Upper = 5.12

    def function(self):
        def evaluate(D, sol):
            val = 0.0
            for i in range(D):
                val = val + sol[i] * sol[i]
            return val
        return evaluate

# example using custom benchmark "MyBenchmark"
logger.info('Running with custom MyBenchmark...')
for i in range(10):
    Algorithm = Runner.getAlgorithm('BA')(D=10, NP=40, nFES=10000, A=0.5, r=0.5, Qmin=0.0, Qmax=2.0, benchmark=MyBenchmark())
    Best = Algorithm.run()
    logger.info(Best)

# example using predifined benchmark function
# available benchmarks are:
# - griewank
# - rastrigin
# - rosenbrock
# - sphere
logger.info('Running with default Griewank benchmark...')

griewank = Griewank()

for i in range(10):
    Algorithm = Runner.getAlgorithm('BA')(D=10, NP=40, nFES=10000, A=0.5, r=0.5, Qmin=0.0, Qmax=2.0, benchmark=griewank)
    Best = Algorithm.run()
    Best = Algorithm.run()
    logger.info(Best)

logger.info(
    'Running with default Griewank benchmark - should be the same as previous implementataion...')

for i in range(10):
    Algorithm = Runner.getAlgorithm('BA')(D=10, NP=40, nFES=10000, A=0.5, r=0.5, Qmin=0.0, Qmax=2.0, benchmark='griewank')
    Best = Algorithm.run()

    logger.info(Best)

# example with changed griewank's lower and upper bounds
logger.info('Running with Griewank with changed Upper and Lower bounds...')

griewank = Griewank(-50, 50)

for i in range(10):
    Algorithm = Runner.getAlgorithm('BA')(D=10, NP=40, nFES=10000, A=0.5, r=0.5, Qmin=0.0, Qmax=2.0, benchmark=griewank)
    Best = Algorithm.run()
    logger.info(Best)
