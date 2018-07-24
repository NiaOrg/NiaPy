# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys
sys.path.append('../')
# End of fix

import random
import logging
import matplotlib.pyplot as plt
from NiaPy.algorithms.basic import FireflyAlgorithm

logging.basicConfig()
logger = logging.getLogger('examples')
logger.setLevel('INFO')

# For reproducive results
random.seed(1234)

global_vector = []


class MyBenchmark(object):
    def __init__(self):
        self.Lower = -11
        self.Upper = 11

    def function(self):
        def evaluate(D, sol):
            val = 0.0
            for i in range(D):
                val = val + sol[i] * sol[i]
            global_vector.append(val)
            return val
        return evaluate


for i in range(10):
    Algorithm = FireflyAlgorithm(D=10, NP=20, nFES=1000, alpha=0.5, betamin=0.2, gamma=1.0, benchmark=MyBenchmark())
    Best = Algorithm.run()
    plt.plot(global_vector)
    global_vector = []
    logger.info(Best)
    
plt.xlabel('Number of evaluations')
plt.ylabel('Fitness function value')
plt.title('Convergence plot')
plt.show()
