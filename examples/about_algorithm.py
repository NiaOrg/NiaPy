# encoding=utf8
# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys
sys.path.append('../')
# End of fix

from NiaPy.algorithms.basic import FireflyAlgorithm

algo = FireflyAlgorithm()
print(algo.algorithmInfo())
