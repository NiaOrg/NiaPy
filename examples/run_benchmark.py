# encoding=utf8
# This is temporary fix to import module from parent folder
# It will be removed when package is published on PyPI
import sys

sys.path.append('../')
# End of fix

import logging

from niapy.benchmarks import Benchmark, Katsuura, Elliptic

logging.basicConfig()
logger = logging.getLogger('examples')
logger.setLevel('INFO')


class MyBenchmark(Benchmark):
    Name = ['MyBenchmark']

    def __init__(self):
        super().__init__(-11, 11)

    def function(self):
        def evaluate(d, x):
            val = 0.0
            for i in range(d):
                val += x[i] ** 2
            return val

        return evaluate


benchmark = MyBenchmark()
benchmark.plot3d()

benc = Katsuura(-1, 1)
benc.plot3d(0.06)

benc = Elliptic()
benc.plot3d(0.65)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
