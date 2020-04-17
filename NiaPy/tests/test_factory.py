# encoding=utf8

from unittest import TestCase

import numpy as np

from NiaPy import Factory
from NiaPy.benchmarks.benchmark import Benchmark

class NoLimits:
    @classmethod
    def function(cls):
        def evaluate(D, x): return 0
        return evaluate


class MyBenchmark(Benchmark):
    def __init__(self):
        Benchmark.__init__(self, -10, 10)

    def function(self):
        return lambda D, x, **kwargs: np.sum(x ** 2)


class FactoryTestCase(TestCase):
    def setUp(self):
        self.factory = Factory()

    def test_get_bad_benchmark_fine(self):
        self.assertRaises(TypeError, lambda: self.factory.get_benchmark('hihihihihihihihihi'))
        self.assertRaises(TypeError, lambda: self.factory.get_benchmark(MyBenchmark))
        self.assertRaises(TypeError, lambda: self.factory.get_benchmark(NoLimits))

    def test_get_bad_algorithm_fine(self):
        self.assertRaises(TypeError, lambda: self.factory.get_algorithm('hihihihihihihihihi'))
        self.assertRaises(TypeError, lambda: self.factory.get_algorithm(MyBenchmark))
        self.assertRaises(TypeError, lambda: self.factory.get_algorithm(NoLimits))
