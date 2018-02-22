from unittest import TestCase

from NiaPy.algorithms.basic import GreyWolfOptimizer


class MyBenchmark(object):

    def __init__(self):
        self.Lower = -5.12
        self.Upper = 5.12

    @classmethod
    def function(cls):
        def evaluate(D, sol):
            val = 0.0
            for i in range(D):
                val = val + sol[i] * sol[i]
            return val
        return evaluate


class BATestCase(TestCase):

    def setUp(self):
        self.gwo_custom = GreyWolfOptimizer(10, 20, 10000, MyBenchmark())
        self.gwo_griewank = GreyWolfOptimizer(10, 20, 10000, 'griewank')

    def test_custom_works_fine(self):
        self.assertTrue(self.gwo_custom.run())

    def test_griewank_works_fine(self):
        self.assertTrue(self.gwo_griewank.run())
