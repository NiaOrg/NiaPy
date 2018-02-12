from unittest import TestCase

from NiaPy.algorithms.basic import BatAlgorithm


class MyBenchmark(object):

    def __init__(self):
        self.Lower = -11
        self.Upper = 11

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
        self.ba_custom = BatAlgorithm(
            10, 20, 10000, 0.5, 0.5, 0.0, 2.0, MyBenchmark())
        self.ba_griewank = BatAlgorithm(
            10, 40, 10000, 0.5, 0.5, 0.0, 2.0, 'griewank')

    def test_custom_works_fine(self):
        self.assertTrue(self.ba_custom.run())

    def test_griewank_works_fine(self):
        self.assertTrue(self.ba_griewank.run())
