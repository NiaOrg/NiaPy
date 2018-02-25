from unittest import TestCase

from NiaPy.algorithms.modified import HybridBatAlgorithm


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


class HBATestCase(TestCase):
    def setUp(self):

        self.hba_custom = HybridBatAlgorithm(
            10, 40, 1000, 0.5, 0.5, 0.5, 0.9, 0.0, 2.0, MyBenchmark())
        self.hba_griewank = HybridBatAlgorithm(
            10, 40, 1000, 0.5, 0.5, 0.5, 0.9, 0.0, 2.0, 'griewank')

    def test_custom_works_fine(self):
        self.assertTrue(self.hba_custom.run())

    def test_griewank_works_fine(self):
        self.assertTrue(self.hba_griewank.run())
