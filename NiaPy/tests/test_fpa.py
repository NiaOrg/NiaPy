from unittest import TestCase

from NiaPy.algorithms.basic import FlowerPollinationAlgorithm


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


class FPATestCase(TestCase):

    def setUp(self):
        self.fpa_custom = FlowerPollinationAlgorithm(
            10, 20, 10000, 0.5, MyBenchmark())

        self.fpa_griewank = FlowerPollinationAlgorithm(
            10, 20, 10000, 0.5, 'griewank')

    def test_custom_works_fine(self):
        self.assertTrue(self.fpa_custom.run())

    def test_griewank_works_fine(self):
        self.assertTrue(self.fpa_griewank.run())
