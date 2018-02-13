from unittest import TestCase

from NiaPy.algorithms.basic import FlowerPollinationAlgorithm


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


class FPATestCase(TestCase):
    def setUp(self):
        self.fpa_custom = FlowerPollinationAlgorithm(
            10, 20, 10000, 0.5, MyBenchmark())

    def test_custom_works_fine(self):
        self.assertTrue(self.fpa_custom.run())
