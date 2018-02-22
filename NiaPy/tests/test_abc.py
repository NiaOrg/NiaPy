from unittest import TestCase

from NiaPy.algorithms.basic import ArtificialBeeColonyAlgorithm


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


class ABCTestCase(TestCase):

    def setUp(self):
        self.abc_custom = ArtificialBeeColonyAlgorithm(
            10, 40, 10000, MyBenchmark())
        self.abc_griewank = ArtificialBeeColonyAlgorithm(
            10, 40, 10000, 'griewank')

    def test_custom_works_fine(self):
        self.assertTrue(self.abc_custom.run())

    def test_griewank_works_fine(self):
        self.assertTrue(self.abc_griewank.run())
