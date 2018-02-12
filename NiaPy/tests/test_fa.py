from unittest import TestCase

from NiaPy.algorithms.basic import FireflyAlgorithm


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


class FATestCase(TestCase):
    def setUp(self):
        self.fa = FireflyAlgorithm(
            10, 20, 10000, 0.5, 0.2, 1.0, MyBenchmark())

    def test_works_fine(self):
        self.assertTrue(self.fa.run())
