from unittest import TestCase

from NiaPy.algorithms.basic import CuckooSearchAlgorithm


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


class CSTestCase(TestCase):

    def setUp(self):
        self.cs_custom = CuckooSearchAlgorithm(
            40, 40, 10000, 0.25, 0.01, MyBenchmark())

        self.cs_sphere = CuckooSearchAlgorithm(
            40, 40, 10000, 0.25, 0.01, 'sphere')

    def test_custom_works_fine(self):
        self.assertTrue(self.cs_custom.run())

    def test_sphere_works_fine(self):
        self.assertTrue(self.cs_sphere.run())
