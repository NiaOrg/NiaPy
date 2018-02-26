from unittest import TestCase

from NiaPy.algorithms.basic import GeneticAlgorithm


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


class GATestCase(TestCase):
    def setUp(self):
        self.ga_custom = GeneticAlgorithm(10, 40, 10000, 4, 0.05, 0.4, MyBenchmark())
        self.ga_griewank = GeneticAlgorithm(10, 40, 10000, 4, 0.05, 0.4, 'griewank')

    def test_custom_works_fine(self):
        self.assertTrue(self.ga_custom.run())

    def test_griewank_works_fine(self):
        self.assertTrue(self.ga_griewank.run())
