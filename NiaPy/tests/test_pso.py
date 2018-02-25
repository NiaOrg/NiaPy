from unittest import TestCase

from NiaPy.algorithms.basic import ParticleSwarmAlgorithm


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
        self.pso_custom = ParticleSwarmAlgorithm(
            40, 40, 10000, 2.0, 2.0, 0.7, -4, 4, MyBenchmark())
        self.pso_griewank = ParticleSwarmAlgorithm(
            40, 40, 10000, 2.0, 2.0, 0.7, -4, 4, 'griewank')

    def test_custom_works_fine(self):
        self.assertTrue(self.pso_custom.run())

    def test_griewank_works_fine(self):
        self.assertTrue(self.pso_griewank.run())
