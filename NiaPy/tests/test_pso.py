# pylint: disable=old-style-class
from unittest import TestCase

from NiaPy.algorithms.basic import ParticleSwarmAlgorithm


class MyBenchmark:

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
        self.pso_custom = ParticleSwarmAlgorithm(NP=40, D=40, nFES=1000, C1=2.0, C2=2.0, w=0.7, vMin=-4, vMax=4, benchmark=MyBenchmark())
        self.pso_griewank = ParticleSwarmAlgorithm(NP=40, D=40, nFES=1000, C1=2.0, C2=2.0, w=0.7, vMin=-4, vMax=4, benchmark='griewank')

    def test_custom_works_fine(self):
        self.assertTrue(self.pso_custom.run())

    def test_griewank_works_fine(self):
        self.assertTrue(self.pso_griewank.run())
