# pylint: disable=old-style-class
from unittest import TestCase
from numpy import array_equal

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


class PSOTestCase(TestCase):

    def setUp(self):
        self.pso_custom = ParticleSwarmAlgorithm(NP=40, D=40, nFES=1000, C1=2.0, C2=2.0, w=0.7, vMin=-4, vMax=4, benchmark=MyBenchmark(), seed=1)
        self.pso_customc = ParticleSwarmAlgorithm(NP=40, D=40, nFES=1000, C1=2.0, C2=2.0, w=0.7, vMin=-4, vMax=4, benchmark=MyBenchmark(), seed=1)
        self.pso_griewank = ParticleSwarmAlgorithm(NP=40, D=40, nFES=1000, C1=2.0, C2=2.0, w=0.7, vMin=-4, vMax=4, benchmark='griewank', seed=1)
        self.pso_griewankc = ParticleSwarmAlgorithm(NP=40, D=40, nFES=1000, C1=2.0, C2=2.0, w=0.7, vMin=-4, vMax=4, benchmark='griewank', seed=1)

    def test_custom_works_fine(self):
        x = self.pso_custom.run()
        self.assertTrue(x)
        y = self.pso_customc.run()
        self.assertTrue(y)
        self.assertTrue(array_equal(x[0], y[0]))
        self.assertEqual(x[1], y[1])

    def test_griewank_works_fine(self):
        x = self.pso_griewank.run()
        self.assertTrue(x)
        y = self.pso_griewankc.run()
        self.assertTrue(y)
        self.assertTrue(array_equal(x[0], y[0]))
        self.assertEqual(x[1], y[1])
