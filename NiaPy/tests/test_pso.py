# pylint: disable=old-style-class
from NiaPy.tests.test_algorithm import AlgorithmTestCase, MyBenchmark
from numpy import array_equal

from NiaPy.algorithms.basic import ParticleSwarmAlgorithm


class PSOTestCase(AlgorithmTestCase):

    def test_custom_works_fine(self):
        pso_custom = ParticleSwarmAlgorithm(NP=40, D=self.D, nFES=self.nFES, nGEN=self.nGEN, C1=2.0, C2=2.0, w=0.7, vMin=-4, vMax=4, benchmark=MyBenchmark(), seed=self.seed)
        pso_customc = ParticleSwarmAlgorithm(NP=40, D=self.D, nFES=self.nFES, nGEN=self.nGEN, C1=2.0, C2=2.0, w=0.7, vMin=-4, vMax=4, benchmark=MyBenchmark(), seed=self.seed)
        AlgorithmTestCase.algorithm_run_test(self, pso_custom, pso_customc)

    def test_griewank_works_fine(self):
        pso_griewank = ParticleSwarmAlgorithm(NP=40, D=self.D, nFES=self.nFES, nGEN=self.nGEN, C1=2.0, C2=2.0, w=0.7, vMin=-4, vMax=4, benchmark='griewank', seed=self.seed)
        pso_griewankc = ParticleSwarmAlgorithm(NP=40, D=self.D, nFES=self.nFES, nGEN=self.nGEN, C1=2.0, C2=2.0, w=0.7, vMin=-4, vMax=4, benchmark='griewank', seed=self.seed)
        AlgorithmTestCase.algorithm_run_test(self, pso_griewank, pso_griewankc)
