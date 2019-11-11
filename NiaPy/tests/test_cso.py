# encoding=utf8

from NiaPy.tests.test_algorithm import AlgorithmTestCase, MyBenchmark
from NiaPy.algorithms.basic import CatSwarmOptimization


class CSOTestCase(AlgorithmTestCase):
    def setUp(self):
        AlgorithmTestCase.setUp(self)
        self.algo = CatSwarmOptimization

    def test_custom_works_fine(self):
        cso_custom = self.algo(NP=20, seed=self.seed)
        cso_customc = self.algo(NP=20, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, cso_custom, cso_customc, MyBenchmark())

    def test_griewank_works_fine(self):
        cso_griewank = self.algo(NP=10, seed=self.seed)
        cso_griewankc = self.algo(NP=10, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, cso_griewank, cso_griewankc)
