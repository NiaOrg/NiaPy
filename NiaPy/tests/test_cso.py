# encoding=utf8
# pylint: disable=too-many-function-args
from NiaPy.tests.test_algorithm import AlgorithmTestCase, MyBenchmark
from NiaPy.algorithms.basic import CatSwarmOptimization


class CSOTestCase(AlgorithmTestCase):
    def test_custom_works_fine(self):
        cso_custom = CatSwarmOptimization(NP=20, seed=self.seed)
        cso_customc = CatSwarmOptimization(NP=20, seed=self.seed)
        AlgorithmTestCase.algorithm_run_test(self, cso_custom, cso_customc, MyBenchmark())

    def test_griewank_works_fine(self):
        cso_griewank = CatSwarmOptimization(NP=10, seed=self.seed)
        cso_griewankc = CatSwarmOptimization(NP=10, seed=self.seed)
        AlgorithmTestCase.algorithm_run_test(self, cso_griewank, cso_griewankc)
