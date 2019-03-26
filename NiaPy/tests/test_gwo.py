# encoding=utf8
# pylint: disable=too-many-function-args
from NiaPy.tests.test_algorithm import AlgorithmTestCase, MyBenchmark
from NiaPy.algorithms.basic import GreyWolfOptimizer


class GWOTestCase(AlgorithmTestCase):

    def test_custom_works_fine(self):
        gwo_custom = GreyWolfOptimizer(NP=20, seed=self.seed)
        gwo_customc = GreyWolfOptimizer(NP=20, seed=self.seed)
        AlgorithmTestCase.algorithm_run_test(self, gwo_custom, gwo_customc, MyBenchmark())

    def test_griewank_works_fine(self):
        gwo_griewank = GreyWolfOptimizer(NP=10, seed=self.seed)
        gwo_griewankc = GreyWolfOptimizer(NP=10, seed=self.seed)
        AlgorithmTestCase.algorithm_run_test(self, gwo_griewank, gwo_griewankc)
