# encoding=utf8
# pylint: disable=too-many-function-args, old-style-class
from NiaPy.tests.test_algorithm import AlgorithmTestCase, MyBenchmark
from NiaPy.algorithms.basic import GreyWolfOptimizer


class GWOTestCase(AlgorithmTestCase):

    def test_custom_works_fine(self):
        gwo_custom = GreyWolfOptimizer(D=self.D, NP=20, nFES=self.nFES, nGEN=self.nGEN, benchmark=MyBenchmark(), seed=self.seed)
        gwo_customc = GreyWolfOptimizer(D=self.D, NP=20, nFES=self.nFES, nGEN=self.nGEN, benchmark=MyBenchmark(), seed=self.seed)
        AlgorithmTestCase.algorithm_run_test(self, gwo_custom, gwo_customc)

    def test_griewank_works_fine(self):
        gwo_griewank = GreyWolfOptimizer(NP=10, D=self.D, nFES=self.nFES, nGEN=self.nGEN, benchmark='griewank', seed=self.seed)
        gwo_griewankc = GreyWolfOptimizer(NP=10, D=self.D, nFES=self.nFES, nGEN=self.nGEN, benchmark='griewank', seed=self.seed)
        AlgorithmTestCase.algorithm_run_test(self, gwo_griewank, gwo_griewankc)
