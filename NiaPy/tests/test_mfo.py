# encoding=utf8
# pylint: disable=too-many-function-args, old-style-class
from NiaPy.tests.test_algorithm import AlgorithmTestCase, MyBenchmark
from NiaPy.algorithms.basic import MothFlameOptimizer


class MFOTestCase(AlgorithmTestCase):

    def test_custom_works_fine(self):
        mfo_custom = MothFlameOptimizer(D=self.D, NP=20, nFES=self.nFES, nGEN=self.nGEN, benchmark=MyBenchmark(), seed=self.seed)
        mfo_customc = MothFlameOptimizer(D=self.D, NP=20, nFES=self.nFES, nGEN=self.nGEN, benchmark=MyBenchmark(), seed=self.seed)
        AlgorithmTestCase.algorithm_run_test(self, mfo_custom, mfo_customc)

    def test_griewank_works_fine(self):
        mfo_griewank = MothFlameOptimizer(NP=10, D=self.D, nFES=self.nFES, nGEN=self.nGEN, benchmark='griewank', seed=self.seed)
        mfo_griewankc = MothFlameOptimizer(NP=10, D=self.D, nFES=self.nFES, nGEN=self.nGEN, benchmark='griewank', seed=self.seed)
        AlgorithmTestCase.algorithm_run_test(self, mfo_griewank, mfo_griewankc)
