# encoding=utf8
# pylint: disable=too-many-function-args
from NiaPy.tests.test_algorithm import AlgorithmTestCase, MyBenchmark
from NiaPy.algorithms.basic import MothFlameOptimizer

class MFOTestCase(AlgorithmTestCase):

    def test_custom_works_fine(self):
        mfo_custom = MothFlameOptimizer(NP=20, seed=self.seed)
        mfo_customc = MothFlameOptimizer(NP=20, seed=self.seed)
        AlgorithmTestCase.algorithm_run_test(self, mfo_custom, mfo_customc, MyBenchmark())

    def test_griewank_works_fine(self):
        mfo_griewank = MothFlameOptimizer(NP=10, seed=self.seed)
        mfo_griewankc = MothFlameOptimizer(NP=10, seed=self.seed)
        AlgorithmTestCase.algorithm_run_test(self, mfo_griewank, mfo_griewankc)
