# encoding=utf8

from NiaPy.algorithms.basic import CoralReefsOptimization
from NiaPy.tests.test_algorithm import (
    AlgorithmTestCase,
    MyBenchmark
)


class CROTestCase(AlgorithmTestCase):
    def setUp(self):
        AlgorithmTestCase.setUp(self)
        self.algo = CoralReefsOptimization

    def test_custom_works_fine(self):
        cro_custom = self.algo(N=20, seed=self.seed)
        cro_customc = self.algo(N=20, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, cro_custom, cro_customc, MyBenchmark())

    def test_griewank_works_fine(self):
        cro_griewank = self.algo(N=10, seed=self.seed)
        cro_griewankc = self.algo(N=10, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, cro_griewank, cro_griewankc)
