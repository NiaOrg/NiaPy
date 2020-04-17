# encoding=utf8

from NiaPy.algorithms.basic import FishSchoolSearch
from NiaPy.tests.test_algorithm import (
    AlgorithmTestCase,
    MyBenchmark
)


class FSSTestCase(AlgorithmTestCase):
    def setUp(self):
        AlgorithmTestCase.setUp(self)
        self.algo = FishSchoolSearch

    def test_custom_works_fine(self):
        fss_custom = self.algo(NP=20, seed=self.seed)
        fss_customc = self.algo(NP=20, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, fss_custom, fss_customc, MyBenchmark())

    def test_griewank_works_fine(self):
        fss_custom = self.algo(NP=10, seed=self.seed)
        fss_customc = self.algo(NP=10, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, fss_custom, fss_customc)
