# encoding=utf8
# pylint: disable=too-many-function-args
from NiaPy.tests.test_algorithm import AlgorithmTestCase, MyBenchmark
from NiaPy.algorithms.basic import FishSchoolSearch


class FSSTestCase(AlgorithmTestCase):

    def test_custom_works_fine(self):
        fss_custom = FishSchoolSearch(NP=20, seed=self.seed)
        fss_customc = FishSchoolSearch(NP=20, seed=self.seed)
        AlgorithmTestCase.algorithm_run_test(self, fss_custom, fss_customc, MyBenchmark())

    def test_griewank_works_fine(self):
        fss_custom = FishSchoolSearch(NP=10, seed=self.seed)
        fss_customc = FishSchoolSearch(NP=10, seed=self.seed)
        AlgorithmTestCase.algorithm_run_test(self, fss_custom, fss_customc)
