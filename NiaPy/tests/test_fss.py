# encoding=utf8
# pylint: disable=too-many-function-args, old-style-class
from NiaPy.tests.test_algorithm import AlgorithmTestCase, MyBenchmark
from NiaPy.algorithms.basic import FishSchoolSearch


class FSSTestCase(AlgorithmTestCase):

    def test_custom_works_fine(self):
        fss_custom = FishSchoolSearch(D=self.D, NP=20, nFES=self.nFES, nGEN=self.nGEN, benchmark=MyBenchmark(), seed=self.seed)
        fss_customc = FishSchoolSearch(D=self.D, NP=20, nFES=self.nFES, nGEN=self.nGEN, benchmark=MyBenchmark(), seed=self.seed)
        AlgorithmTestCase.algorithm_run_test(self, fss_custom, fss_customc)

    def test_griewank_works_fine(self):
        fss_custom = FishSchoolSearch(NP=10, D=self.D, nFES=self.nFES, nGEN=self.nGEN, benchmark='griewank', seed=self.seed)
        fss_customc = FishSchoolSearch(NP=10, D=self.D, nFES=self.nFES, nGEN=self.nGEN, benchmark='griewank', seed=self.seed)
        AlgorithmTestCase.algorithm_run_test(self, fss_custom, fss_customc)
