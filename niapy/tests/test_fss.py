# encoding=utf8
from niapy.algorithms.basic import FishSchoolSearch
from niapy.tests.test_algorithm import AlgorithmTestCase, MyProblem


class FSSTestCase(AlgorithmTestCase):
    def setUp(self):
        AlgorithmTestCase.setUp(self)
        self.algo = FishSchoolSearch

    def test_custom(self):
        fss_custom = self.algo(population_size=10, seed=self.seed)
        fss_customc = self.algo(population_size=10, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, fss_custom, fss_customc, MyProblem())

    def test_griewank(self):
        fss_custom = self.algo(population_size=10, seed=self.seed)
        fss_customc = self.algo(population_size=10, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, fss_custom, fss_customc)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
