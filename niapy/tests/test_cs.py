# encoding=utf8
from niapy.algorithms.basic import CuckooSearch
from niapy.tests.test_algorithm import AlgorithmTestCase, MyProblem


class CSTestCase(AlgorithmTestCase):
    def setUp(self):
        AlgorithmTestCase.setUp(self)
        self.algo = CuckooSearch

    def test_custom(self):
        cs_custom = self.algo(population_size=10, seed=self.seed)
        cs_customc = self.algo(population_size=10, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, cs_custom, cs_customc, MyProblem())

    def test_griewank(self):
        cs_griewank = self.algo(population_size=10, seed=self.seed)
        cs_griewankc = self.algo(population_size=10, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, cs_griewank, cs_griewankc)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
