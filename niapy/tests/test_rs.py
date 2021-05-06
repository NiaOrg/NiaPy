# encoding=utf8

from niapy.algorithms.other import RandomSearch
from niapy.tests.test_algorithm import AlgorithmTestCase, MyBenchmark


class RSTestCase(AlgorithmTestCase):
    def setUp(self):
        AlgorithmTestCase.setUp(self)
        self.algo = RandomSearch

    def test_type_parameters(self):
        d = self.algo.type_parameters()
        self.assertTrue(d['population_size'](10))
        self.assertFalse(d['population_size'](0))
        self.assertFalse(d['population_size'](-10))

    def test_custom(self):
        ca_custom = self.algo(NP=40, seed=self.seed)
        ca_customc = self.algo(NP=40, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, ca_custom, ca_customc, MyBenchmark())

    def test_griewank(self):
        ca_griewank = self.algo(NP=40, seed=self.seed)
        ca_griewankc = self.algo(NP=40, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, ca_griewank, ca_griewankc)
