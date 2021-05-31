# encoding=utf8
from niapy.algorithms.basic import CamelAlgorithm
from niapy.tests.test_algorithm import AlgorithmTestCase, MyProblem


class CATestCase(AlgorithmTestCase):
    def setUp(self):
        AlgorithmTestCase.setUp(self)
        self.algo = CamelAlgorithm

    def test_custom(self):
        ca_custom = self.algo(population_size=10, seed=self.seed)
        ca_customc = self.algo(population_size=10, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, ca_custom, ca_customc, MyProblem())

    def test_griewank(self):
        ca_griewank = self.algo(population_size=10, seed=self.seed)
        ca_griewankc = self.algo(population_size=10, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, ca_griewank, ca_griewankc)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
