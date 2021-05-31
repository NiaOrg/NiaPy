# encoding=utf8
from niapy.algorithms.basic import GravitationalSearchAlgorithm
from niapy.tests.test_algorithm import AlgorithmTestCase, MyProblem


class GSATestCase(AlgorithmTestCase):
    def setUp(self):
        AlgorithmTestCase.setUp(self)
        self.algo = GravitationalSearchAlgorithm

    def test_Custom(self):
        gsa_custom = self.algo(population_size=10, seed=self.seed)
        gsa_customc = self.algo(population_size=10, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, gsa_custom, gsa_customc, MyProblem())

    def test_griewank(self):
        gsa_griewank = self.algo(population_size=10, seed=self.seed)
        gsa_griewankc = self.algo(population_size=10, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, gsa_griewank, gsa_griewankc)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
