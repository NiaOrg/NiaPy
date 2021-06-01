# encoding=utf8

from niapy.algorithms.basic import HarrisHawksOptimization
from niapy.tests.test_algorithm import AlgorithmTestCase, MyProblem


class HHOTestCase(AlgorithmTestCase):
    def setUp(self):
        AlgorithmTestCase.setUp(self)
        self.algo = HarrisHawksOptimization

    def test_custom(self):
        hho_custom = self.algo(population_size=10, levy=0.01, seed=self.seed)
        hho_customc = self.algo(population_size=10, levy=0.01, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, hho_custom, hho_customc, MyProblem())

    def test_griewank(self):
        hho_griewank = self.algo(population_size=10, nFES=4000, nGEN=200, levy=0.01, seed=self.seed)
        hho_griewankc = self.algo(population_size=10, nFES=4000, nGEN=200, levy=0.01, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, hho_griewank, hho_griewankc)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
