# encoding=utf8

from niapy.algorithms.other import RandomSearch
from niapy.tests.test_algorithm import AlgorithmTestCase, MyProblem


class RSTestCase(AlgorithmTestCase):
    def setUp(self):
        AlgorithmTestCase.setUp(self)
        self.algo = RandomSearch

    def test_custom(self):
        ca_custom = self.algo(population_size=10, seed=self.seed)
        ca_customc = self.algo(population_size=10, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, ca_custom, ca_customc, MyProblem(), max_iters=100, max_evals=50000)

    def test_griewank(self):
        ca_griewank = self.algo(population_size=10, seed=self.seed)
        ca_griewankc = self.algo(population_size=10, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, ca_griewank, ca_griewankc, max_iters=100, max_evals=50000)
