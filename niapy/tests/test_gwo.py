# encoding=utf8
from niapy.algorithms.basic import GreyWolfOptimizer
from niapy.tests.test_algorithm import AlgorithmTestCase, MyProblem


class GWOTestCase(AlgorithmTestCase):
    def setUp(self):
        AlgorithmTestCase.setUp(self)
        self.algo = GreyWolfOptimizer

    def test_custom(self):
        gwo_custom = self.algo(population_size=10, seed=self.seed)
        gwo_customc = self.algo(population_size=10, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, gwo_custom, gwo_customc, MyProblem())

    def test_griewank(self):
        gwo_griewank = self.algo(population_size=10, seed=self.seed)
        gwo_griewankc = self.algo(population_size=10, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, gwo_griewank, gwo_griewankc)
