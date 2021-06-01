# encoding=utf8

from niapy.algorithms.basic import CatSwarmOptimization
from niapy.tests.test_algorithm import AlgorithmTestCase, MyProblem


class CSOTestCase(AlgorithmTestCase):
    def setUp(self):
        AlgorithmTestCase.setUp(self)
        self.algo = CatSwarmOptimization

    def test_custom(self):
        cso_custom = self.algo(population_size=10, seed=self.seed)
        cso_customc = self.algo(population_size=10, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, cso_custom, cso_customc, MyProblem())

    def test_griewank(self):
        cso_griewank = self.algo(population_size=10, seed=self.seed)
        cso_griewankc = self.algo(population_size=10, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, cso_griewank, cso_griewankc)
