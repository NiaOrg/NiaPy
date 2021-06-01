# encoding=utf8

from niapy.algorithms.basic import CoralReefsOptimization
from niapy.tests.test_algorithm import AlgorithmTestCase, MyProblem


class CROTestCase(AlgorithmTestCase):
    def setUp(self):
        AlgorithmTestCase.setUp(self)
        self.algo = CoralReefsOptimization

    def test_custom(self):
        cro_custom = self.algo(population_size=10, seed=self.seed)
        cro_customc = self.algo(population_size=10, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, cro_custom, cro_customc, MyProblem(), max_iters=100)

    def test_griewank(self):
        cro_griewank = self.algo(population_size=10, seed=self.seed)
        cro_griewankc = self.algo(population_size=10, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, cro_griewank, cro_griewankc, max_iters=100)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
