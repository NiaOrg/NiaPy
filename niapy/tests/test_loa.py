# encoding=utf8
from niapy.algorithms.basic import LionOptimizationAlgorithm
from niapy.tests.test_algorithm import AlgorithmTestCase, MyProblem


class LOATestCase(AlgorithmTestCase):
    def setUp(self):
        AlgorithmTestCase.setUp(self)
        self.algo = LionOptimizationAlgorithm

    def test_custom(self):
        loa_custom = self.algo(population_size=10, seed=self.seed)
        loa_customc = self.algo(population_size=10, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, loa_custom, loa_customc, MyProblem())

    def test_griewank(self):
        loa_griewank = self.algo(population_size=10, seed=self.seed)
        loa_griewankc = self.algo(population_size=10, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, loa_griewank, loa_griewankc)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
