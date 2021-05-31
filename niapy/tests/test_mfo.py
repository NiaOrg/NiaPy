# encoding=utf8
from niapy.algorithms.basic import MothFlameOptimizer
from niapy.tests.test_algorithm import AlgorithmTestCase, MyProblem


class MFOTestCase(AlgorithmTestCase):
    def setUp(self):
        AlgorithmTestCase.setUp(self)
        self.algo = MothFlameOptimizer

    def test_custom(self):
        mfo_custom = self.algo(population_size=10, seed=self.seed)
        mfo_customc = self.algo(population_size=10, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, mfo_custom, mfo_customc, MyProblem())

    def test_griewank(self):
        mfo_griewank = self.algo(population_size=10, seed=self.seed)
        mfo_griewankc = self.algo(population_size=10, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, mfo_griewank, mfo_griewankc)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
