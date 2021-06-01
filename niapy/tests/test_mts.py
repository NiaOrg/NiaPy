# encoding=utf8
from niapy.algorithms.other import MultipleTrajectorySearch, MultipleTrajectorySearchV1
from niapy.tests.test_algorithm import AlgorithmTestCase, MyProblem


class MTSTestCase(AlgorithmTestCase):
    def setUp(self):
        AlgorithmTestCase.setUp(self)
        self.algo = MultipleTrajectorySearch

    def test_custom(self):
        mts_custom = self.algo(population_size=10, C_a=2, C_r=0.5, seed=self.seed)
        mts_customc = self.algo(population_size=10, C_a=2, C_r=0.5, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, mts_custom, mts_customc, MyProblem(), max_iters=100)

    def test_griewank(self):
        mts_griewank = self.algo(population_size=10, C_a=5, C_r=0.5, seed=self.seed)
        mts_griewankc = self.algo(population_size=10, C_a=5, C_r=0.5, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, mts_griewank, mts_griewankc, max_iters=100)


class MTSv1TestCase(AlgorithmTestCase):
    def setUp(self):
        AlgorithmTestCase.setUp(self)
        self.algo = MultipleTrajectorySearchV1

    def test_custom(self):
        mts_custom = self.algo(population_size=10, C_a=2, C_r=0.5, seed=self.seed)
        mts_customc = self.algo(population_size=10, C_a=2, C_r=0.5, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, mts_custom, mts_customc, MyProblem())

    def test_griewank(self):
        mts_griewank = self.algo(population_size=10, C_a=5, C_r=0.5, seed=self.seed)
        mts_griewankc = self.algo(population_size=10, C_a=5, C_r=0.5, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, mts_griewank, mts_griewankc)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
