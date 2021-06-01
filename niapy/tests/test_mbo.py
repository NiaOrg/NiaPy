from niapy.algorithms.basic import MonarchButterflyOptimization
from niapy.tests.test_algorithm import AlgorithmTestCase, MyProblem


class MBOTestCase(AlgorithmTestCase):
    def setUp(self):
        AlgorithmTestCase.setUp(self)
        self.algo = MonarchButterflyOptimization

    def test(self):
        mbo = self.algo(population_size=10, PAR=5.0 / 12.0, PER=1.2, seed=self.seed)
        mboc = self.algo(population_size=10, PAR=5.0 / 12.0, PER=1.2, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, mbo, mboc, MyProblem())

    def test_griewank(self):
        mbo_griewank = self.algo(population_size=10, PAR=5.0 / 12.0, PER=1.2, seed=self.seed)
        mbo_griewankc = self.algo(population_size=10, PAR=5.0 / 12.0, PER=1.2, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, mbo_griewank, mbo_griewankc)
