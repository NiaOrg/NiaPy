from niapy.algorithms.basic import FireflyAlgorithm
from niapy.tests.test_algorithm import AlgorithmTestCase, MyProblem


class FATestCase(AlgorithmTestCase):
    def setUp(self):
        AlgorithmTestCase.setUp(self)
        self.algo = FireflyAlgorithm

    def test(self):
        fa = self.algo(population_size=10, alpha=0.5, beta_min=0.2, gamma=1.0, seed=self.seed)
        fac = self.algo(population_size=10, alpha=0.5, beta_min=0.2, gamma=1.0, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, fa, fac, MyProblem())

    def test_griewank(self):
        fa_griewank = self.algo(population_size=10, alpha=0.5, beta_min=0.2, gamma=1.0, seed=self.seed)
        fa_griewankc = self.algo(population_size=10, alpha=0.5, beta_min=0.2, gamma=1.0, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, fa_griewank, fa_griewankc)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
