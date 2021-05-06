from niapy.algorithms.basic import FireflyAlgorithm
from niapy.tests.test_algorithm import AlgorithmTestCase, MyBenchmark


class FATestCase(AlgorithmTestCase):
    def setUp(self):
        AlgorithmTestCase.setUp(self)
        self.algo = FireflyAlgorithm

    def test_type_parameters(self):
        d = self.algo.type_parameters()
        self.assertTrue(d['alpha'](10))
        self.assertFalse(d['alpha'](-10))
        self.assertTrue(d['beta_min'](10))
        self.assertFalse(d['beta_min'](-10))
        self.assertTrue(d['gamma'](10))
        self.assertFalse(d['gamma'](-10))
        self.assertTrue(d['population_size'](1))
        self.assertFalse(d['population_size'](0))
        self.assertFalse(d['population_size'](-1))

    def test(self):
        fa = self.algo(population_size=20, alpha=0.5, beta_min=0.2, gamma=1.0, seed=self.seed)
        fac = self.algo(population_size=20, alpha=0.5, beta_min=0.2, gamma=1.0, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, fa, fac, MyBenchmark())

    def test_griewank(self):
        fa_griewank = self.algo(population_size=20, alpha=0.5, beta_min=0.2, gamma=1.0, seed=self.seed)
        fa_griewankc = self.algo(population_size=20, alpha=0.5, beta_min=0.2, gamma=1.0, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, fa_griewank, fa_griewankc)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
