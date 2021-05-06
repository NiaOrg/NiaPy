# encoding=utf8
from niapy.algorithms.basic import FlowerPollinationAlgorithm
from niapy.tests.test_algorithm import AlgorithmTestCase, MyBenchmark


class FPATestCase(AlgorithmTestCase):
    def setUp(self):
        AlgorithmTestCase.setUp(self)
        self.algo = FlowerPollinationAlgorithm

    def test_type_parameters(self):
        d = self.algo.type_parameters()
        self.assertTrue(d['population_size'](10))
        self.assertFalse(d['population_size'](-10))
        self.assertFalse(d['population_size'](0))
        self.assertTrue(d['beta'](10))
        self.assertFalse(d['beta'](0))
        self.assertFalse(d['beta'](-10))
        self.assertTrue(d['p'](0.5))
        self.assertFalse(d['p'](-0.5))
        self.assertFalse(d['p'](1.5))

    def test_custom(self):
        fpa_custom = self.algo(population_size=10, p=0.5, seed=self.seed)
        fpa_customc = self.algo(population_size=10, p=0.5, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, fpa_custom, fpa_customc, MyBenchmark())

    def test_griewank(self):
        fpa_griewank = self.algo(population_size=20, p=0.5, seed=self.seed)
        fpa_griewankc = self.algo(population_size=20, p=0.5, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, fpa_griewank, fpa_griewankc)

    def test_griewank_with_beta(self):
        fpa_beta_griewank = self.algo(population_size=20, p=0.5, beta=1.2, seed=self.seed)
        fpa_beta_griewankc = self.algo(population_size=20, p=0.5, beta=1.2, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, fpa_beta_griewank, fpa_beta_griewankc)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
