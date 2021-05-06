# encoding=utf8
from niapy.algorithms.other import NelderMeadMethod
from niapy.tests.test_algorithm import AlgorithmTestCase, MyBenchmark


class NMMTestCase(AlgorithmTestCase):
    def setUp(self):
        AlgorithmTestCase.setUp(self)
        self.algo = NelderMeadMethod

    def test_type_parameters(self):
        d = NelderMeadMethod.type_parameters()
        self.assertIsNotNone(d.get('population_size', None))
        self.assertIsNotNone(d.get('alpha', None))
        self.assertIsNotNone(d.get('gamma', None))
        self.assertIsNotNone(d.get('rho', None))
        self.assertIsNotNone(d.get('sigma', None))

    def test_custom(self):
        nmm_custom = self.algo(n=10, C_a=2, C_r=0.5, seed=self.seed)
        nmm_customc = self.algo(n=10, C_a=2, C_r=0.5, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, nmm_custom, nmm_customc, MyBenchmark())

    def test_griewank(self):
        nmm_griewank = self.algo(n=10, C_a=5, C_r=0.5, seed=self.seed)
        nmm_griewankc = self.algo(n=10, C_a=5, C_r=0.5, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, nmm_griewank, nmm_griewankc)

    def test_michalewichz(self):
        nmm_griewank = self.algo(n=10, C_a=5, C_r=0.5, seed=self.seed)
        nmm_griewankc = self.algo(n=10, C_a=5, C_r=0.5, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, nmm_griewank, nmm_griewankc, 'michalewicz', max_iters=10000000)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
