# encoding=utf8
from niapy.algorithms.other import NelderMeadMethod
from niapy.tests.test_algorithm import AlgorithmTestCase, MyProblem


class NMMTestCase(AlgorithmTestCase):
    def setUp(self):
        AlgorithmTestCase.setUp(self)
        self.algo = NelderMeadMethod

    def test_custom(self):
        nmm_custom = self.algo(population_size=10, C_a=2, C_r=0.5, seed=self.seed)
        nmm_customc = self.algo(population_size=10, C_a=2, C_r=0.5, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, nmm_custom, nmm_customc, MyProblem())

    def test_griewank(self):
        nmm_griewank = self.algo(population_size=10, C_a=5, C_r=0.5, seed=self.seed)
        nmm_griewankc = self.algo(population_size=10, C_a=5, C_r=0.5, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, nmm_griewank, nmm_griewankc)

    def test_michalewichz(self):
        nmm_griewank = self.algo(population_size=10, C_a=5, C_r=0.5, seed=self.seed)
        nmm_griewankc = self.algo(population_size=10, C_a=5, C_r=0.5, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, nmm_griewank, nmm_griewankc, 'michalewicz', max_iters=10000000)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
