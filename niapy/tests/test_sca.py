# encoding=utf8
from niapy.algorithms.basic import SineCosineAlgorithm
from niapy.tests.test_algorithm import AlgorithmTestCase, MyProblem


class SCATestCase(AlgorithmTestCase):
    def setUp(self):
        AlgorithmTestCase.setUp(self)
        self.algo = SineCosineAlgorithm

    def test_algorithm_info(self):
        self.assertIsNotNone(self.algo.info())

    def test_custom(self):
        sca_custom = self.algo(population_size=35, a=7, r_min=0.1, r_max=3, seed=self.seed)
        sca_customc = self.algo(population_size=35, a=7, r_min=0.1, r_max=3, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, sca_custom, sca_customc, MyProblem())

    def test_griewank(self):
        sca_griewank = self.algo(population_size=10, a=5, r_min=0.01, r_max=3, seed=self.seed)
        sca_griewankc = self.algo(population_size=10, a=5, r_min=0.01, r_max=3, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, sca_griewank, sca_griewankc)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
