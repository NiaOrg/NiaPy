# encoding=utf8
from niapy.algorithms.basic import FlowerPollinationAlgorithm
from niapy.tests.test_algorithm import AlgorithmTestCase, MyProblem


class FPATestCase(AlgorithmTestCase):
    def setUp(self):
        AlgorithmTestCase.setUp(self)
        self.algo = FlowerPollinationAlgorithm

    def test_custom(self):
        fpa_custom = self.algo(population_size=10, p=0.5, seed=self.seed)
        fpa_customc = self.algo(population_size=10, p=0.5, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, fpa_custom, fpa_customc, MyProblem())

    def test_griewank(self):
        fpa_griewank = self.algo(population_size=10, p=0.5, seed=self.seed)
        fpa_griewankc = self.algo(population_size=10, p=0.5, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, fpa_griewank, fpa_griewankc)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
