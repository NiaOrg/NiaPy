# encoding=utf8

from niapy.algorithms.basic import HarmonySearch, HarmonySearchV1
from niapy.tests.test_algorithm import AlgorithmTestCase, MyProblem


class HSTestCase(AlgorithmTestCase):
    def setUp(self):
        AlgorithmTestCase.setUp(self)
        self.algo = HarmonySearch

    def test_custom(self):
        hs_costom = self.algo(population_size=10, seed=self.seed)
        hs_costomc = self.algo(population_size=10, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, hs_costom, hs_costomc, MyProblem())

    def test_griewank(self):
        hs_griewank = self.algo(population_size=10, seed=self.seed)
        hs_griewankc = self.algo(population_size=10, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, hs_griewank, hs_griewankc)


class HSv1TestCase(AlgorithmTestCase):
    def setUp(self):
        AlgorithmTestCase.setUp(self)
        self.algo = HarmonySearchV1

    def test_custom(self):
        hs_costom = self.algo(population_size=10, seed=self.seed)
        hs_costomc = self.algo(population_size=10, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, hs_costom, hs_costomc, MyProblem())

    def test_griewank(self):
        hs_griewank = self.algo(population_size=10, seed=self.seed)
        hs_griewankc = self.algo(population_size=10, seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, hs_griewank, hs_griewankc)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
