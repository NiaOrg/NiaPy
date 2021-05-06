# encoding=utf8

from niapy.algorithms.basic import HarmonySearch, HarmonySearchV1
from niapy.tests.test_algorithm import AlgorithmTestCase, MyBenchmark


class HSTestCase(AlgorithmTestCase):
    def setUp(self):
        AlgorithmTestCase.setUp(self)
        self.algo = HarmonySearch

    def test_custom(self):
        hs_costom = self.algo(seed=self.seed)
        hs_costomc = self.algo(seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, hs_costom, hs_costomc, MyBenchmark())

    def test_griewank(self):
        hs_griewank = self.algo(seed=self.seed)
        hs_griewankc = self.algo(seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, hs_griewank, hs_griewankc)


class HSv1TestCase(AlgorithmTestCase):
    def setUp(self):
        AlgorithmTestCase.setUp(self)
        self.algo = HarmonySearchV1

    def test_custom(self):
        hs_costom = self.algo(seed=self.seed)
        hs_costomc = self.algo(seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, hs_costom, hs_costomc, MyBenchmark())

    def test_griewank(self):
        hs_griewank = self.algo(seed=self.seed)
        hs_griewankc = self.algo(seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, hs_griewank, hs_griewankc)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
