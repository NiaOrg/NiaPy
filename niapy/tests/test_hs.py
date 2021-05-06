# encoding=utf8

from niapy.algorithms.basic import HarmonySearch, HarmonySearchV1
from niapy.tests.test_algorithm import AlgorithmTestCase, MyBenchmark


class HSTestCase(AlgorithmTestCase):
    def setUp(self):
        AlgorithmTestCase.setUp(self)
        self.algo = HarmonySearch

    def test_type_parameters(self):
        d = self.algo.type_parameters()
        self.assertIsNotNone(d.get('r_accept', None))
        self.assertTrue(d['r_accept'](.3))
        self.assertTrue(d['r_accept'](0.99))
        self.assertFalse(d['r_accept'](-0.99))
        self.assertFalse(d['r_accept'](9))
        self.assertIsNotNone(d.get('r_pa', None))
        self.assertTrue(d['r_pa'](.3))
        self.assertTrue(d['r_pa'](0.99))
        self.assertFalse(d['r_pa'](-0.99))
        self.assertFalse(d['r_pa'](9))
        self.assertIsNotNone(d.get('b_range', None))
        self.assertTrue(d['b_range'](10))
        self.assertFalse(d['b_range'](-10))
        self.assertFalse(d['b_range'](-10.3))

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

    def test_type_parameters(self):
        d = self.algo.type_parameters()
        self.assertIsNone(d.get('b_range', None))
        self.assertIsNotNone(d.get('bw_min', None))
        self.assertIsNotNone(d.get('bw_max', None))
        self.assertTrue(d['bw_min'](10))
        self.assertFalse(d['bw_min'](-10))
        self.assertTrue(d['bw_max'](10))
        self.assertFalse(d['bw_max'](-10))

    def test_custom(self):
        hs_costom = self.algo(seed=self.seed)
        hs_costomc = self.algo(seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, hs_costom, hs_costomc, MyBenchmark())

    def test_griewank(self):
        hs_griewank = self.algo(seed=self.seed)
        hs_griewankc = self.algo(seed=self.seed)
        AlgorithmTestCase.test_algorithm_run(self, hs_griewank, hs_griewankc)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
