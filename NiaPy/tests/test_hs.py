# encoding=utf8
# pylint: disable=mixed-indentation, multiple-statements
from NiaPy.tests.test_algorithm import AlgorithmTestCase, MyBenchmark
from NiaPy.algorithms.basic import HarmonySearch, HarmonySearchV1

class HSTestCase(AlgorithmTestCase):
	def test_type_parameters(self):
		d = HarmonySearch.typeParameters()
		self.assertIsNotNone(d.get('HMS', None))
		self.assertTrue(d['HMS'](10))
		self.assertFalse(d['HMS'](-10))
		self.assertFalse(d['HMS'](-10.3))
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

	def test_custom_works_fine(self):
		hs_costom = HarmonySearch(seed=self.seed)
		hs_costomc = HarmonySearch(seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, hs_costom, hs_costomc, MyBenchmark())

	def test_griewank_works_fine(self):
		hs_griewank = HarmonySearch(seed=self.seed)
		hs_griewankc = HarmonySearch(seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, hs_griewank, hs_griewankc)

class HSv1TestCase(AlgorithmTestCase):
	def test_type_parameters(self):
		d = HarmonySearchV1.typeParameters()
		self.assertIsNone(d.get('b_range', None))
		self.assertIsNotNone(d.get('dw_min', None))
		self.assertIsNotNone(d.get('dw_max', None))
		self.assertTrue(d['dw_min'](10))
		self.assertFalse(d['dw_min'](-10))
		self.assertTrue(d['dw_max'](10))
		self.assertFalse(d['dw_max'](-10))

	def test_custom_works_fine(self):
		hs_costom = HarmonySearchV1(seed=self.seed)
		hs_costomc = HarmonySearchV1(seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, hs_costom, hs_costomc, MyBenchmark())

	def test_griewank_works_fine(self):
		hs_griewank = HarmonySearchV1(seed=self.seed)
		hs_griewankc = HarmonySearchV1(seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, hs_griewank, hs_griewankc)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
