# encoding=utf8

from NiaPy.algorithms.other import MultipleTrajectorySearch, MultipleTrajectorySearchV1
from NiaPy.tests.test_algorithm import AlgorithmTestCase, MyBenchmark

class MTSTestCase(AlgorithmTestCase):
	def test_type_parameters(self):
		d = MultipleTrajectorySearch.typeParameters()
		self.assertTrue(d['NoLsTests'](10))
		self.assertTrue(d['NoLsTests'](0))
		self.assertFalse(d['NoLsTests'](-10))
		self.assertTrue(d['NoLs'](10))
		self.assertTrue(d['NoLs'](0))
		self.assertFalse(d['NoLs'](-10))
		self.assertTrue(d['NoLsBest'](10))
		self.assertTrue(d['NoLsBest'](0))
		self.assertFalse(d['NoLsBest'](-10))
		self.assertTrue(d['NoEnabled'](10))
		self.assertFalse(d['NoEnabled'](0))
		self.assertFalse(d['NoEnabled'](-10))

	def test_custom_works_fine(self):
		mts_custom = MultipleTrajectorySearch(n=10, C_a=2, C_r=0.5, seed=self.seed)
		mts_customc = MultipleTrajectorySearch(n=10, C_a=2, C_r=0.5, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, mts_custom, mts_customc, MyBenchmark())

	def test_griewank_works_fine(self):
		mts_griewank = MultipleTrajectorySearch(n=10, C_a=5, C_r=0.5, seed=self.seed)
		mts_griewankc = MultipleTrajectorySearch(n=10, C_a=5, C_r=0.5, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, mts_griewank, mts_griewankc)

class MTSv1TestCase(AlgorithmTestCase):
	def test_custom_works_fine(self):
		mts_custom = MultipleTrajectorySearchV1(n=10, C_a=2, C_r=0.5, seed=self.seed)
		mts_customc = MultipleTrajectorySearchV1(n=10, C_a=2, C_r=0.5, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, mts_custom, mts_customc, MyBenchmark())

	def test_griewank_works_fine(self):
		mts_griewank = MultipleTrajectorySearchV1(n=10, C_a=5, C_r=0.5, seed=self.seed)
		mts_griewankc = MultipleTrajectorySearchV1(n=10, C_a=5, C_r=0.5, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, mts_griewank, mts_griewankc)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
