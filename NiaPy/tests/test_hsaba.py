# encoding=utf8
from NiaPy.tests.test_algorithm import AlgorithmTestCase, MyBenchmark
from NiaPy.algorithms.modified import HybridSelfAdaptiveBatAlgorithm

class HSABATestCase(AlgorithmTestCase):
	r"""Test case for HybridSelfAdaptiveBatAlgorithm algorithm.

	Date:
		April 2019

	Author:
		Klemen Berkovič

	See Also:
		* :class:`NiaPy.algorithms.modified.HybridSelfAdaptiveBatAlgorithm`
	"""
	def setUp(self):
		AlgorithmTestCase.setUp(self)
		self.algo = HybridSelfAdaptiveBatAlgorithm

	def test_algorithm_info_fine(self):
		"""Test case for algorithm info."""
		i = self.algo.algorithmInfo()
		self.assertIsNotNone(i)

	def test_type_parameters_fine(self):
		"""Test case for type parameters."""
		d = self.algo.typeParameters()
		# Test F parameter check
		self.assertIsNotNone(d.get('F', None))
		self.assertFalse(d['F'](-30))
		self.assertFalse(d['F'](-.3))
		self.assertTrue(d['F'](.3))
		self.assertTrue(d['F'](.39))
		# Test CR parameter check
		self.assertIsNotNone(d.get('CR', None))
		self.assertFalse(d['CR'](10))
		self.assertFalse(d['CR'](-10))
		self.assertFalse(d['CR'](-1))
		self.assertTrue(d['CR'](.3))
		self.assertTrue(d['CR'](.0))
		self.assertTrue(d['CR'](1.))

	def test_custom_works_fine(self):
		"""Test case for running algorithm on costume benchmarks."""
		hsaba_custom = self.algo(NP=10, Limit=2, seed=self.seed)
		hsaba_customc = self.algo(NP=10, Limit=2, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, hsaba_custom, hsaba_customc, MyBenchmark())

	def test_griewank_works_fine(self):
		"""Test case for running algorithm on benchmark."""
		hsaba_griewank = self.algo(NP=10, seed=self.seed)
		hsaba_griewankc = self.algo(NP=10, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, hsaba_griewank, hsaba_griewankc)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
