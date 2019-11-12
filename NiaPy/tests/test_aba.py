# pylint: disable=line-too-long
from NiaPy.tests.test_algorithm import AlgorithmTestCase, MyBenchmark
from NiaPy.algorithms.modified import AdaptiveBatAlgorithm

class ABATestCase(AlgorithmTestCase):
	r"""Test case for AdaptiveBatAlgorithm algorithm.

	Date:
		April 2019

	Author:
		Klemen Berkovic

	See Also:
		* :class:`NiaPy.algorithms.modified.AdaptiveBatAlgorithm`
	"""
	def test_algorithm_info(self):
		"""Test algorithm info method of class AdaptiveBatAlgorithm."""
		self.assertIsNotNone(AdaptiveBatAlgorithm.algorithmInfo())

	def test_type_parameters(self):
		"""Test type parameters method of class AdaptiveBatAlgorithm."""
		d = AdaptiveBatAlgorithm.typeParameters()
		# Test epsilon parameter check
		self.assertIsNotNone(d.get('epsilon', None))
		self.assertFalse(d['epsilon'](-100))
		self.assertFalse(d['epsilon'](-.3))
		self.assertTrue(d['epsilon'](3))
		self.assertTrue(d['epsilon'](.3))
		self.assertTrue(d['epsilon'](300))
		# Test alpha parameter check
		self.assertIsNotNone(d.get('alpha', None))
		self.assertFalse(d['alpha'](-100))
		self.assertFalse(d['alpha'](-.3))
		self.assertTrue(d['alpha'](3))
		self.assertTrue(d['alpha'](.3))
		self.assertTrue(d['alpha'](300))
		# Test r parameter check
		self.assertIsNotNone(d.get('r', None))
		self.assertFalse(d['r'](-100))
		self.assertFalse(d['r'](-.3))
		self.assertTrue(d['r'](3))
		self.assertTrue(d['r'](.3))
		self.assertTrue(d['r'](300))
		# Test Qmin parameter check
		self.assertIsNotNone(d.get('Qmin', None))
		self.assertTrue(d['Qmin'](3))
		# Test Qmax parameter check
		self.assertIsNotNone(d.get('Qmax', None))
		self.assertTrue(d['Qmax'](300))

	def test_custom_works_fine(self):
		aba_custom = AdaptiveBatAlgorithm(NP=40, A=.75, epsilon=2, alpha=0.65, r=0.7, Qmin=0.0, Qmax=2.0, seed=self.seed)
		aba_customc = AdaptiveBatAlgorithm(NP=40, A=.75, epsilon=2, alpha=0.65, r=0.7, Qmin=0.0, Qmax=2.0, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, aba_custom, aba_customc, MyBenchmark())

	def test_griewank_works_fine(self):
		aba_griewank = AdaptiveBatAlgorithm(NP=40, A=0.5, r=0.5, F=0.5, CR=0.9, Qmin=0.0, Qmax=2.0, seed=self.seed)
		aba_griewankc = AdaptiveBatAlgorithm(NP=40, A=0.5, r=0.5, F=0.5, CR=0.9, Qmin=0.0, Qmax=2.0, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, aba_griewank, aba_griewankc)
