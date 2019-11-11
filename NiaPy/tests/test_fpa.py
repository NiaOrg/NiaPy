# encoding=utf8
from NiaPy.tests.test_algorithm import AlgorithmTestCase, MyBenchmark
from NiaPy.algorithms.basic import FlowerPollinationAlgorithm

class FPATestCase(AlgorithmTestCase):
	def setUp(self):
		AlgorithmTestCase.setUp(self)
		self.algo = FlowerPollinationAlgorithm

	def test_type_parameters(self):
		d = self.algo.typeParameters()
		self.assertTrue(d['NP'](10))
		self.assertFalse(d['NP'](-10))
		self.assertFalse(d['NP'](0))
		self.assertTrue(d['beta'](10))
		self.assertFalse(d['beta'](0))
		self.assertFalse(d['beta'](-10))
		self.assertTrue(d['p'](0.5))
		self.assertFalse(d['p'](-0.5))
		self.assertFalse(d['p'](1.5))

	def test_custom_works_fine(self):
		fpa_custom = self.algo(NP=10, p=0.5, seed=self.seed)
		fpa_customc = self.algo(NP=10, p=0.5, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, fpa_custom, fpa_customc, MyBenchmark())

	def test_griewank_works_fine(self):
		fpa_griewank = self.algo(NP=20, p=0.5, seed=self.seed)
		fpa_griewankc = self.algo(NP=20, p=0.5, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, fpa_griewank, fpa_griewankc)

	def test_griewank_works_fine_with_beta(self):
		fpa_beta_griewank = self.algo(NP=20, p=0.5, beta=1.2, seed=self.seed)
		fpa_beta_griewankc = self.algo(NP=20, p=0.5, beta=1.2, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, fpa_beta_griewank, fpa_beta_griewankc)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
