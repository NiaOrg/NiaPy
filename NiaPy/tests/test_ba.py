# encoding=utf8

from NiaPy.tests.test_algorithm import AlgorithmTestCase, MyBenchmark
from NiaPy.algorithms.basic import BatAlgorithm

class BATestCase(AlgorithmTestCase):
	def setUp(self):
		AlgorithmTestCase.setUp(self)
		self.algo = BatAlgorithm

	def test_parameter_type(self):
		d = self.algo.typeParameters()
		self.assertTrue(d['Qmax'](10))
		self.assertTrue(d['Qmin'](10))
		self.assertTrue(d['r'](10))
		self.assertFalse(d['r'](-10))
		self.assertFalse(d['r'](0))
		self.assertFalse(d['A'](0))
		self.assertFalse(d['A'](-19))
		self.assertTrue(d['NP'](10))
		self.assertFalse(d['NP'](-10))
		self.assertFalse(d['NP'](0))
		self.assertTrue(d['A'](10))
		self.assertFalse(d['Qmin'](None))
		self.assertFalse(d['Qmax'](None))

	def test_custom_works_fine(self):
		ba_custom = self.algo(NP=20, A=0.5, r=0.5, Qmin=0.0, Qmax=2.0, seed=self.seed)
		ba_customc = self.algo(NP=20, A=0.5, r=0.5, Qmin=0.0, Qmax=2.0, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, ba_custom, ba_customc, MyBenchmark())

	def test_griewank_works_fine(self):
		ba_griewank = self.algo(NP=10, A=0.5, r=0.5, Qmin=0.0, Qmax=2.0, seed=self.seed)
		ba_griewankc = self.algo(NP=10, A=0.5, r=0.5, Qmin=0.0, Qmax=2.0, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, ba_griewank, ba_griewankc)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
