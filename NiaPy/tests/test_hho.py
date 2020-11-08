# encoding=utf8

from NiaPy.tests.test_algorithm import AlgorithmTestCase, MyBenchmark
from NiaPy.algorithms.basic import HarrisHawksOptimization

class HHOTestCase(AlgorithmTestCase):
	def setUp(self):
		AlgorithmTestCase.setUp(self)
		self.algo = HarrisHawksOptimization

	def test_parameter_type(self):
		d = self.algo.typeParameters()
		self.assertTrue(d['levy'](0.01))
		self.assertFalse(d['levy'](-0.01))
		self.assertTrue(d['NP'](10))
		self.assertFalse(d['NP'](-10))
		self.assertFalse(d['NP'](0))

	def test_custom_works_fine(self):
		hho_custom = self.algo(NP=20, levy=0.01, seed=self.seed)
		hho_customc = self.algo(NP=20, levy=0.01, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, hho_custom, hho_customc, MyBenchmark())

	def test_griewank_works_fine(self):
		hho_griewank = self.algo(NP=20, nFES=4000, nGEN=200, levy=0.01, seed=self.seed)
		hho_griewankc = self.algo(NP=20, nFES=4000, nGEN=200, levy=0.01, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, hho_griewank, hho_griewankc)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
