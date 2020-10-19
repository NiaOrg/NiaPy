# encoding=utf8

from NiaPy.tests.test_algorithm import AlgorithmTestCase, MyBenchmark
from NiaPy.algorithms.other import RandomSearch

class RSTestCase(AlgorithmTestCase):
	def setUp(self):
		AlgorithmTestCase.setUp(self)
		self.algo = RandomSearch

	def test_type_parameters(self):
		d = self.algo.typeParameters()
		self.assertTrue(d['NP'](10))
		self.assertFalse(d['NP'](0))
		self.assertFalse(d['NP'](-10))

	def test_custom_works_fine(self):
		ca_custom = self.algo(NP=40, seed=self.seed)
		ca_customc = self.algo(NP=40, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, ca_custom, ca_customc, MyBenchmark())

	def test_griewank_works_fine(self):
		ca_griewank = self.algo(NP=40, seed=self.seed)
		ca_griewankc = self.algo(NP=40, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, ca_griewank, ca_griewankc)
