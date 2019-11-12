# encoding=utf8
from NiaPy.tests.test_algorithm import AlgorithmTestCase, MyBenchmark
from NiaPy.algorithms.basic import ArtificialBeeColonyAlgorithm

class ABCTestCase(AlgorithmTestCase):
	def setUp(self):
		AlgorithmTestCase.setUp(self)
		self.algo = ArtificialBeeColonyAlgorithm

	def test_type_parameters(self):
		d = self.algo.typeParameters()
		self.assertEqual(len(d), 2)
		self.assertTrue(d['NP'](10))
		self.assertFalse(d['NP'](0))
		self.assertFalse(d['NP'](-10))
		self.assertTrue(d['Limit'](10))
		self.assertFalse(d['Limit'](0))
		self.assertFalse(d['Limit'](-10))

	def test_custom_works_fine(self):
		abc_custom = self.algo(NP=10, Limit=2, seed=self.seed)
		abc_customc = self.algo(NP=10, Limit=2, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, abc_custom, abc_customc, MyBenchmark())

	def test_griewank_works_fine(self):
		abc_griewank = self.algo(NP=10, seed=self.seed)
		abc_griewankc = self.algo(NP=10, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, abc_griewank, abc_griewankc)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
