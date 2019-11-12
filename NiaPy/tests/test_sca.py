# encoding=utf8
from NiaPy.algorithms.basic import SineCosineAlgorithm
from NiaPy.tests.test_algorithm import AlgorithmTestCase, MyBenchmark

class SCATestCase(AlgorithmTestCase):
	def setUp(self):
		AlgorithmTestCase.setUp(self)
		self.algo = SineCosineAlgorithm

	def test_algorithm_info_fine(self):
		self.assertIsNotNone(self.algo.algorithmInfo())

	def test_type_parameters(self):
		d = self.algo.typeParameters()
		self.assertIsNotNone(d.get('NP', None))
		self.assertIsNotNone(d.get('a', None))
		self.assertIsNotNone(d.get('Rmin', None))
		self.assertIsNotNone(d.get('Rmax', None))

	def test_custom_works_fine(self):
		sca_custom = self.algo(NP=35, a=7, Rmin=0.1, Rmax=3, seed=self.seed)
		sca_customc = self.algo(NP=35, a=7, Rmin=0.1, Rmax=3, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, sca_custom, sca_customc, MyBenchmark())

	def test_griewank_works_fine(self):
		sca_griewank = self.algo(NP=10, a=5, Rmin=0.01, Rmax=3, seed=self.seed)
		sca_griewankc = self.algo(NP=10, a=5, Rmin=0.01, Rmax=3, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, sca_griewank, sca_griewankc)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
