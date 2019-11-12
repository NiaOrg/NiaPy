# encoding=utf8
from NiaPy.tests.test_algorithm import AlgorithmTestCase, MyBenchmark
from NiaPy.algorithms.other import NelderMeadMethod

class NMMTestCase(AlgorithmTestCase):
	def setUp(self):
		AlgorithmTestCase.setUp(self)
		self.algo = NelderMeadMethod

	def test_algorithm_info(self):
		self.assertIsNotNone(NelderMeadMethod.algorithmInfo())

	def test_type_parameters(self):
		d = NelderMeadMethod.typeParameters()
		self.assertIsNotNone(d.get('NP', None))
		self.assertIsNotNone(d.get('alpha', None))
		self.assertIsNotNone(d.get('gamma', None))
		self.assertIsNotNone(d.get('rho', None))
		self.assertIsNotNone(d.get('sigma', None))

	def test_custom_works_fine(self):
		nmm_custom = self.algo(n=10, C_a=2, C_r=0.5, seed=self.seed)
		nmm_customc = self.algo(n=10, C_a=2, C_r=0.5, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, nmm_custom, nmm_customc, MyBenchmark())

	def test_griewank_works_fine(self):
		nmm_griewank = self.algo(n=10, C_a=5, C_r=0.5, seed=self.seed)
		nmm_griewankc = self.algo(n=10, C_a=5, C_r=0.5, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, nmm_griewank, nmm_griewankc)

	def test_michalewichz_works_fine(self):
		nmm_griewank = self.algo(n=10, C_a=5, C_r=0.5, seed=self.seed)
		nmm_griewankc = self.algo(n=10, C_a=5, C_r=0.5, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, nmm_griewank, nmm_griewankc, 'michalewicz', nGEN=10000000)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
