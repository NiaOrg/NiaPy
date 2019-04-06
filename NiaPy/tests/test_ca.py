# encoding=utf8
# pylint: disable=mixed-indentation, multiple-statements
from NiaPy.tests.test_algorithm import AlgorithmTestCase, MyBenchmark
from NiaPy.algorithms.basic import CamelAlgorithm

class CATestCase(AlgorithmTestCase):
	def test_type_parameters(self):
		d = CamelAlgorithm.typeParameters()
		self.assertTrue(d['NP'](1))
		self.assertFalse(d['NP'](0))
		self.assertFalse(d['NP'](-1))
		self.assertTrue(d['omega'](.1))
		self.assertTrue(d['omega'](10))
		self.assertFalse(d['omega'](None))
		self.assertTrue(d['alpha'](.342))
		self.assertTrue(d['mu'](.342))
		self.assertTrue(d['omega'](3))
		self.assertTrue(d['omega'](-3))
		self.assertFalse(d['mu'](3))
		self.assertFalse(d['mu'](-3))
		self.assertFalse(d['S_init'](-1))
		self.assertFalse(d['E_init'](-1))
		self.assertFalse(d['T_min'](-1))
		self.assertFalse(d['T_max'](-1))
		self.assertTrue(d['S_init'](10))
		self.assertTrue(d['E_init'](10))
		self.assertTrue(d['T_min'](10))
		self.assertTrue(d['T_max'](10))

	def test_custom_works_fine(self):
		ca_custom = CamelAlgorithm(NP=40, seed=self.seed)
		ca_customc = CamelAlgorithm(NP=40, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, ca_custom, ca_customc, MyBenchmark())

	def test_griewank_works_fine(self):
		ca_griewank = CamelAlgorithm(NP=40, seed=self.seed)
		ca_griewankc = CamelAlgorithm(NP=40, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, ca_griewank, ca_griewankc)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
