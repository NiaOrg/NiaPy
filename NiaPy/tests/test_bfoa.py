# encoding=utf8

from NiaPy.tests.test_algorithm import AlgorithmTestCase, MyBenchmark
from NiaPy.algorithms.basic import BacterialForagingOptimizationAlgorithm


class BFOATestCase(AlgorithmTestCase):
	def setUp(self):
		AlgorithmTestCase.setUp(self)
		self.algo = BacterialForagingOptimizationAlgorithm

	def test_parameter_type(self):
		parameters = self.algo.typeParameters()
		self.assertTrue(parameters['n_chemotactic'](10))
		self.assertFalse(parameters['n_chemotactic'](0))
		self.assertFalse(parameters['n_chemotactic'](10.0))
		self.assertFalse(parameters['n_chemotactic'](-10))

		self.assertTrue(parameters['n_swim'](10))
		self.assertFalse(parameters['n_swim'](0))
		self.assertFalse(parameters['n_swim'](10.0))
		self.assertFalse(parameters['n_swim'](-10))

		self.assertTrue(parameters['n_reproduction'](10))
		self.assertFalse(parameters['n_reproduction'](0))
		self.assertFalse(parameters['n_reproduction'](10.0))
		self.assertFalse(parameters['n_reproduction'](-10))

		self.assertTrue(parameters['n_elimination'](10))
		self.assertFalse(parameters['n_elimination'](0))
		self.assertFalse(parameters['n_elimination'](10.0))
		self.assertFalse(parameters['n_elimination'](-10))

		self.assertTrue(parameters['prob_elimination'](0.25))
		self.assertFalse(parameters['prob_elimination'](1))
		self.assertFalse(parameters['prob_elimination'](-0.25))
		self.assertFalse(parameters['prob_elimination'](1.25))

		self.assertTrue(parameters['step_size'](0.1))
		self.assertFalse(parameters['step_size'](1))
		self.assertFalse(parameters['step_size'](-1.0))

		self.assertTrue(parameters['d_attract'](0.2))
		self.assertFalse(parameters['d_attract'](1))
		self.assertFalse(parameters['d_attract'](-1.0))

		self.assertTrue(parameters['w_attract'](0.1))
		self.assertFalse(parameters['w_attract'](1))
		self.assertFalse(parameters['w_attract'](-1.0))

		self.assertTrue(parameters['h_repel'](0.1))
		self.assertFalse(parameters['h_repel'](1))
		self.assertFalse(parameters['h_repel'](-1.0))

		self.assertTrue(parameters['w_repel'](10.0))
		self.assertFalse(parameters['w_repel'](1))
		self.assertFalse(parameters['w_repel'](-1.0))

	def test_custom_works_fine(self):
		bfoa_custom = self.algo(NP=50, n_chemotactic=100, n_reproduction=4, n_elimination=1, step_size=0.1, seed=self.seed)
		bfoa_customc = self.algo(NP=50, n_chemotactic=100, n_reproduction=4, n_elimination=1, step_size=0.1, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, bfoa_custom, bfoa_customc, MyBenchmark())

	def test_griewank_works_fine(self):
		bfoa_custom = self.algo(NP=50, n_chemotactic=100, n_reproduction=4, n_elimination=1, step_size=0.1, seed=self.seed)
		bfoa_customc = self.algo(NP=50, n_chemotactic=100, n_reproduction=4, n_elimination=1, step_size=0.1, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, bfoa_custom, bfoa_customc)

	def test_griewank_works_fine_nfes(self):
		bfoa_custom = self.algo(NP=50, n_chemotactic=100, n_reproduction=4, n_elimination=1, step_size=0.1, seed=self.seed)
		bfoa_customc = self.algo(NP=50, n_chemotactic=100, n_reproduction=4, n_elimination=1, step_size=0.1, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, bfoa_custom, bfoa_customc, nFES=10000)

	def test_griewank_works_fine_ngen(self):
		bfoa_custom = self.algo(NP=50, n_chemotactic=100, n_reproduction=4, n_elimination=1, step_size=0.1, seed=self.seed)
		bfoa_customc = self.algo(NP=50, n_chemotactic=100, n_reproduction=4, n_elimination=1, step_size=0.1, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, bfoa_custom, bfoa_customc, nGEN=10000)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
