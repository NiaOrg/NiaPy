# encoding=utf8
# pylint: disable=mixed-indentation, multiple-statements
from NiaPy.tests.test_algorithm import AlgorithmTestCase, MyBenchmark
from NiaPy.algorithms.other import SimulatedAnnealing
from NiaPy.algorithms.other.sa import coolLinear

class SATestCase(AlgorithmTestCase):
	def test_type_parameters(self):
		d = SimulatedAnnealing.typeParameters()
		self.assertTrue(d['delta'](1))
		self.assertFalse(d['delta'](0))
		self.assertFalse(d['delta'](-1))
		self.assertTrue(d['T'](1))
		self.assertFalse(d['T'](0))
		self.assertFalse(d['T'](-1))
		self.assertTrue(d['deltaT'](1))
		self.assertFalse(d['deltaT'](0))
		self.assertFalse(d['deltaT'](-1))
		self.assertTrue(d['epsilon'](0.1))
		self.assertFalse(d['epsilon'](-0.1))
		self.assertFalse(d['epsilon'](10))

	def test_custom_works_fine(self):
		ca_custom = SimulatedAnnealing(seed=self.seed)
		ca_customc = SimulatedAnnealing(seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, ca_custom, ca_customc, MyBenchmark())

	def test_griewank_works_fine(self):
		ca_griewank = SimulatedAnnealing(seed=self.seed)
		ca_griewankc = SimulatedAnnealing(seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, ca_griewank, ca_griewankc)

	def test_custom1_works_fine(self):
		ca_custom = SimulatedAnnealing(seed=self.seed, coolingMethod=coolLinear)
		ca_customc = SimulatedAnnealing(seed=self.seed, coolingMethod=coolLinear)
		AlgorithmTestCase.algorithm_run_test(self, ca_custom, ca_customc, MyBenchmark())

	def test_griewank1_works_fine(self):
		ca_griewank = SimulatedAnnealing(seed=self.seed, coolingMethod=coolLinear)
		ca_griewankc = SimulatedAnnealing(seed=self.seed, coolingMethod=coolLinear)
		AlgorithmTestCase.algorithm_run_test(self, ca_griewank, ca_griewankc)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
