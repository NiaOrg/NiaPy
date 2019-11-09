# encoding=utf8
# pylint: disable=mixed-indentation, multiple-statements
from unittest import skip

from NiaPy.tests.test_algorithm import AlgorithmTestCase, MyBenchmark
from NiaPy.algorithms.modified import AdaptiveArchiveDifferentialEvolution

class JADETestCase(AlgorithmTestCase):
	def setUp(self):
		AlgorithmTestCase.setUp(self)
		self.algo = AdaptiveArchiveDifferentialEvolution

	@skip('Not implemented jet!!!')
	def test_custom_works_fine(self):
		jade_custom = self.algo(n=10, C_a=2, C_r=0.5, seed=self.seed)
		jade_customc = self.algo(n=10, C_a=2, C_r=0.5, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, jade_custom, jade_customc, MyBenchmark())

	@skip('Not implemented jet!!!')
	def test_griewank_works_fine(self):
		jade_griewank = self.algo(n=10, C_a=5, C_r=0.5, seed=self.seed)
		jade_griewankc = self.algo(n=10, C_a=5, C_r=0.5, seed=self.seed)
		AlgorithmTestCase.test_algorithm_run(self, jade_griewank, jade_griewankc)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
