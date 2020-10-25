# encoding=utf8
from NiaPy.tests.test_algorithm import AlgorithmTestCase, MyBenchmark
from NiaPy.algorithms.modified import ParameterFreeBatAlgorithm

class HBATestCase(AlgorithmTestCase):
	def setUp(self):
		AlgorithmTestCase.setUp(self)
		self.algo = ParameterFreeBatAlgorithm

	def test_custom_works_fine(self):
		plba_custom = self.algo()
		AlgorithmTestCase.test_algorithm_run(self, plba_custom, MyBenchmark())
