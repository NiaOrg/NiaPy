# pylint: disable=line-too-long
from NiaPy.tests.test_algorithm import AlgorithmTestCase, MyBenchmark
from NiaPy.algorithms.modified import HybridBatAlgorithm

class HBATestCase(AlgorithmTestCase):
	def test_algorithm_info(self):
		self.assertIsNotNone(HybridBatAlgorithm.algorithmInfo())

	def test_type_parameters(self):
		d = HybridBatAlgorithm.typeParameters()
		self.assertIsNotNone(d.pop('NP', None))
		self.assertIsNotNone(d.pop('F', None))
		self.assertIsNotNone(d.pop('CR', None))

	def test_custom_works_fine(self):
		hba_custom = HybridBatAlgorithm(NP=40, A=0.5, r=0.5, F=0.5, CR=0.9, Qmin=0.0, Qmax=2.0, seed=self.seed)
		hba_customc = HybridBatAlgorithm(NP=40, A=0.5, r=0.5, F=0.5, CR=0.9, Qmin=0.0, Qmax=2.0, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, hba_custom, hba_customc, MyBenchmark())

	def test_griewank_works_fine(self):
		hba_griewank = HybridBatAlgorithm(NP=40, A=0.5, r=0.5, F=0.5, CR=0.9, Qmin=0.0, Qmax=2.0, seed=self.seed)
		hba_griewankc = HybridBatAlgorithm(NP=40, A=0.5, r=0.5, F=0.5, CR=0.9, Qmin=0.0, Qmax=2.0, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, hba_griewank, hba_griewankc)
