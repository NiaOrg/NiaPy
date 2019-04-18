# encoding=utf8
# pylint: disable=mixed-indentation, multiple-statements
from NiaPy.tests.test_algorithm import AlgorithmTestCase, MyBenchmark
from NiaPy.algorithms.other import HillClimbAlgorithm

class HCTestCase(AlgorithmTestCase):
	r"""Test case for HillClimbAlgorithm algorithm.

	Date:
		April 2019

	Author:
		Klemen Berkovič and Jan Popič

	See Also:
		* :class:`NiaPy.algorithms.other.HillClimbAlgorithm`
	"""
	def test_algorithm_info_fine(self):
		self.assertIsNotNone(HillClimbAlgorithm.algorithmInfo())

	def test_type_parameters_fine(self):
		d = HillClimbAlgorithm.typeParameters()
		self.assertIsNotNone(d.get('delta', None))

	def test_custom_works_fine(self):
		ihc_custom = HillClimbAlgorithm(delta=0.4, seed=self.seed)
		ihc_customc = HillClimbAlgorithm(delta=0.4, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, ihc_custom, ihc_customc, MyBenchmark())

	def test_griewank_works_fine(self):
		ihc_griewank = HillClimbAlgorithm(delta=0.1, seed=self.seed)
		ihc_griewankc = HillClimbAlgorithm(delta=0.1, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, ihc_griewank, ihc_griewankc)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
