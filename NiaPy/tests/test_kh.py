# encoding=utf8
# pylint: disable=mixed-indentation, multiple-statements
from NiaPy.tests.test_algorithm import AlgorithmTestCase, MyBenchmark
from NiaPy.algorithms.basic import KrillHerdV1, KrillHerdV2, KrillHerdV3, KrillHerdV4, KrillHerdV11

class KHV1TestCase(AlgorithmTestCase):
	r"""Test case for KrillHerdV1 algorithm.

	Date:
		April 2019

	Author:
		Klemen Berkovič

	See Also:
		* :class:`NiaPy.tests.test_algorithm.AlgorithmTestCase`
	"""
	def test_type_parametes(self):
		d = KrillHerdV1.typeParameters()
		self.assertIsNotNone(d.get('N_max', None))
		self.assertIsNotNone(d.get('V_f', None))
		self.assertIsNotNone(d.get('D_max', None))
		self.assertIsNotNone(d.get('C_t', None))
		self.assertIsNotNone(d.get('W_n', None))
		self.assertIsNotNone(d.get('W_f', None))
		self.assertIsNotNone(d.get('d_s', None))
		self.assertIsNotNone(d.get('nn', None))
		self.assertIsNotNone(d.get('Cr', None))
		self.assertIsNotNone(d.get('Mu', None))
		self.assertIsNotNone(d.get('epsilon', None))

	def test_custom_works_fine(self):
		kh_custom = KrillHerdV1(n=10, C_a=2, C_r=0.5, seed=self.seed)
		kh_customc = KrillHerdV1(n=10, C_a=2, C_r=0.5, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, kh_custom, kh_customc, MyBenchmark())

	def test_griewank_works_fine(self):
		kh_griewank = KrillHerdV1(n=10, C_a=5, C_r=0.5, seed=self.seed)
		kh_griewankc = KrillHerdV1(n=10, C_a=5, C_r=0.5, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, kh_griewank, kh_griewankc)

class KHV2TestCase(AlgorithmTestCase):
	r"""Test case for KrillHerdV2 algorithm.

	Date:
		April 2019

	Author:
		Klemen Berkovič

	See Also:
		* :class:`NiaPy.tests.test_algorithm.AlgorithmTestCase`
	"""
	def test_type_parametes(self):
		d = KrillHerdV2.typeParameters()
		self.assertIsNotNone(d.get('N_max', None))
		self.assertIsNotNone(d.get('V_f', None))
		self.assertIsNotNone(d.get('D_max', None))
		self.assertIsNotNone(d.get('C_t', None))
		self.assertIsNotNone(d.get('W_n', None))
		self.assertIsNotNone(d.get('W_f', None))
		self.assertIsNotNone(d.get('d_s', None))
		self.assertIsNotNone(d.get('nn', None))
		self.assertIsNotNone(d.get('Cr', None))
		self.assertIsNone(d.get('Mu', None))
		self.assertIsNotNone(d.get('epsilon', None))

	def test_custom_works_fine(self):
		kh_custom = KrillHerdV2(n=10, C_a=2, C_r=0.5, seed=self.seed)
		kh_customc = KrillHerdV2(n=10, C_a=2, C_r=0.5, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, kh_custom, kh_customc, MyBenchmark())

	def test_griewank_works_fine(self):
		kh_griewank = KrillHerdV2(n=10, C_a=5, C_r=0.5, seed=self.seed)
		kh_griewankc = KrillHerdV2(n=10, C_a=5, C_r=0.5, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, kh_griewank, kh_griewankc)

class KHV3TestCase(AlgorithmTestCase):
	r"""Test case for KrillHerdV3 algorithm.

	Date:
		April 2019

	Author:
		Klemen Berkovič

	See Also:
		* :class:`NiaPy.tests.test_algorithm.AlgorithmTestCase`
	"""
	def test_type_parametes(self):
		d = KrillHerdV3.typeParameters()
		self.assertIsNotNone(d.get('N_max', None))
		self.assertIsNotNone(d.get('V_f', None))
		self.assertIsNotNone(d.get('D_max', None))
		self.assertIsNotNone(d.get('C_t', None))
		self.assertIsNotNone(d.get('W_n', None))
		self.assertIsNotNone(d.get('W_f', None))
		self.assertIsNotNone(d.get('d_s', None))
		self.assertIsNotNone(d.get('nn', None))
		self.assertIsNone(d.get('Cr', None))
		self.assertIsNotNone(d.get('Mu', None))
		self.assertIsNotNone(d.get('epsilon', None))

	def test_custom_works_fine(self):
		kh_custom = KrillHerdV3(n=10, C_a=2, C_r=0.5, seed=self.seed)
		kh_customc = KrillHerdV3(n=10, C_a=2, C_r=0.5, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, kh_custom, kh_customc, MyBenchmark())

	def test_griewank_works_fine(self):
		kh_griewank = KrillHerdV3(n=10, C_a=5, C_r=0.5, seed=self.seed)
		kh_griewankc = KrillHerdV3(n=10, C_a=5, C_r=0.5, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, kh_griewank, kh_griewankc)

class KHV4TestCase(AlgorithmTestCase):
	r"""Test case for KrillHerdV4 algorithm.

	Date:
		April 2019

	Author:
		Klemen Berkovič

	See Also:
		* :class:`NiaPy.tests.test_algorithm.AlgorithmTestCase`
	"""
	def test_type_parametes(self):
		d = KrillHerdV4.typeParameters()
		self.assertIsNotNone(d.get('N_max', None))
		self.assertIsNotNone(d.get('V_f', None))
		self.assertIsNotNone(d.get('D_max', None))
		self.assertIsNotNone(d.get('C_t', None))
		self.assertIsNotNone(d.get('W_n', None))
		self.assertIsNotNone(d.get('W_f', None))
		self.assertIsNotNone(d.get('d_s', None))
		self.assertIsNotNone(d.get('nn', None))
		self.assertIsNone(d.get('Cr', None))
		self.assertIsNone(d.get('Mu', None))
		self.assertIsNone(d.get('epsilon', None))

	def test_custom_works_fine(self):
		kh_custom = KrillHerdV4(n=10, C_a=2, C_r=0.5, seed=self.seed)
		kh_customc = KrillHerdV4(n=10, C_a=2, C_r=0.5, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, kh_custom, kh_customc, MyBenchmark())

	def test_griewank_works_fine(self):
		kh_griewank = KrillHerdV4(n=10, C_a=5, C_r=0.5, seed=self.seed)
		kh_griewankc = KrillHerdV4(n=10, C_a=5, C_r=0.5, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, kh_griewank, kh_griewankc)

class KHV11TestCase(AlgorithmTestCase):
	r"""Test case for KrillHerdV11 algorithm.

	Date:
		April 2019

	Author:
		Klemen Berkovič

	See Also:
		* :class:`NiaPy.tests.test_algorithm.AlgorithmTestCase`
	"""
	def test_type_parametes(self):
		d = KrillHerdV11.typeParameters()
		self.assertIsNotNone(d.get('N_max', None))
		self.assertIsNotNone(d.get('V_f', None))
		self.assertIsNotNone(d.get('D_max', None))
		self.assertIsNotNone(d.get('C_t', None))
		self.assertIsNotNone(d.get('W_n', None))
		self.assertIsNotNone(d.get('W_f', None))
		self.assertIsNotNone(d.get('d_s', None))
		self.assertIsNotNone(d.get('nn', None))
		self.assertIsNotNone(d.get('Cr', None))
		self.assertIsNotNone(d.get('Mu', None))
		self.assertIsNotNone(d.get('epsilon', None))

	def test_custom_works_fine(self):
		kh_custom = KrillHerdV11(n=10, C_a=2, C_r=0.5, seed=self.seed)
		kh_customc = KrillHerdV11(n=10, C_a=2, C_r=0.5, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, kh_custom, kh_customc, MyBenchmark())

	def test_griewank_works_fine(self):
		kh_griewank = KrillHerdV11(n=10, C_a=5, C_r=0.5, seed=self.seed)
		kh_griewankc = KrillHerdV11(n=10, C_a=5, C_r=0.5, seed=self.seed)
		AlgorithmTestCase.algorithm_run_test(self, kh_griewank, kh_griewankc)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
