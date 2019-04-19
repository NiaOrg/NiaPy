# encoding=utf8
# pylint: disable=mixed-indentation, multiple-statements, unused-variable, unused-argument, redefined-builtin, no-init, line-too-long, broad-except
from unittest import TestCase

from numpy import full, random as rnd, inf, sum, array_equal, asarray

from NiaPy.util import Utility, StoppingTask, ThrowingTask, fullArray, limit_repair, limit_inverse_repair, wang_repair, rand_repair, reflect_repair

class FullArrayTestCase(TestCase):
	r"""Test case for testing method `fullarray`.

	Date:
		April 2019

	Author:
		Klemen Berkovič

	See Also:
		* :func:`NiaPy.util.fullArray`
	"""
	def test_a_float_fine(self):
		A = fullArray(25.25, 10)
		self.assertTrue(array_equal(A, full(10, 25.25)))

	def test_a_int_fine(self):
		A = fullArray(25, 10)
		self.assertTrue(array_equal(A, full(10, 25)))

	def test_a_float_list_fine(self):
		a = [25.25 for i in range(10)]
		A = fullArray(a, 10)
		self.assertTrue(array_equal(A, full(10, 25.25)))

	def test_a_int_list_fine(self):
		a = [25 for i in range(10)]
		A = fullArray(a, 10)
		self.assertTrue(array_equal(A, full(10, 25)))

	def test_a_float_array_fine(self):
		a = asarray([25.25 for i in range(10)])
		A = fullArray(a, 10)
		self.assertTrue(array_equal(A, full(10, 25.25)))

	def test_a_int_array_fine(self):
		a = asarray([25 for i in range(10)])
		A = fullArray(a, 10)
		self.assertTrue(array_equal(A, full(10, 25)))

	def test_a_float_list1_fine(self):
		a = [25.25 + i for i in range(10)]
		A = fullArray(a, 15)
		a.extend([25.25 + i for i in range(5)])
		self.assertTrue(array_equal(A, asarray(a)))

	def test_a_int_list1_fine(self):
		a = [25 + i for i in range(10)]
		A = fullArray(a, 15)
		a.extend([25 + i for i in range(5)])
		self.assertTrue(array_equal(A, asarray(a)))

	def test_a_float_array1_fine(self):
		a = [25.25 + i for i in range(10)]
		A = fullArray(asarray(a), 15)
		a.extend([25.25 + i for i in range(5)])
		self.assertTrue(array_equal(A, asarray(a)))

	def test_a_int_array1_fine(self):
		a = [25 + i for i in range(10)]
		A = fullArray(asarray(a), 15)
		a.extend([25 + i for i in range(5)])
		self.assertTrue(array_equal(A, asarray(a)))

	def test_a_float_list2_fine(self):
		a = [25.25 + i for i in range(10)]
		A = fullArray(a, 13)
		a.extend([25.25 + i for i in range(3)])
		self.assertTrue(array_equal(A, asarray(a)))

	def test_a_int_list2_fine(self):
		a = [25 + i for i in range(10)]
		A = fullArray(a, 13)
		a.extend([25 + i for i in range(3)])
		self.assertTrue(array_equal(A, asarray(a)))

	def test_a_float_array2_fine(self):
		a = [25.25 + i for i in range(10)]
		A = fullArray(asarray(a), 13)
		a.extend([25.25 + i for i in range(3)])
		self.assertTrue(array_equal(A, asarray(a)))

	def test_a_int_array2_fine(self):
		a = [25 + i for i in range(10)]
		A = fullArray(asarray(a), 13)
		a.extend([25 + i for i in range(3)])
		self.assertTrue(array_equal(A, asarray(a)))

	def test_a_float_list3_fine(self):
		a = [25.25 + i for i in range(10)]
		A = fullArray(a, 9)
		a.remove(34.25)
		self.assertTrue(array_equal(A, asarray(a)))

	def test_a_int_list3_fine(self):
		a = [25 + i for i in range(10)]
		A = fullArray(a, 9)
		a.remove(34)
		self.assertTrue(array_equal(A, asarray(a)))

	def test_a_float_array3_fine(self):
		a = [25.25 + i for i in range(10)]
		A = fullArray(asarray(a), 9)
		a.remove(34.25)
		self.assertTrue(array_equal(A, asarray(a)))

	def test_a_int_array3_fine(self):
		a = [25 + i for i in range(10)]
		A = fullArray(asarray(a), 9)
		a.remove(34)
		self.assertTrue(array_equal(A, asarray(a)))

class NoLimits:
	@classmethod
	def function(cls):
		def evaluate(D, x): return 0
		return evaluate

class MyBenchmark:
	def __init__(self):
		self.Lower = -10.0
		self.Upper = 10

	@classmethod
	def function(cls):
		def evaluate(D, x): return sum(x ** 2)
		return evaluate

class UtilityTestCase(TestCase):
	r"""Test case for testing Utility class.

	Date:
		April 2019

	Author:
		Klemen Berkovič

	See Also:
		* :class:`NiaPy.util.Utility`
	"""
	def setUp(self):
		self.u = Utility()

	def test_get_bad_benchmark_fine(self):
		self.assertRaises(TypeError, lambda: self.u.get_benchmark('hihihihihihihihihi'))
		self.assertRaises(TypeError, lambda: self.u.get_benchmark(MyBenchmark))
		self.assertRaises(TypeError, lambda: self.u.get_benchmark(NoLimits))

class RepairMethodsTest(TestCase):
	r"""Test case for repair methods.

	Date:
		April 2019

	Author:
		Klemen Berkovič

	See Also:
		* :func:`NiaPy.util.limitRepair`
		* :func:`NiaPy.util.limitInversRepair`
		* :func:`NiaPy.util.wangRepair`
		* :func:`NiaPy.util.randRepair`
		* :func:`NiaPy.util.reflectRepair`
	"""
	def setUp(self):
		self.D = 10
		self.lower, self.upper = rnd.uniform(-10, 0, self.D), rnd.uniform(0, 10, self.D)
		self.x1 = rnd.uniform(self.lower, self.upper)
		self.x2 = rnd.uniform(self.lower, self.upper) - self.lower * 5
		self.x3 = rnd.uniform(self.lower, self.upper) + self.upper * 5
		self.x4 = rnd.uniform(self.lower, self.upper)
		self.x4[[4, 8]] = [-20, 40]

	def is_solution_in_range(self, x):
		il, iu = x < self.lower, x > self.upper
		return True not in il and True not in iu

	def test_limit_repair(self):
		self.assertTrue(self.is_solution_in_range(limit_repair(self.x1, self.lower, self.upper)))
		self.assertTrue(self.is_solution_in_range(limit_repair(self.x2, self.lower, self.upper)))
		self.assertTrue(self.is_solution_in_range(limit_repair(self.x3, self.lower, self.upper)))
		self.assertTrue(self.is_solution_in_range(limit_repair(self.x4, self.lower, self.upper)))

	def test_limit_inverse_repair(self):
		self.assertTrue(self.is_solution_in_range(limit_inverse_repair(self.x1, self.lower, self.upper)))
		self.assertTrue(self.is_solution_in_range(limit_inverse_repair(self.x2, self.lower, self.upper)))
		self.assertTrue(self.is_solution_in_range(limit_inverse_repair(self.x3, self.lower, self.upper)))
		self.assertTrue(self.is_solution_in_range(limit_inverse_repair(self.x4, self.lower, self.upper)))

	def test_wang_repair(self):
		self.assertTrue(self.is_solution_in_range(wang_repair(self.x1, self.lower, self.upper)))
		self.assertTrue(self.is_solution_in_range(wang_repair(self.x2, self.lower, self.upper)))
		self.assertTrue(self.is_solution_in_range(wang_repair(self.x3, self.lower, self.upper)))
		self.assertTrue(self.is_solution_in_range(wang_repair(self.x4, self.lower, self.upper)))

	def test_rand_repair(self):
		self.assertTrue(self.is_solution_in_range(rand_repair(self.x1, self.lower, self.upper)))
		self.assertTrue(self.is_solution_in_range(rand_repair(self.x2, self.lower, self.upper)))
		self.assertTrue(self.is_solution_in_range(rand_repair(self.x3, self.lower, self.upper)))
		self.assertTrue(self.is_solution_in_range(rand_repair(self.x4, self.lower, self.upper)))

	def test_reflect_repair(self):
		self.assertTrue(self.is_solution_in_range(reflect_repair(self.x1, self.lower, self.upper)))
		self.assertTrue(self.is_solution_in_range(reflect_repair(self.x2, self.lower, self.upper)))
		self.assertTrue(self.is_solution_in_range(reflect_repair(self.x3, self.lower, self.upper)))
		self.assertTrue(self.is_solution_in_range(reflect_repair(self.x4, self.lower, self.upper)))

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
