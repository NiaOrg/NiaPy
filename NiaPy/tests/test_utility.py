# encoding=utf8
# pylint: disable=mixed-indentation, multiple-statements, unused-variable, unused-argument, redefined-builtin, old-style-class, no-init
from unittest import TestCase
from numpy import full, random as rnd, inf, sum, array_equal, asarray
from NiaPy.util import Utility, Task, fullArray, ScaledTask

class FullArrayTestCase(TestCase):
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
	def setUp(self):
		self.u = Utility()

	def test_get_bad_benchmark_fine(self):
		self.assertRaises(TypeError, lambda: self.u.get_benchmark('hihihihihihihihihi'))
		self.assertRaises(TypeError, lambda: self.u.get_benchmark(MyBenchmark))
		self.assertRaises(TypeError, lambda: self.u.get_benchmark(NoLimits))

class TaskTestCase(TestCase):
	def setUp(self):
		self.D, self.nFES, self.nGEN = 10, 10, 10
		self.t = Task(self.D, self.nFES, self.nGEN, MyBenchmark())

	def test_isFeasible_fine(self):
		x = full(self.D, 10)
		self.assertTrue(self.t.isFeasible(x))
		x = full(self.D, -10)
		self.assertTrue(self.t.isFeasible(x))
		x = rnd.uniform(-10, 10, self.D)
		self.assertTrue(self.t.isFeasible(x))
		x = full(self.D, -20)
		self.assertFalse(self.t.isFeasible(x))
		x = full(self.D, 20)
		self.assertFalse(self.t.isFeasible(x))

	def test_nextIter_fine(self):
		for i in range(self.nGEN):
			self.assertFalse(self.t.stopCond())
			self.t.nextIter()
		self.assertTrue(self.t.stopCond())

	def test_stopCondI(self):
		for i in range(self.nGEN):
			self.assertFalse(self.t.stopCondI())
		self.assertTrue(self.t.stopCondI())

	def test_eval_fine(self):
		x = full(self.D, 0.0)
		for i in range(self.nFES): self.assertAlmostEqual(self.t.eval(x), 0.0, msg='Error at %s iteration!!!' % (i))
		self.assertEqual(self.t.eval(x), inf)

	def test_eval_over_nFES_fine(self):
		x = full(self.D, 0.0)
		for i in range(self.nFES):
			self.t.eval(x)
		self.assertEqual(self.t.eval(x), inf)

	def test_eval_over_nGEN_fine(self):
		x = full(self.D, 0.0)
		for i in range(self.nGEN): self.t.nextIter()
		self.assertEqual(self.t.eval(x), inf)

	def test_nFES_count_fine(self):
		x = full(self.D, 0.0)
		for i in range(self.nFES):
			self.t.eval(x)
			self.assertEqual(self.t.Evals, i + 1, 'Error at %s. evaluation' % (i + 1))

	def test_nGEN_count_fine(self):
		x = full(self.D, 0.0)
		for i in range(self.nGEN):
			self.t.nextIter()
			self.assertEqual(self.t.Iters, i + 1, 'Error at %s. iteration' % (i + 1))

	def test_stopCond_evals_fine(self):
		x = full(self.D, 0.0)
		for i in range(self.nFES - 1):
			self.t.eval(x)
			self.assertFalse(self.t.stopCond())
		self.t.eval(x)
		self.assertTrue(self.t.stopCond())

	def test_stopCond_iters_fine(self):
		x = full(self.D, 0.0)
		for i in range(self.nGEN - 1):
			self.t.nextIter()
			self.assertFalse(self.t.stopCond())
		self.t.nextIter()
		self.assertTrue(self.t.stopCond())

class ScaledTaskTestCase(TestCase):
	def setUp(self):
		self.D, self.nFES, self.nGEN = 10, 10, 10
		self.t = Task(self.D, self.nFES, self.nGEN, MyBenchmark())
		d1, d2 = self.t.bcLower() + self.t.bcRange() / 2, self.t.bcRange() * 0.2
		L, U = d1, d1 + d2
		self.tc = ScaledTask(self.t, L, U)

	def test_isFeasible_fine(self):
		x = full(self.D, 10)
		self.assertTrue(self.t.isFeasible(x))
		self.assertTrue(self.tc.isFeasible(x))
		x = full(self.D, -10)
		self.assertTrue(self.t.isFeasible(x))
		self.assertTrue(self.tc.isFeasible(x))
		x = rnd.uniform(-10, 10, self.D)
		self.assertTrue(self.t.isFeasible(x))
		self.assertTrue(self.tc.isFeasible(x))
		x = full(self.D, -20)
		self.assertFalse(self.t.isFeasible(x))
		self.assertFalse(self.tc.isFeasible(x))
		x = full(self.D, 20)
		self.assertFalse(self.t.isFeasible(x))
		self.assertFalse(self.tc.isFeasible(x))

	def test_nextIter_fine(self):
		for i in range(self.nGEN):
			self.assertFalse(self.t.stopCond())
			self.assertFalse(self.tc.stopCond())
			self.t.nextIter()
		self.assertTrue(self.t.stopCond())
		self.assertTrue(self.tc.stopCond())

	def test_nextIter_two_fine(self):
		for i in range(int(self.nGEN / 2)):
			self.assertFalse(self.t.stopCond())
			self.assertFalse(self.tc.stopCond())
			self.tc.nextIter()
			self.t.nextIter()
		self.assertTrue(self.t.stopCond())
		self.assertTrue(self.tc.stopCond())

	def test_stopCondI(self):
		for i in range(int(self.nGEN / 2)):
			self.assertFalse(self.t.stopCondI())
			self.assertFalse(self.tc.stopCondI())
		self.assertTrue(self.t.stopCondI())
		self.assertTrue(self.tc.stopCondI())

	def test_eval_fine(self):
		x = full(self.D, 0.0)
		for i in range(int(self.nFES / 2)):
			self.assertAlmostEqual(self.t.eval(x), 0.0, msg='Error at %s iteration!!!' % (i))
			self.assertAlmostEqual(self.tc.eval(x), 0.0, msg='Error at %s iteration!!!' % (i))
		self.assertEqual(self.t.eval(x), inf)
		self.assertEqual(self.tc.eval(x), inf)

	def test_eval_over_nFES_fine(self):
		x = full(self.D, 0.0)
		for i in range(int(self.nFES / 2)):
			self.t.eval(x)
			self.tc.eval(x)
		self.assertEqual(self.t.eval(x), inf)
		self.assertEqual(self.tc.eval(x), inf)

	def test_eval_over_nGEN_fine(self):
		x = full(self.D, 0.0)
		for i in range(int(self.nGEN / 2)):
			self.t.nextIter()
			self.tc.nextIter()
		self.assertEqual(self.t.eval(x), inf)
		self.assertEqual(self.tc.eval(x), inf)

	def test_nFES_count_fine(self):
		x = full(self.D, 0.0)
		for i in range(self.nFES):
			self.t.eval(x)
			self.assertEqual(self.t.evals(), 2 * i + 1, 'Error at %s. evaluation' % (i + 1))
			self.tc.eval(x)
			self.assertEqual(self.tc.evals(), 2 * i + 2, 'Error at %s. evaluation' % (i + 1))

	def test_nGEN_count_fine(self):
		x = full(self.D, 0.0)
		for i in range(int((self.nFES - 1 if self.nFES % 2 > 0 else self.nFES - 2) / 2)):
			self.t.nextIter()
			self.assertEqual(self.t.iters(), 2 * i + 1, 'Error at %s. iteration' % (i + 1))
			self.tc.nextIter()
			self.assertEqual(self.tc.iters(), 2 * i + 2, 'Error at %s. iteration' % (i + 1))

	def test_stopCond_evals_fine(self):
		x = full(self.D, 0.0)
		for i in range(int((self.nFES - 1 if self.nFES % 2 > 0 else self.nFES - 2) / 2)):
			self.assertFalse(self.t.stopCond())
			self.t.eval(x)
			self.assertFalse(self.tc.stopCond())
			self.tc.eval(x)
		self.assertFalse(self.t.stopCond())
		self.t.eval(x)
		self.assertFalse(self.tc.stopCond())
		self.tc.eval(x)
		self.assertTrue(self.t.stopCond())
		self.assertTrue(self.tc.stopCond())

	def test_stopCond_iters_fine(self):
		x = full(self.D, 0.0)
		for i in range(int((self.nGEN - 1 if self.nGEN % 2 > 0 else self.nGEN - 2) / 2)):
			self.assertFalse(self.t.stopCond())
			self.t.nextIter()
			self.assertFalse(self.tc.stopCond())
			self.tc.nextIter()
		self.assertFalse(self.t.stopCond())
		self.t.nextIter()
		self.assertFalse(self.tc.stopCond())
		self.tc.nextIter()
		self.assertTrue(self.t.stopCond())
		self.assertTrue(self.tc.stopCond())

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
