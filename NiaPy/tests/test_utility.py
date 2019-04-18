# encoding=utf8
# pylint: disable=mixed-indentation, multiple-statements, unused-variable, unused-argument, redefined-builtin, no-init, line-too-long, broad-except
from unittest import TestCase

from numpy import full, random as rnd, inf, sum, array_equal, asarray

from NiaPy.util import Utility, StoppingTask, ThrowingTask, fullArray, limitRepair, limitInversRepair, wangRepair, randRepair, reflectRepair, ScaledTask, TaskConvPrint, TaskComposition, FesException, GenException, RefException  # TimeException

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
		self.assertTrue(self.is_solution_in_range(limitRepair(self.x1, self.lower, self.upper)))
		self.assertTrue(self.is_solution_in_range(limitRepair(self.x2, self.lower, self.upper)))
		self.assertTrue(self.is_solution_in_range(limitRepair(self.x3, self.lower, self.upper)))
		self.assertTrue(self.is_solution_in_range(limitRepair(self.x4, self.lower, self.upper)))

	def test_limit_inverse_repair(self):
		self.assertTrue(self.is_solution_in_range(limitInversRepair(self.x1, self.lower, self.upper)))
		self.assertTrue(self.is_solution_in_range(limitInversRepair(self.x2, self.lower, self.upper)))
		self.assertTrue(self.is_solution_in_range(limitInversRepair(self.x3, self.lower, self.upper)))
		self.assertTrue(self.is_solution_in_range(limitInversRepair(self.x4, self.lower, self.upper)))

	def test_wang_repair(self):
		self.assertTrue(self.is_solution_in_range(wangRepair(self.x1, self.lower, self.upper)))
		self.assertTrue(self.is_solution_in_range(wangRepair(self.x2, self.lower, self.upper)))
		self.assertTrue(self.is_solution_in_range(wangRepair(self.x3, self.lower, self.upper)))
		self.assertTrue(self.is_solution_in_range(wangRepair(self.x4, self.lower, self.upper)))

	def test_rand_repair(self):
		self.assertTrue(self.is_solution_in_range(randRepair(self.x1, self.lower, self.upper)))
		self.assertTrue(self.is_solution_in_range(randRepair(self.x2, self.lower, self.upper)))
		self.assertTrue(self.is_solution_in_range(randRepair(self.x3, self.lower, self.upper)))
		self.assertTrue(self.is_solution_in_range(randRepair(self.x4, self.lower, self.upper)))

	def test_reflect_repair(self):
		self.assertTrue(self.is_solution_in_range(reflectRepair(self.x1, self.lower, self.upper)))
		self.assertTrue(self.is_solution_in_range(reflectRepair(self.x2, self.lower, self.upper)))
		self.assertTrue(self.is_solution_in_range(reflectRepair(self.x3, self.lower, self.upper)))
		self.assertTrue(self.is_solution_in_range(reflectRepair(self.x4, self.lower, self.upper)))

class StoppingTaskBaseTestCase(TestCase):
	r"""Test case for testing `Task`, `StoppingTask` and `CountingTask` classes.

	Date:
		April 2019

	Author:
		Klemen Berkovič

	See Also:
		* :class:`NiaPy.util.Task`
		* :class:`NiaPy.util.CountingTask`
		* :class:`NiaPy.util.StoppingTask`
	"""
	def setUp(self):
		self.D = 6
		self.Lower, self.Upper = [2, 1, 1], [10, 10, 2]
		self.task = StoppingTask(Lower=self.Lower, Upper=self.Upper, D=self.D)

	def test_dim_ok(self):
		self.assertEqual(self.D, self.task.D)
		self.assertEqual(self.D, self.task.dim())

	def test_lower(self):
		self.assertTrue(array_equal(fullArray(self.Lower, self.D), self.task.Lower))
		self.assertTrue(array_equal(fullArray(self.Lower, self.D), self.task.bcLower()))

	def test_upper(self):
		self.assertTrue(array_equal(fullArray(self.Upper, self.D), self.task.Upper))
		self.assertTrue(array_equal(fullArray(self.Upper, self.D), self.task.bcUpper()))

	def test_range(self):
		self.assertTrue(array_equal(fullArray(self.Upper, self.D) - fullArray(self.Lower, self.D), self.task.bRange))
		self.assertTrue(array_equal(fullArray(self.Upper, self.D) - fullArray(self.Lower, self.D), self.task.bcRange()))

	def test_ngens(self):
		self.assertEqual(inf, self.task.nGEN)

	def test_nfess(self):
		self.assertEqual(inf, self.task.nFES)

	def test_stop_cond(self):
		self.assertFalse(self.task.stopCond())

	def test_stop_condi(self):
		self.assertFalse(self.task.stopCondI())

	def test_eval(self):
		self.assertRaises(AttributeError, lambda: self.task.eval([]))

	def test_evals(self):
		self.assertEqual(0, self.task.evals())

	def test_iters(self):
		self.assertEqual(0, self.task.iters())

	def test_next_iter(self):
		self.assertEqual(None, self.task.nextIter())

	def test_is_feasible(self):
		self.assertFalse(self.task.isFeasible(fullArray([1, 2, 3], self.D)))

class StoppingTaskTestCase(TestCase):
	r"""Test case for testing `Task`, `StoppingTask` and `CountingTask` classes.

	Date:
		April 2019

	Author:
		Klemen Berkovič

	See Also:
		* :class:`NiaPy.util.Task`
		* :class:`NiaPy.util.CountingTask`
		* :class:`NiaPy.util.StoppingTask`
	"""
	def setUp(self):
		self.D, self.nFES, self.nGEN = 10, 10, 10
		self.t = StoppingTask(D=self.D, nFES=self.nFES, nGEN=self.nGEN, refValue=1, benchmark=MyBenchmark())

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
		for i in range(self.nGEN): self.assertFalse(self.t.stopCondI(), msg='Error at %s iteration!!!' % (i))
		self.assertTrue(self.t.stopCondI())

	def test_eval_fine(self):
		x = full(self.D, 1.0)
		for i in range(self.nFES): self.assertAlmostEqual(self.t.eval(x), self.D, msg='Error at %s iteration!!!' % (i))
		self.assertTrue(self.t.stopCond())

	def test_eval_over_nFES_fine(self):
		x = full(self.D, 1.0)
		for i in range(self.nFES): self.t.eval(x)
		self.assertEqual(inf, self.t.eval(x))
		self.assertTrue(self.t.stopCond())

	def test_eval_over_nGEN_fine(self):
		x = full(self.D, 1.0)
		for i in range(self.nGEN): self.t.nextIter()
		self.assertEqual(inf, self.t.eval(x))
		self.assertTrue(self.t.stopCond())

	def test_nFES_count_fine(self):
		x = full(self.D, 1.0)
		for i in range(self.nFES):
			self.t.eval(x)
			self.assertEqual(self.t.Evals, i + 1, 'Error at %s. evaluation' % (i + 1))

	def test_nGEN_count_fine(self):
		x = full(self.D, 1.0)
		for i in range(self.nGEN):
			self.t.nextIter()
			self.assertEqual(self.t.Iters, i + 1, 'Error at %s. iteration' % (i + 1))

	def test_stopCond_evals_fine(self):
		x = full(self.D, 1.0)
		for i in range(self.nFES - 1):
			self.t.eval(x)
			self.assertFalse(self.t.stopCond())
		self.t.eval(x)
		self.assertTrue(self.t.stopCond())

	def test_stopCond_iters_fine(self):
		x = full(self.D, 1.0)
		for i in range(self.nGEN - 1):
			self.t.nextIter()
			self.assertFalse(self.t.stopCond())
		self.t.nextIter()
		self.assertTrue(self.t.stopCond())

	def test_stopCond_refValue_fine(self):
		x = full(self.D, 1.0)
		for i in range(self.nGEN - 5):
			self.assertFalse(self.t.stopCond())
			self.assertEqual(self.D, self.t.eval(x))
			self.t.nextIter()
		x = full(self.D, 0.0)
		self.assertEqual(0, self.t.eval(x))
		self.assertTrue(self.t.stopCond())
		self.assertEqual(self.nGEN - 5, self.t.Iters)

class ThrowingTaskTestCase(TestCase):
	r"""Test case for testing `ThrowingTask` class.

	Date:
		April 2019

	Author:
		Klemen Berkovič

	See Also:
		* :class:`NiaPy.util.ThrowingTask`
	"""
	def setUp(self):
		self.D, self.nFES, self.nGEN = 10, 10, 10
		self.t = ThrowingTask(D=self.D, nFES=self.nFES, nGEN=self.nGEN, refValue=0, benchmark=MyBenchmark())

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
		for i in range(self.nGEN): self.assertFalse(self.t.stopCondI())
		self.assertTrue(self.t.stopCondI())

	def test_eval_fine(self):
		x = full(self.D, 1.0)
		for i in range(self.nFES): self.assertAlmostEqual(self.t.eval(x), self.D, msg='Error at %s iteration!!!' % (i))
		self.assertRaises(FesException, lambda: self.t.eval(x))

	def test_eval_over_nFES_fine(self):
		x = full(self.D, 1.0)
		for i in range(self.nFES):
			self.t.eval(x)
		self.assertRaises(FesException, lambda: self.t.eval(x))

	def test_eval_over_nGEN_fine(self):
		x = full(self.D, 1.0)
		for i in range(self.nGEN): self.t.nextIter()
		self.assertRaises(GenException, lambda: self.t.eval(x))

	def test_nFES_count_fine(self):
		x = full(self.D, 1.0)
		for i in range(self.nFES):
			self.t.eval(x)
			self.assertEqual(self.t.Evals, i + 1, 'Error at %s. evaluation' % (i + 1))

	def test_nGEN_count_fine(self):
		x = full(self.D, 1.0)
		for i in range(self.nGEN):
			self.t.nextIter()
			self.assertEqual(self.t.Iters, i + 1, 'Error at %s. iteration' % (i + 1))

	def test_stopCond_evals_fine(self):
		x = full(self.D, 1.0)
		for i in range(self.nFES - 1):
			self.t.eval(x)
			self.assertFalse(self.t.stopCond())
		self.t.eval(x)
		self.assertTrue(self.t.stopCond())

	def test_stopCond_iters_fine(self):
		x = full(self.D, 1.0)
		for i in range(self.nGEN - 1):
			self.t.nextIter()
			self.assertFalse(self.t.stopCond())
		self.t.nextIter()
		self.assertTrue(self.t.stopCond())

	def test_stopCond_refValue_fine(self):
		x = full(self.D, 1.0)
		for i in range(self.nGEN - 5):
			self.assertFalse(self.t.stopCond())
			self.assertEqual(self.D, self.t.eval(x))
			self.t.nextIter()
		x = full(self.D, 0.0)
		self.assertEqual(0, self.t.eval(x))
		self.assertRaises(RefException, lambda: self.t.eval(x))

class ScaledTaskTestCase(TestCase):
	def setUp(self):
		self.D, self.nFES, self.nGEN = 10, 10, 10
		self.t = StoppingTask(D=self.D, nFES=self.nFES, nGEN=self.nGEN, benchmark=MyBenchmark())
		d1, d2 = self.t.bcLower() + self.t.bcRange() / 2, self.t.bcUpper() - self.t.bcRange() * 0.2
		L, U = d1, d1 + d2
		self.tc = ScaledTask(self.t, L, U)

	def test_isFeasible_fine(self):
		x = full(self.D, 10)
		self.assertTrue(self.t.isFeasible(x))
		self.assertFalse(self.tc.isFeasible(x))
		x = full(self.D, -10)
		self.assertTrue(self.t.isFeasible(x))
		self.assertFalse(self.tc.isFeasible(x))
		x = rnd.uniform(0, 5, self.D)
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
		self.assertEqual(inf, self.t.eval(x))
		self.assertEqual(inf, self.tc.eval(x))

	def test_eval_over_nFES_fine(self):
		x = full(self.D, 0.0)
		for i in range(int(self.nFES / 2)):
			self.t.eval(x)
			self.tc.eval(x)
		self.assertEqual(inf, self.t.eval(x))
		self.assertEqual(inf, self.tc.eval(x))

	def test_eval_over_nGEN_fine(self):
		x = full(self.D, 0.0)
		for i in range(int(self.nGEN / 2)):
			self.t.nextIter()
			self.tc.nextIter()
		self.assertEqual(inf, self.t.eval(x))
		self.assertEqual(inf, self.tc.eval(x))

	def test_nFES_count_fine(self):
		x = full(self.D, 0.0)
		for i in range(self.nFES // 2):
			try: self.t.eval(x)
			except Exception: pass
			self.assertEqual(self.t.evals(), 2 * i + 1, 'Error at %s. evaluation' % (i + 1))
			try: self.tc.eval(x)
			except Exception: pass
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

class TaskConvPrintTestCase(TestCase):
	def setUp(self):
		self.D, self.nFES, self.nGEN = 10, 10, 10
		self.t = TaskConvPrint(D=self.D, nFES=self.nFES, nGEN=self.nGEN, benchmark=MyBenchmark())

	def test_eval(self):
		self.t.eval(full(self.D, 2.))
		self.assertTrue(array_equal(full(self.D, 2.), self.t.xb))
		self.t.eval(full(self.D, -1.))
		self.assertTrue(array_equal(full(self.D, -1.), self.t.xb))
		self.t.eval(full(self.D, .0))
		self.assertTrue(array_equal(full(self.D, .0), self.t.xb))

class TaskConvPlotTestCase(TestCase):
	def setUp(self):
		self.D, self.nFES, self.nGEN = 10, 10, 10
		self.t = TaskConvPrint(D=self.D, nFES=self.nFES, nGEN=self.nGEN, benchmark=MyBenchmark())

	def test_eval(self):
		pass

class TaskCompositionTestCase(TestCase):
	def setUp(self):
		self.D, self.nFES, self.nGEN = 10, 10, 10
		self.t = TaskComposition(D=self.D, nFES=self.nFES, nGEN=self.nGEN, benchmark=MyBenchmark())

	def test_eval(self):
		self.assertEqual(self.t.eval(full(self.D, 0)), inf)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
