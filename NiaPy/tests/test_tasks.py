# encoding=utf8
from unittest import TestCase

from numpy import full, random as rnd, inf, array_equal

from NiaPy.util import fullArray, StoppingTask, ThrowingTask, ScaledTask, TaskComposition, TaskPlotBest, TaskLogBest, GenException, FesException, RefException

class MyBenchmark:
	def __init__(self):
		self.Lower = -10.0
		self.Upper = 10

	@classmethod
	def function(cls):
		def evaluate(D, x): return sum(x ** 2)
		return evaluate

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
		self.assertEqual(self.D, self.task.dim)

	def test_lower(self):
		self.assertTrue(array_equal(fullArray(self.Lower, self.D), self.task.lower))

	def test_upper(self):
		self.assertTrue(array_equal(fullArray(self.Upper, self.D), self.task.upper))

	def test_range(self):
		self.assertTrue(array_equal(fullArray(self.Upper, self.D) - fullArray(self.Lower, self.D), self.task.range))

	def test_ngens(self):
		self.assertEqual(inf, self.task.nGEN)

	def test_nfess(self):
		self.assertEqual(inf, self.task.nFES)

	def test_stop_cond(self):
		self.assertFalse(self.task.is_stopping_cond())

	def test_stop_condi(self):
		self.assertFalse(self.task.is_stopping_cond_next_iter())

	def test_eval(self):
		self.assertRaises(AttributeError, lambda: self.task.eval([]))

	def test_evals(self):
		self.assertEqual(0, self.task.no_eval)

	def test_iters(self):
		self.assertEqual(0, self.task.no_iter)

	def test_next_iter(self):
		self.assertEqual(None, self.task.next_iter())

	def test_is_feasible(self):
		self.assertFalse(self.task.is_feasible(fullArray([1, 2, 3], self.D)))

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
		self.assertTrue(self.t.is_feasible(x))
		x = full(self.D, -10)
		self.assertTrue(self.t.is_feasible(x))
		x = rnd.uniform(-10, 10, self.D)
		self.assertTrue(self.t.is_feasible(x))
		x = full(self.D, -20)
		self.assertFalse(self.t.is_feasible(x))
		x = full(self.D, 20)
		self.assertFalse(self.t.is_feasible(x))

	def test_nextIter_fine(self):
		for i in range(self.nGEN):
			self.assertFalse(self.t.is_stopping_cond())
			self.t.next_iter()
		self.assertTrue(self.t.is_stopping_cond())

	def test_stopCondI(self):
		for i in range(self.nGEN): self.assertFalse(self.t.is_stopping_cond_next_iter(), msg='Error at %s iteration!!!' % (i))
		self.assertTrue(self.t.is_stopping_cond_next_iter())

	def test_eval_fine(self):
		x = full(self.D, 1.0)
		for i in range(self.nFES): self.assertAlmostEqual(self.t.eval(x), self.D, msg='Error at %s iteration!!!' % (i))
		self.assertTrue(self.t.is_stopping_cond())

	def test_eval_over_nFES_fine(self):
		x = full(self.D, 1.0)
		for i in range(self.nFES): self.t.eval(x)
		self.assertEqual(inf, self.t.eval(x))
		self.assertTrue(self.t.is_stopping_cond())

	def test_eval_over_nGEN_fine(self):
		x = full(self.D, 1.0)
		for i in range(self.nGEN): self.t.next_iter()
		self.assertEqual(inf, self.t.eval(x))
		self.assertTrue(self.t.is_stopping_cond())

	def test_nFES_count_fine(self):
		x = full(self.D, 1.0)
		for i in range(self.nFES):
			self.t.eval(x)
			self.assertEqual(self.t.no_eval, i + 1, 'Error at %s. evaluation' % (i + 1))

	def test_nGEN_count_fine(self):
		x = full(self.D, 1.0)
		for i in range(self.nGEN):
			self.t.next_iter()
			self.assertEqual(self.t.no_iter, i + 1, 'Error at %s. iteration' % (i + 1))

	def test_stopCond_evals_fine(self):
		x = full(self.D, 1.0)
		for i in range(self.nFES - 1):
			self.t.eval(x)
			self.assertFalse(self.t.is_stopping_cond())
		self.t.eval(x)
		self.assertTrue(self.t.is_stopping_cond())

	def test_stopCond_iters_fine(self):
		x = full(self.D, 1.0)
		for i in range(self.nGEN - 1):
			self.t.next_iter()
			self.assertFalse(self.t.is_stopping_cond())
		self.t.next_iter()
		self.assertTrue(self.t.is_stopping_cond())

	def test_stopCond_refValue_fine(self):
		x = full(self.D, 1.0)
		for i in range(self.nGEN - 5):
			self.assertFalse(self.t.is_stopping_cond())
			self.assertEqual(self.D, self.t.eval(x))
			self.t.next_iter()
		x = full(self.D, 0.0)
		self.assertEqual(0, self.t.eval(x))
		self.assertTrue(self.t.is_stopping_cond())
		self.assertEqual(self.nGEN - 5, self.t.no_iter)

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
		self.assertTrue(self.t.is_feasible(x))
		x = full(self.D, -10)
		self.assertTrue(self.t.is_feasible(x))
		x = rnd.uniform(-10, 10, self.D)
		self.assertTrue(self.t.is_feasible(x))
		x = full(self.D, -20)
		self.assertFalse(self.t.is_feasible(x))
		x = full(self.D, 20)
		self.assertFalse(self.t.is_feasible(x))

	def test_nextIter_fine(self):
		for i in range(self.nGEN):
			self.assertFalse(self.t.is_stopping_cond())
			self.t.next_iter()
		self.assertTrue(self.t.is_stopping_cond())

	def test_stopCondI(self):
		for i in range(self.nGEN): self.assertFalse(self.t.is_stopping_cond_next_iter())
		self.assertTrue(self.t.is_stopping_cond_next_iter())

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
		for i in range(self.nGEN): self.t.next_iter()
		self.assertRaises(GenException, lambda: self.t.eval(x))

	def test_nFES_count_fine(self):
		x = full(self.D, 1.0)
		for i in range(self.nFES):
			self.t.eval(x)
			self.assertEqual(self.t.no_eval, i + 1, 'Error at %s. evaluation' % (i + 1))

	def test_nGEN_count_fine(self):
		x = full(self.D, 1.0)
		for i in range(self.nGEN):
			self.t.next_iter()
			self.assertEqual(self.t.no_iter, i + 1, 'Error at %s. iteration' % (i + 1))

	def test_stopCond_evals_fine(self):
		x = full(self.D, 1.0)
		for i in range(self.nFES - 1):
			self.t.eval(x)
			self.assertFalse(self.t.is_stopping_cond())
		self.t.eval(x)
		self.assertTrue(self.t.is_stopping_cond())

	def test_stopCond_iters_fine(self):
		x = full(self.D, 1.0)
		for i in range(self.nGEN - 1):
			self.t.next_iter()
			self.assertFalse(self.t.is_stopping_cond())
		self.t.next_iter()
		self.assertTrue(self.t.is_stopping_cond())

	def test_stopCond_refValue_fine(self):
		x = full(self.D, 1.0)
		for i in range(self.nGEN - 5):
			self.assertFalse(self.t.is_stopping_cond())
			self.assertEqual(self.D, self.t.eval(x))
			self.t.next_iter()
		x = full(self.D, 0.0)
		self.assertEqual(0, self.t.eval(x))
		self.assertRaises(RefException, lambda: self.t.eval(x))

class ScaledTaskTestCase(TestCase):
	def setUp(self):
		self.D, self.nFES, self.nGEN = 10, 10, 10
		self.t = StoppingTask(D=self.D, nFES=self.nFES, nGEN=self.nGEN, benchmark=MyBenchmark())
		d1, d2 = self.t.lower + self.t.range / 2, self.t.upper - self.t.range * 0.2
		L, U = d1, d1 + d2
		self.tc = ScaledTask(self.t, L, U)

	def test_isFeasible_fine(self):
		x = full(self.D, 10)
		self.assertTrue(self.t.is_feasible(x))
		self.assertFalse(self.tc.is_feasible(x))
		x = full(self.D, -10)
		self.assertTrue(self.t.is_feasible(x))
		self.assertFalse(self.tc.is_feasible(x))
		x = rnd.uniform(0, 5, self.D)
		self.assertTrue(self.t.is_feasible(x))
		self.assertTrue(self.tc.is_feasible(x))
		x = full(self.D, -20)
		self.assertFalse(self.t.is_feasible(x))
		self.assertFalse(self.tc.is_feasible(x))
		x = full(self.D, 20)
		self.assertFalse(self.t.is_feasible(x))
		self.assertFalse(self.tc.is_feasible(x))

	def test_nextIter_fine(self):
		for i in range(self.nGEN):
			self.assertFalse(self.t.is_stopping_cond())
			self.assertFalse(self.tc.is_stopping_cond())
			self.t.next_iter()
		self.assertTrue(self.t.is_stopping_cond())
		self.assertTrue(self.tc.is_stopping_cond())

	def test_nextIter_two_fine(self):
		for i in range(int(self.nGEN / 2)):
			self.assertFalse(self.t.is_stopping_cond())
			self.assertFalse(self.tc.is_stopping_cond())
			self.tc.next_iter()
			self.t.next_iter()
		self.assertTrue(self.t.is_stopping_cond())
		self.assertTrue(self.tc.is_stopping_cond())

	def test_stopCondI(self):
		for i in range(int(self.nGEN / 2)):
			self.assertFalse(self.t.is_stopping_cond_next_iter())
			self.assertFalse(self.tc.is_stopping_cond_next_iter())
		self.assertTrue(self.t.is_stopping_cond_next_iter())
		self.assertTrue(self.tc.is_stopping_cond_next_iter())

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
			self.t.next_iter()
			self.tc.next_iter()
		self.assertEqual(inf, self.t.eval(x))
		self.assertEqual(inf, self.tc.eval(x))

	def test_nFES_count_fine(self):
		x = full(self.D, 0.0)
		for i in range(self.nFES // 2):
			try: self.t.eval(x)
			except Exception: pass
			self.assertEqual(self.t.no_eval, 2 * i + 1, 'Error at %s. evaluation' % (i + 1))
			try: self.tc.eval(x)
			except Exception: pass
			self.assertEqual(self.tc.no_eval, 2 * i + 2, 'Error at %s. evaluation' % (i + 1))

	def test_nGEN_count_fine(self):
		x = full(self.D, 0.0)
		for i in range(int((self.nFES - 1 if self.nFES % 2 > 0 else self.nFES - 2) / 2)):
			self.t.next_iter()
			self.assertEqual(self.t.no_iter, 2 * i + 1, 'Error at %s. iteration' % (i + 1))
			self.tc.next_iter()
			self.assertEqual(self.tc.no_iter, 2 * i + 2, 'Error at %s. iteration' % (i + 1))

	def test_stopCond_evals_fine(self):
		x = full(self.D, 0.0)
		for i in range(int((self.nFES - 1 if self.nFES % 2 > 0 else self.nFES - 2) / 2)):
			self.assertFalse(self.t.is_stopping_cond())
			self.t.eval(x)
			self.assertFalse(self.tc.is_stopping_cond())
			self.tc.eval(x)
		self.assertFalse(self.t.is_stopping_cond())
		self.t.eval(x)
		self.assertFalse(self.tc.is_stopping_cond())
		self.tc.eval(x)
		self.assertTrue(self.t.is_stopping_cond())
		self.assertTrue(self.tc.is_stopping_cond())

	def test_stopCond_iters_fine(self):
		x = full(self.D, 0.0)
		for i in range(int((self.nGEN - 1 if self.nGEN % 2 > 0 else self.nGEN - 2) / 2)):
			self.assertFalse(self.t.is_stopping_cond())
			self.t.next_iter()
			self.assertFalse(self.tc.is_stopping_cond())
			self.tc.next_iter()
		self.assertFalse(self.t.is_stopping_cond())
		self.t.next_iter()
		self.assertFalse(self.tc.is_stopping_cond())
		self.tc.next_iter()
		self.assertTrue(self.t.is_stopping_cond())
		self.assertTrue(self.tc.is_stopping_cond())

class TaskLogBestTestCase(TestCase):
	def setUp(self):
		self.D, self.nFES, self.nGEN = 10, 10, 10
		self.t = TaskLogBest(D=self.D, nFES=self.nFES, nGEN=self.nGEN, benchmark=MyBenchmark())

	def test_eval(self):
		self.t.eval(full(self.D, 2.))
		self.assertTrue(array_equal(full(self.D, 2.), self.t.xb))
		self.t.eval(full(self.D, -1.))
		self.assertTrue(array_equal(full(self.D, -1.), self.t.xb))
		self.t.eval(full(self.D, .0))
		self.assertTrue(array_equal(full(self.D, .0), self.t.xb))

class TaskPlotBestTestCase(TestCase):
	def setUp(self):
		self.D, self.nFES, self.nGEN = 10, 10, 10
		self.t = TaskPlotBest(D=self.D, nFES=self.nFES, nGEN=self.nGEN, benchmark=MyBenchmark())

	def test_eval(self):
		pass

class TaskCompositionTestCase(TestCase):
	def setUp(self):
		self.D, self.nFES, self.nGEN = 10, 10, 10
		self.t = TaskComposition(D=self.D, nFES=self.nFES, nGEN=self.nGEN, benchmark=MyBenchmark())

	def test_eval(self):
		self.assertEqual(self.t.eval(full(self.D, 0)), inf)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3