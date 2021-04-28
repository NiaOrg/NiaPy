# encoding=utf8
from unittest import TestCase

import numpy as np
from numpy.random import default_rng

from niapy.benchmarks import Benchmark
from niapy.util import full_array, FesException, GenException, RefException
from niapy.task import StoppingTask, ThrowingTask

class MyBenchmark(Benchmark):
	def __init__(self):
		super().__init__(-10, 10)

	def function(self):
		def evaluate(D, x): return sum(x ** 2)
		return evaluate

class StoppingTaskBaseTestCase(TestCase):
	r"""Test case for testing `Task`, `StoppingTask` and `CountingTask` classes.

	Date:
		April 2019

	Author:
		Klemen Berkovič

	See Also:
		* :class:`niapy.util.Task`
		* :class:`niapy.util.CountingTask`
		* :class:`niapy.util.StoppingTask`
	"""
	def setUp(self):
		self.D = 6
		self.Lower, self.Upper = [2, 1, 1], [10, 10, 2]
		self.task = StoppingTask(Lower=self.Lower, Upper=self.Upper, D=self.D)

	def test_dim_ok(self):
		self.assertEqual(self.D, self.task.D)
		self.assertEqual(self.D, self.task.dim())

	def test_lower(self):
		self.assertTrue(np.array_equal(full_array(self.Lower, self.D), self.task.Lower))
		self.assertTrue(np.array_equal(full_array(self.Lower, self.D), self.task.bcLower()))

	def test_upper(self):
		self.assertTrue(np.array_equal(full_array(self.Upper, self.D), self.task.Upper))
		self.assertTrue(np.array_equal(full_array(self.Upper, self.D), self.task.bcUpper()))

	def test_range(self):
		self.assertTrue(np.array_equal(full_array(self.Upper, self.D) - full_array(self.Lower, self.D), self.task.bRange))
		self.assertTrue(np.array_equal(full_array(self.Upper, self.D) - full_array(self.Lower, self.D), self.task.bcRange()))

	def test_ngens(self):
		self.assertEqual(np.inf, self.task.nGEN)

	def test_nfess(self):
		self.assertEqual(np.inf, self.task.nFES)

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
		self.assertFalse(self.task.isFeasible(full_array([1, 2, 3], self.D)))

class StoppingTaskTestCase(TestCase):
	r"""Test case for testing `Task`, `StoppingTask` and `CountingTask` classes.

	Date:
		April 2019

	Author:
		Klemen Berkovič

	See Also:
		* :class:`niapy.util.Task`
		* :class:`niapy.util.CountingTask`
		* :class:`niapy.util.StoppingTask`
	"""
	def setUp(self):
		self.D, self.nFES, self.nGEN = 10, 10, 10
		self.t = StoppingTask(D=self.D, nFES=self.nFES, nGEN=self.nGEN, refValue=1, benchmark=MyBenchmark())

	def test_isFeasible_fine(self):
		x = np.full(self.D, 10)
		self.assertTrue(self.t.isFeasible(x))
		x = np.full(self.D, -10)
		self.assertTrue(self.t.isFeasible(x))
		x = default_rng().uniform(-10, 10, self.D)
		self.assertTrue(self.t.isFeasible(x))
		x = np.full(self.D, -20)
		self.assertFalse(self.t.isFeasible(x))
		x = np.full(self.D, 20)
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
		x = np.ones(self.D)
		for i in range(self.nFES): self.assertAlmostEqual(self.t.eval(x), self.D, msg='Error at %s iteration!!!' % (i))
		self.assertTrue(self.t.stopCond())

	def test_eval_over_nFES_fine(self):
		x = np.ones(self.D)
		for i in range(self.nFES): self.t.eval(x)
		self.assertEqual(np.inf, self.t.eval(x))
		self.assertTrue(self.t.stopCond())

	def test_eval_over_nGEN_fine(self):
		x = np.ones(self.D)
		for i in range(self.nGEN): self.t.nextIter()
		self.assertEqual(np.inf, self.t.eval(x))
		self.assertTrue(self.t.stopCond())

	def test_nFES_count_fine(self):
		x = np.ones(self.D)
		for i in range(self.nFES):
			self.t.eval(x)
			self.assertEqual(self.t.Evals, i + 1, 'Error at %s. evaluation' % (i + 1))

	def test_nGEN_count_fine(self):
		x = np.ones(self.D)
		for i in range(self.nGEN):
			self.t.nextIter()
			self.assertEqual(self.t.Iters, i + 1, 'Error at %s. iteration' % (i + 1))

	def test_stopCond_evals_fine(self):
		x = np.ones(self.D)
		for i in range(self.nFES - 1):
			self.t.eval(x)
			self.assertFalse(self.t.stopCond())
		self.t.eval(x)
		self.assertTrue(self.t.stopCond())

	def test_stopCond_iters_fine(self):
		x = np.ones(self.D)
		for i in range(self.nGEN - 1):
			self.t.nextIter()
			self.assertFalse(self.t.stopCond())
		self.t.nextIter()
		self.assertTrue(self.t.stopCond())

	def test_stopCond_refValue_fine(self):
		x = np.ones(self.D)
		for i in range(self.nGEN - 5):
			self.assertFalse(self.t.stopCond())
			self.assertEqual(self.D, self.t.eval(x))
			self.t.nextIter()
		x = np.zeros(self.D)
		self.assertEqual(0, self.t.eval(x))
		self.assertTrue(self.t.stopCond())
		self.assertEqual(self.nGEN - 5, self.t.Iters)

	def test_print_conv_one_fine(self):
		r1, r2 = [], []
		for i in range(self.nFES):
			x = np.full(self.D, 10 - i)
			r1.append(i + 1), r2.append(self.t.eval(x))
		t_r1, t_r2 = self.t.return_conv()
		self.assertTrue(np.array_equal(r1, t_r1))
		self.assertTrue(np.array_equal(r2, t_r2))

	def test_print_conv_two_fine(self):
		r1, r2 = [], []
		for i in range(self.nFES):
			x = np.full(self.D, 10 - i if i not in (3, 4, 5) else 4)
			r1.append(i + 1), r2.append(self.t.eval(x))
		t_r1, t_r2 = self.t.return_conv()
		self.assertTrue(np.array_equal(r2, t_r2))
		self.assertTrue(np.array_equal(r1, t_r1))

class ThrowingTaskTestCase(TestCase):
	r"""Test case for testing `ThrowingTask` class.

	Date:
		April 2019

	Author:
		Klemen Berkovič

	See Also:
		* :class:`niapy.util.ThrowingTask`
	"""
	def setUp(self):
		self.D, self.nFES, self.nGEN = 10, 10, 10
		self.t = ThrowingTask(D=self.D, nFES=self.nFES, nGEN=self.nGEN, refValue=0, benchmark=MyBenchmark())

	def test_isFeasible_fine(self):
		x = np.full(self.D, 10)
		self.assertTrue(self.t.isFeasible(x))
		x = np.full(self.D, -10)
		self.assertTrue(self.t.isFeasible(x))
		x = default_rng().uniform(-10, 10, self.D)
		self.assertTrue(self.t.isFeasible(x))
		x = np.full(self.D, -20)
		self.assertFalse(self.t.isFeasible(x))
		x = np.full(self.D, 20)
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
		x = np.ones(self.D)
		for i in range(self.nFES): self.assertAlmostEqual(self.t.eval(x), self.D, msg='Error at %s iteration!!!' % (i))
		self.assertRaises(FesException, lambda: self.t.eval(x))

	def test_eval_over_nFES_fine(self):
		x = np.ones(self.D)
		for i in range(self.nFES):
			self.t.eval(x)
		self.assertRaises(FesException, lambda: self.t.eval(x))

	def test_eval_over_nGEN_fine(self):
		x = np.ones(self.D)
		for i in range(self.nGEN): self.t.nextIter()
		self.assertRaises(GenException, lambda: self.t.eval(x))

	def test_nFES_count_fine(self):
		x = np.ones(self.D)
		for i in range(self.nFES):
			self.t.eval(x)
			self.assertEqual(self.t.Evals, i + 1, 'Error at %s. evaluation' % (i + 1))

	def test_nGEN_count_fine(self):
		x = np.ones(self.D)
		for i in range(self.nGEN):
			self.t.nextIter()
			self.assertEqual(self.t.Iters, i + 1, 'Error at %s. iteration' % (i + 1))

	def test_stopCond_evals_fine(self):
		x = np.ones(self.D)
		for i in range(self.nFES - 1):
			self.t.eval(x)
			self.assertFalse(self.t.stopCond())
		self.t.eval(x)
		self.assertTrue(self.t.stopCond())

	def test_stopCond_iters_fine(self):
		x = np.ones(self.D)
		for i in range(self.nGEN - 1):
			self.t.nextIter()
			self.assertFalse(self.t.stopCond())
		self.t.nextIter()
		self.assertTrue(self.t.stopCond())

	def test_stopCond_refValue_fine(self):
		x = np.ones(self.D)
		for i in range(self.nGEN - 5):
			self.assertFalse(self.t.stopCond())
			self.assertEqual(self.D, self.t.eval(x))
			self.t.nextIter()
		x = np.zeros(self.D)
		self.assertEqual(0, self.t.eval(x))
		self.assertRaises(RefException, lambda: self.t.eval(x))

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3