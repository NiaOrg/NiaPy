# encoding=utf8
from unittest import TestCase

import numpy as np
from numpy.random import default_rng

from niapy.benchmarks import Benchmark
from niapy.task import StoppingTask, ThrowingTask
from niapy.util import full_array, FesException, GenException, RefException


class MyBenchmark(Benchmark):
    def __init__(self):
        super().__init__(-10, 10)

    def function(self):
        def evaluate(D, x):
            return sum(x ** 2)

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
        self.task = StoppingTask(dimension=self.D, lower=self.Lower, upper=self.Upper)

    def test_dim_ok(self):
        self.assertEqual(self.D, self.task.dimension)
        self.assertEqual(self.D, self.task.dimension)

    def test_lower(self):
        self.assertTrue(np.array_equal(full_array(self.Lower, self.D), self.task.lower))
        self.assertTrue(np.array_equal(full_array(self.Lower, self.D), self.task.lower))

    def test_upper(self):
        self.assertTrue(np.array_equal(full_array(self.Upper, self.D), self.task.upper))
        self.assertTrue(np.array_equal(full_array(self.Upper, self.D), self.task.upper))

    def test_range(self):
        self.assertTrue(
            np.array_equal(full_array(self.Upper, self.D) - full_array(self.Lower, self.D), self.task.range))
        self.assertTrue(
            np.array_equal(full_array(self.Upper, self.D) - full_array(self.Lower, self.D), self.task.range))

    def test_ngens(self):
        self.assertEqual(np.inf, self.task.max_iters)

    def test_nfess(self):
        self.assertEqual(np.inf, self.task.max_evals)

    def test_stop_cond(self):
        self.assertFalse(self.task.stopping_condition())

    def test_stop_condi(self):
        self.assertFalse(self.task.stopping_condition_iter())

    def test_eval(self):
        self.assertRaises(AttributeError, lambda: self.task.eval([]))

    def test_evals(self):
        self.assertEqual(0, self.task.evals)

    def test_iters(self):
        self.assertEqual(0, self.task.iters)

    def test_next_iter(self):
        self.assertEqual(None, self.task.next_iter())

    def test_is_feasible(self):
        self.assertFalse(self.task.is_feasible(full_array([1, 2, 3], self.D)))


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
        self.t = StoppingTask(max_evals=self.nFES, max_iters=self.nGEN, cutoff_value=1, dimension=self.D,
                              benchmark=MyBenchmark())

    def test_isFeasible(self):
        x = np.full(self.D, 10)
        self.assertTrue(self.t.is_feasible(x))
        x = np.full(self.D, -10)
        self.assertTrue(self.t.is_feasible(x))
        x = default_rng().uniform(-10, 10, self.D)
        self.assertTrue(self.t.is_feasible(x))
        x = np.full(self.D, -20)
        self.assertFalse(self.t.is_feasible(x))
        x = np.full(self.D, 20)
        self.assertFalse(self.t.is_feasible(x))

    def test_nextIter(self):
        for i in range(self.nGEN):
            self.assertFalse(self.t.stopping_condition())
            self.t.next_iter()
        self.assertTrue(self.t.stopping_condition())

    def test_stopCondI(self):
        for i in range(self.nGEN):
            self.assertFalse(self.t.stopping_condition_iter(), msg='Error at %s iteration!!!' % i)
        self.assertTrue(self.t.stopping_condition_iter())

    def test_eval(self):
        x = np.ones(self.D)
        for i in range(self.nFES):
            self.assertAlmostEqual(self.t.eval(x), self.D, msg='Error at %s iteration!!!' % i)
        self.assertTrue(self.t.stopping_condition())

    def test_eval_over_nFES(self):
        x = np.ones(self.D)
        for i in range(self.nFES):
            self.t.eval(x)
        self.assertEqual(np.inf, self.t.eval(x))
        self.assertTrue(self.t.stopping_condition())

    def test_eval_over_nGEN(self):
        x = np.ones(self.D)
        for i in range(self.nGEN):
            self.t.next_iter()
        self.assertEqual(np.inf, self.t.eval(x))
        self.assertTrue(self.t.stopping_condition())

    def test_nFES_count(self):
        x = np.ones(self.D)
        for i in range(self.nFES):
            self.t.eval(x)
            self.assertEqual(self.t.evals, i + 1, 'Error at %s. evaluation' % (i + 1))

    def test_nGEN_count(self):
        x = np.ones(self.D)
        for i in range(self.nGEN):
            self.t.next_iter()
            self.assertEqual(self.t.iters, i + 1, 'Error at %s. iteration' % (i + 1))

    def test_stopCond_evals(self):
        x = np.ones(self.D)
        for i in range(self.nFES - 1):
            self.t.eval(x)
            self.assertFalse(self.t.stopping_condition())
        self.t.eval(x)
        self.assertTrue(self.t.stopping_condition())

    def test_stopCond_iters(self):
        x = np.ones(self.D)
        for i in range(self.nGEN - 1):
            self.t.next_iter()
            self.assertFalse(self.t.stopping_condition())
        self.t.next_iter()
        self.assertTrue(self.t.stopping_condition())

    def test_stopCond_refValue(self):
        x = np.ones(self.D)
        for i in range(self.nGEN - 5):
            self.assertFalse(self.t.stopping_condition())
            self.assertEqual(self.D, self.t.eval(x))
            self.t.next_iter()
        x = np.zeros(self.D)
        self.assertEqual(0, self.t.eval(x))
        self.assertTrue(self.t.stopping_condition())
        self.assertEqual(self.nGEN - 5, self.t.iters)

    def test_print_conv_one(self):
        r1, r2 = [], []
        for i in range(self.nFES):
            x = np.full(self.D, 10 - i)
            r1.append(i + 1), r2.append(self.t.eval(x))
        t_r1, t_r2 = self.t.return_conv()
        self.assertTrue(np.array_equal(r1, t_r1))
        self.assertTrue(np.array_equal(r2, t_r2))

    def test_print_conv_two(self):
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
        self.t = ThrowingTask(dimension=self.D, max_evals=self.nFES, max_iters=self.nGEN, cutoff_value=0,
                              benchmark=MyBenchmark())

    def test_isFeasible(self):
        x = np.full(self.D, 10)
        self.assertTrue(self.t.is_feasible(x))
        x = np.full(self.D, -10)
        self.assertTrue(self.t.is_feasible(x))
        x = default_rng().uniform(-10, 10, self.D)
        self.assertTrue(self.t.is_feasible(x))
        x = np.full(self.D, -20)
        self.assertFalse(self.t.is_feasible(x))
        x = np.full(self.D, 20)
        self.assertFalse(self.t.is_feasible(x))

    def test_nextIter(self):
        for i in range(self.nGEN):
            self.assertFalse(self.t.stopping_condition())
            self.t.next_iter()
        self.assertTrue(self.t.stopping_condition())

    def test_stopCondI(self):
        for i in range(self.nGEN):
            self.assertFalse(self.t.stopping_condition_iter())
        self.assertTrue(self.t.stopping_condition_iter())

    def test_eval(self):
        x = np.ones(self.D)
        for i in range(self.nFES):
            self.assertAlmostEqual(self.t.eval(x), self.D, msg='Error at %s iteration!!!' % i)
        self.assertRaises(FesException, lambda: self.t.eval(x))

    def test_eval_over_nFES(self):
        x = np.ones(self.D)
        for i in range(self.nFES):
            self.t.eval(x)
        self.assertRaises(FesException, lambda: self.t.eval(x))

    def test_eval_over_nGEN(self):
        x = np.ones(self.D)
        for i in range(self.nGEN):
            self.t.next_iter()
        self.assertRaises(GenException, lambda: self.t.eval(x))

    def test_nFES_count(self):
        x = np.ones(self.D)
        for i in range(self.nFES):
            self.t.eval(x)
            self.assertEqual(self.t.evals, i + 1, 'Error at %s. evaluation' % (i + 1))

    def test_nGEN_count(self):
        x = np.ones(self.D)
        for i in range(self.nGEN):
            self.t.next_iter()
            self.assertEqual(self.t.iters, i + 1, 'Error at %s. iteration' % (i + 1))

    def test_stopCond_evals(self):
        x = np.ones(self.D)
        for i in range(self.nFES - 1):
            self.t.eval(x)
            self.assertFalse(self.t.stopping_condition())
        self.t.eval(x)
        self.assertTrue(self.t.stopping_condition())

    def test_stopCond_iters(self):
        x = np.ones(self.D)
        for i in range(self.nGEN - 1):
            self.t.next_iter()
            self.assertFalse(self.t.stopping_condition())
        self.t.next_iter()
        self.assertTrue(self.t.stopping_condition())

    def test_stopCond_refValue(self):
        x = np.ones(self.D)
        for i in range(self.nGEN - 5):
            self.assertFalse(self.t.stopping_condition())
            self.assertEqual(self.D, self.t.eval(x))
            self.t.next_iter()
        x = np.zeros(self.D)
        self.assertEqual(0, self.t.eval(x))
        self.assertRaises(RefException, lambda: self.t.eval(x))

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
