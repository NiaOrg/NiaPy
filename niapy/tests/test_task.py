# encoding=utf8
from unittest import TestCase

import numpy as np
from numpy.random import default_rng

from niapy.problems import Problem
from niapy.task import Task
from niapy.util.array import full_array


class MyProblem(Problem):
    def __init__(self, dimension=20):
        super().__init__(dimension, -10, 10)

    def _evaluate(self, x):
        return np.sum(x ** 2)


class TaskTestCase(TestCase):
    r"""Test case for testing the Task class.

    Date:
        April 2019

    Author:
        Klemen Berkovič

    See Also:
        * :class:`niapy.util.Task`

    """

    def setUp(self):
        self.D, self.nFES, self.nGEN = 10, 10, 10
        self.Lower, self.Upper = [2, 1, 1], [10, 10, 5]
        self.task = Task(dimension=self.D, lower=self.Lower, upper=self.Upper, problem='sphere', max_evals=self.nFES, max_iters=self.nGEN, cutoff_value=0.0)

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

    def test_max_iters(self):
        self.assertEqual(self.nGEN, self.task.max_iters)

    def test_max_evals(self):
        self.assertEqual(self.nFES, self.task.max_evals)

    def test_is_feasible(self):
        x = np.full(self.D, 2)
        self.assertTrue(self.task.is_feasible(x))
        x = np.full(self.D, 3)
        self.assertTrue(self.task.is_feasible(x))
        x = default_rng().uniform(self.task.lower, self.task.upper, self.D)
        self.assertTrue(self.task.is_feasible(x))
        x = np.full(self.D, -20)
        self.assertFalse(self.task.is_feasible(x))
        x = np.full(self.D, 20)
        self.assertFalse(self.task.is_feasible(x))

    def test_next_iter(self):
        for i in range(self.nGEN):
            self.assertFalse(self.task.stopping_condition())
            self.task.next_iter()
        self.assertTrue(self.task.stopping_condition())

    def test_stop_cond_iter(self):
        for i in range(self.nGEN):
            self.assertFalse(self.task.stopping_condition_iter(), msg='Error at %s iteration!!!' % i)
        self.assertTrue(self.task.stopping_condition_iter())

    def test_eval(self):
        x = np.ones(self.D)
        for i in range(self.nFES):
            self.assertAlmostEqual(self.task.eval(x), self.D, msg='Error at %s iteration!!!' % i)
        self.assertTrue(self.task.stopping_condition())

    def test_eval_over_max_evals(self):
        x = np.ones(self.D)
        for i in range(self.nFES):
            self.task.eval(x)
        self.assertEqual(np.inf, self.task.eval(x))
        self.assertTrue(self.task.stopping_condition())

    def test_eval_over_max_iters(self):
        x = np.ones(self.D)
        for i in range(self.nGEN):
            self.task.next_iter()
        self.assertEqual(np.inf, self.task.eval(x))
        self.assertTrue(self.task.stopping_condition())

    def test_evals_count(self):
        x = np.ones(self.D)
        for i in range(self.nFES):
            self.task.eval(x)
            self.assertEqual(self.task.evals, i + 1, 'Error at %s. evaluation' % (i + 1))

    def test_iters_count(self):
        for i in range(self.nGEN):
            self.task.next_iter()
            self.assertEqual(self.task.iters, i + 1, 'Error at %s. iteration' % (i + 1))

    def test_stop_cond_evals(self):
        x = np.ones(self.D)
        for i in range(self.nFES - 1):
            self.task.eval(x)
            self.assertFalse(self.task.stopping_condition())
        self.task.eval(x)
        self.assertTrue(self.task.stopping_condition())

    def test_stop_cond_iters(self):
        for i in range(self.nGEN - 1):
            self.task.next_iter()
            self.assertFalse(self.task.stopping_condition())
        self.task.next_iter()
        self.assertTrue(self.task.stopping_condition())

    def test_stop_cond_cutoff_value(self):
        x = np.ones(self.D)
        for i in range(self.nGEN - 5):
            self.assertFalse(self.task.stopping_condition())
            self.assertEqual(self.D, self.task.eval(x))
            self.task.next_iter()
        x = np.zeros(self.D)
        self.assertEqual(0, self.task.eval(x))
        self.assertTrue(self.task.stopping_condition())
        self.assertEqual(self.nGEN - 5, self.task.iters)

    def test_print_conv_one(self):
        r1, r2 = [], []
        for i in range(self.nFES):
            x = np.full(self.D, 10 - i)
            r1.append(i + 1), r2.append(self.task.eval(x))
        t_r1, t_r2 = self.task.return_conv()
        self.assertTrue(np.array_equal(r1, t_r1))
        self.assertTrue(np.array_equal(r2, t_r2))

    def test_print_conv_two(self):
        r1, r2 = [], []
        for i in range(self.nFES):
            x = np.full(self.D, 10 - i if i not in (3, 4, 5) else 4)
            r1.append(i + 1), r2.append(self.task.eval(x))
        t_r1, t_r2 = self.task.return_conv()
        self.assertTrue(np.array_equal(r2, t_r2))
        self.assertTrue(np.array_equal(r1, t_r1))

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
