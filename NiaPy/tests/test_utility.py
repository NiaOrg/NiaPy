# encoding=utf8

from unittest import TestCase

from numpy import full, random as rnd, inf, sum, array_equal, asarray

from NiaPy.util import fullArray, limit_repair, limitInversRepair, wangRepair, randRepair, reflectRepair, \
    FesException, GenException, RefException
from NiaPy.benchmarks.utility import Utility
from NiaPy.benchmarks import Benchmark
from NiaPy.task import StoppingTask, ThrowingTask


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


class NoLimits(Benchmark):
    @classmethod
    def function(cls):
        def evaluate(D, x): return 0
        return evaluate


class MyBenchmark(Benchmark):
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
        for i in range(self.nGEN):
            self.assertFalse(self.t.stopCondI(), msg='Error at %s iteration!!!' % (i))
        self.assertTrue(self.t.stopCondI())

    def test_eval_fine(self):
        x = full(self.D, 1.0)
        for i in range(self.nFES):
            self.assertAlmostEqual(self.t.eval(x), self.D, msg='Error at %s iteration!!!' % (i))
        self.assertTrue(self.t.stopCond())

    def test_eval_over_nFES_fine(self):
        x = full(self.D, 1.0)
        for i in range(self.nFES):
            self.t.eval(x)
        self.assertEqual(inf, self.t.eval(x))
        self.assertTrue(self.t.stopCond())

    def test_eval_over_nGEN_fine(self):
        x = full(self.D, 1.0)
        for i in range(self.nGEN):
            self.t.nextIter()
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
        for i in range(self.nGEN):
            self.assertFalse(self.t.stopCondI())
        self.assertTrue(self.t.stopCondI())

    def test_eval_fine(self):
        x = full(self.D, 1.0)
        for i in range(self.nFES):
            self.assertAlmostEqual(self.t.eval(x), self.D, msg='Error at %s iteration!!!' % (i))
        self.assertRaises(FesException, lambda: self.t.eval(x))

    def test_eval_over_nFES_fine(self):
        x = full(self.D, 1.0)
        for i in range(self.nFES):
            self.t.eval(x)
        self.assertRaises(FesException, lambda: self.t.eval(x))

    def test_eval_over_nGEN_fine(self):
        x = full(self.D, 1.0)
        for i in range(self.nGEN):
            self.t.nextIter()
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
