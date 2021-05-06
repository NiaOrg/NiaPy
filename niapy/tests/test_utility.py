# encoding=utf8
from unittest import TestCase

import numpy as np
from numpy.random import default_rng

from niapy.algorithms import Algorithm
from niapy.benchmarks import Benchmark
from niapy.util import full_array, repair


class FullArrayTestCase(TestCase):
    def test_a_float(self):
        A = full_array(25.25, 10)
        self.assertTrue(np.array_equal(A, np.full(10, 25.25)))

    def test_a_int(self):
        A = full_array(25, 10)
        self.assertTrue(np.array_equal(A, np.full(10, 25)))

    def test_a_float_list(self):
        a = [25.25 for i in range(10)]
        A = full_array(a, 10)
        self.assertTrue(np.array_equal(A, np.full(10, 25.25)))

    def test_a_int_list(self):
        a = [25 for i in range(10)]
        A = full_array(a, 10)
        self.assertTrue(np.array_equal(A, np.full(10, 25)))

    def test_a_float_array(self):
        a = np.asarray([25.25 for i in range(10)])
        A = full_array(a, 10)
        self.assertTrue(np.array_equal(A, np.full(10, 25.25)))

    def test_a_int_array(self):
        a = np.asarray([25 for i in range(10)])
        A = full_array(a, 10)
        self.assertTrue(np.array_equal(A, np.full(10, 25)))

    def test_a_float_list1(self):
        a = [25.25 + i for i in range(10)]
        A = full_array(a, 15)
        a.extend([25.25 + i for i in range(5)])
        self.assertTrue(np.array_equal(A, np.asarray(a)))

    def test_a_int_list1(self):
        a = [25 + i for i in range(10)]
        A = full_array(a, 15)
        a.extend([25 + i for i in range(5)])
        self.assertTrue(np.array_equal(A, np.asarray(a)))

    def test_a_float_array1(self):
        a = [25.25 + i for i in range(10)]
        A = full_array(np.asarray(a), 15)
        a.extend([25.25 + i for i in range(5)])
        self.assertTrue(np.array_equal(A, np.asarray(a)))

    def test_a_int_array1(self):
        a = [25 + i for i in range(10)]
        A = full_array(np.asarray(a), 15)
        a.extend([25 + i for i in range(5)])
        self.assertTrue(np.array_equal(A, np.asarray(a)))

    def test_a_float_list2(self):
        a = [25.25 + i for i in range(10)]
        A = full_array(a, 13)
        a.extend([25.25 + i for i in range(3)])
        self.assertTrue(np.array_equal(A, np.asarray(a)))

    def test_a_int_list2(self):
        a = [25 + i for i in range(10)]
        A = full_array(a, 13)
        a.extend([25 + i for i in range(3)])
        self.assertTrue(np.array_equal(A, np.asarray(a)))

    def test_a_float_array2(self):
        a = [25.25 + i for i in range(10)]
        A = full_array(np.asarray(a), 13)
        a.extend([25.25 + i for i in range(3)])
        self.assertTrue(np.array_equal(A, np.asarray(a)))

    def test_a_int_array2(self):
        a = [25 + i for i in range(10)]
        A = full_array(np.asarray(a), 13)
        a.extend([25 + i for i in range(3)])
        self.assertTrue(np.array_equal(A, np.asarray(a)))

    def test_a_float_list3(self):
        a = [25.25 + i for i in range(10)]
        A = full_array(a, 9)
        a.remove(34.25)
        self.assertTrue(np.array_equal(A, np.asarray(a)))

    def test_a_int_list3(self):
        a = [25 + i for i in range(10)]
        A = full_array(a, 9)
        a.remove(34)
        self.assertTrue(np.array_equal(A, np.asarray(a)))

    def test_a_float_array3(self):
        a = [25.25 + i for i in range(10)]
        A = full_array(np.asarray(a), 9)
        a.remove(34.25)
        self.assertTrue(np.array_equal(A, np.asarray(a)))

    def test_a_int_array3(self):
        a = [25 + i for i in range(10)]
        A = full_array(np.asarray(a), 9)
        a.remove(34)
        self.assertTrue(np.array_equal(A, np.asarray(a)))


class NoLimits:
    @classmethod
    def function(cls):
        def evaluate(D, x):
            return 0

        return evaluate


class MyBenchmark(Benchmark):
    def __init__(self):
        Benchmark.__init__(self, -10, 10)

    def function(self):
        def evaluate(D, x):
            return np.sum(x ** 2)

        return evaluate


class MyFakeAlgorithm:
    def __init__(self):
        pass


class MyCustomAlgorithm(Algorithm):
    pass


class LimitRepairTestCase(TestCase):
    def setUp(self):
        self.D = 10
        self.Upper, self.Lower = full_array(10, self.D), full_array(-10, self.D)
        self.met = repair.limit

    def generateIndividual(self, D, upper, lower):
        upp, low = full_array(upper, D), full_array(lower, D)
        return default_rng().uniform(low, upp, D)

    def test_limit_repair_good_solution(self):
        x = self.generateIndividual(self.D, self.Upper, self.Lower)
        x = self.met(x, self.Lower, self.Upper)
        self.assertFalse((x > self.Upper).any())
        self.assertFalse((x < self.Lower).any())

    def test_limit_repair_bad_upper_solution(self):
        x = self.generateIndividual(self.D, 12, 11)
        x = self.met(x, self.Lower, self.Upper)
        self.assertFalse((x > self.Upper).any())
        self.assertFalse((x < self.Lower).any())

    def test_limit_repair_bad_lower_soluiton(self):
        x = self.generateIndividual(self.D, -11, -12)
        x = self.met(x, self.Lower, self.Upper)
        self.assertFalse((x > self.Upper).any())
        self.assertFalse((x < self.Lower).any())

    def test_limit_repair_bad_upper_lower_soluiton(self):
        x = self.generateIndividual(self.D, 100, -100)
        x = self.met(x, self.Lower, self.Upper)
        self.assertFalse((x > self.Upper).any())
        self.assertFalse((x < self.Lower).any())


class LimitInverseRepairTestCase(LimitRepairTestCase):
    def setUp(self):
        LimitRepairTestCase.setUp(self)
        self.met = repair.limit_inverse


class WangRepairTestCase(LimitRepairTestCase):
    def setUp(self):
        LimitRepairTestCase.setUp(self)
        self.met = repair.wang


class RandRepairTestCase(LimitRepairTestCase):
    def setUp(self):
        LimitRepairTestCase.setUp(self)
        self.met = repair.rand


class ReflectRepairTestCase(LimitRepairTestCase):
    def setUp(self):
        LimitRepairTestCase.setUp(self)
        self.met = repair.reflect

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
