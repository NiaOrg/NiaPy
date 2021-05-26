# encoding=utf8
from unittest import TestCase

import numpy as np
from numpy.random import default_rng

from niapy.util import full_array, repair


class FullArrayTestCase(TestCase):
    def test_a_float(self):
        arr = full_array(25.25, 10)
        self.assertTrue(np.array_equal(arr, np.full(10, 25.25)))

    def test_a_int(self):
        arr = full_array(25, 10)
        self.assertTrue(np.array_equal(arr, np.full(10, 25)))

    def test_a_float_list(self):
        a = [25.25] * 10
        arr = full_array(a, 10)
        self.assertTrue(np.array_equal(arr, np.full(10, 25.25)))

    def test_a_int_list(self):
        a = [25] * 10
        arr = full_array(a, 10)
        self.assertTrue(np.array_equal(arr, np.full(10, 25)))

    def test_a_float_array(self):
        a = np.asarray([25.25] * 10)
        arr = full_array(a, 10)
        self.assertTrue(np.array_equal(arr, np.full(10, 25.25)))

    def test_a_int_array(self):
        a = np.asarray([25] * 10)
        arr = full_array(a, 10)
        self.assertTrue(np.array_equal(arr, np.full(10, 25)))

    def test_a_float_list1(self):
        a = [25.25 + i for i in range(10)]
        arr = full_array(a, 15)
        a.extend([25.25 + i for i in range(5)])
        self.assertTrue(np.array_equal(arr, np.asarray(a)))

    def test_a_int_list1(self):
        a = [25 + i for i in range(10)]
        arr = full_array(a, 15)
        a.extend([25 + i for i in range(5)])
        self.assertTrue(np.array_equal(arr, np.asarray(a)))

    def test_a_float_array1(self):
        a = [25.25 + i for i in range(10)]
        arr = full_array(np.asarray(a), 15)
        a.extend([25.25 + i for i in range(5)])
        self.assertTrue(np.array_equal(arr, np.asarray(a)))

    def test_a_int_array1(self):
        a = [25 + i for i in range(10)]
        arr = full_array(np.asarray(a), 15)
        a.extend([25 + i for i in range(5)])
        self.assertTrue(np.array_equal(arr, np.asarray(a)))

    def test_a_float_list2(self):
        a = [25.25 + i for i in range(10)]
        arr = full_array(a, 13)
        a.extend([25.25 + i for i in range(3)])
        self.assertTrue(np.array_equal(arr, np.asarray(a)))

    def test_a_int_list2(self):
        a = [25 + i for i in range(10)]
        arr = full_array(a, 13)
        a.extend([25 + i for i in range(3)])
        self.assertTrue(np.array_equal(arr, np.asarray(a)))

    def test_a_float_array2(self):
        a = [25.25 + i for i in range(10)]
        arr = full_array(np.asarray(a), 13)
        a.extend([25.25 + i for i in range(3)])
        self.assertTrue(np.array_equal(arr, np.asarray(a)))

    def test_a_int_array2(self):
        a = [25 + i for i in range(10)]
        arr = full_array(np.asarray(a), 13)
        a.extend([25 + i for i in range(3)])
        self.assertTrue(np.array_equal(arr, np.asarray(a)))

    def test_a_float_list3(self):
        a = [25.25 + i for i in range(10)]
        arr = full_array(a, 9)
        a.remove(34.25)
        self.assertTrue(np.array_equal(arr, np.asarray(a)))

    def test_a_int_list3(self):
        a = [25 + i for i in range(10)]
        arr = full_array(a, 9)
        a.remove(34)
        self.assertTrue(np.array_equal(arr, np.asarray(a)))

    def test_a_float_array3(self):
        a = [25.25 + i for i in range(10)]
        arr = full_array(np.asarray(a), 9)
        a.remove(34.25)
        self.assertTrue(np.array_equal(arr, np.asarray(a)))

    def test_a_int_array3(self):
        a = [25 + i for i in range(10)]
        arr = full_array(np.asarray(a), 9)
        a.remove(34)
        self.assertTrue(np.array_equal(arr, np.asarray(a)))


def generate_individual(dim, upper, lower):
    upp, low = full_array(upper, dim), full_array(lower, dim)
    return default_rng().uniform(low, upp, dim)


class LimitRepairTestCase(TestCase):
    def setUp(self):
        self.D = 10
        self.Upper, self.Lower = full_array(10, self.D), full_array(-10, self.D)
        self.met = repair.limit

    def test_limit_repair_good_solution(self):
        x = generate_individual(self.D, self.Upper, self.Lower)
        x = self.met(x, self.Lower, self.Upper)
        self.assertFalse((x > self.Upper).any())
        self.assertFalse((x < self.Lower).any())

    def test_limit_repair_bad_upper_solution(self):
        x = generate_individual(self.D, 12, 11)
        x = self.met(x, self.Lower, self.Upper)
        self.assertFalse((x > self.Upper).any())
        self.assertFalse((x < self.Lower).any())

    def test_limit_repair_bad_lower_solution(self):
        x = generate_individual(self.D, -11, -12)
        x = self.met(x, self.Lower, self.Upper)
        self.assertFalse((x > self.Upper).any())
        self.assertFalse((x < self.Lower).any())

    def test_limit_repair_bad_upper_lower_solution(self):
        x = generate_individual(self.D, 100, -100)
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
