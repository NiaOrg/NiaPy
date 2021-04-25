# encoding=utf8
from unittest import TestCase

import numpy as np

from NiaPy.util import full_array, repair, FesException, GenException  # TimeException, RefException
from NiaPy.task import Utility, StoppingTask, ThrowingTask
from NiaPy.benchmarks import Benchmark
from NiaPy.algorithms import Algorithm, AlgorithmUtility
from NiaPy.algorithms.basic import GreyWolfOptimizer

class FullArrayTestCase(TestCase):
	def test_a_float_fine(self):
		A = full_array(25.25, 10)
		self.assertTrue(np.array_equal(A, np.full(10, 25.25)))

	def test_a_int_fine(self):
		A = full_array(25, 10)
		self.assertTrue(np.array_equal(A, np.full(10, 25)))

	def test_a_float_list_fine(self):
		a = [25.25 for i in range(10)]
		A = full_array(a, 10)
		self.assertTrue(np.array_equal(A, np.full(10, 25.25)))

	def test_a_int_list_fine(self):
		a = [25 for i in range(10)]
		A = full_array(a, 10)
		self.assertTrue(np.array_equal(A, np.full(10, 25)))

	def test_a_float_array_fine(self):
		a = np.asarray([25.25 for i in range(10)])
		A = full_array(a, 10)
		self.assertTrue(np.array_equal(A, np.full(10, 25.25)))

	def test_a_int_array_fine(self):
		a = np.asarray([25 for i in range(10)])
		A = full_array(a, 10)
		self.assertTrue(np.array_equal(A, np.full(10, 25)))

	def test_a_float_list1_fine(self):
		a = [25.25 + i for i in range(10)]
		A = full_array(a, 15)
		a.extend([25.25 + i for i in range(5)])
		self.assertTrue(np.array_equal(A, np.asarray(a)))

	def test_a_int_list1_fine(self):
		a = [25 + i for i in range(10)]
		A = full_array(a, 15)
		a.extend([25 + i for i in range(5)])
		self.assertTrue(np.array_equal(A, np.asarray(a)))

	def test_a_float_array1_fine(self):
		a = [25.25 + i for i in range(10)]
		A = full_array(np.asarray(a), 15)
		a.extend([25.25 + i for i in range(5)])
		self.assertTrue(np.array_equal(A, np.asarray(a)))

	def test_a_int_array1_fine(self):
		a = [25 + i for i in range(10)]
		A = full_array(np.asarray(a), 15)
		a.extend([25 + i for i in range(5)])
		self.assertTrue(np.array_equal(A, np.asarray(a)))

	def test_a_float_list2_fine(self):
		a = [25.25 + i for i in range(10)]
		A = full_array(a, 13)
		a.extend([25.25 + i for i in range(3)])
		self.assertTrue(np.array_equal(A, np.asarray(a)))

	def test_a_int_list2_fine(self):
		a = [25 + i for i in range(10)]
		A = full_array(a, 13)
		a.extend([25 + i for i in range(3)])
		self.assertTrue(np.array_equal(A, np.asarray(a)))

	def test_a_float_array2_fine(self):
		a = [25.25 + i for i in range(10)]
		A = full_array(np.asarray(a), 13)
		a.extend([25.25 + i for i in range(3)])
		self.assertTrue(np.array_equal(A, np.asarray(a)))

	def test_a_int_array2_fine(self):
		a = [25 + i for i in range(10)]
		A = full_array(np.asarray(a), 13)
		a.extend([25 + i for i in range(3)])
		self.assertTrue(np.array_equal(A, np.asarray(a)))

	def test_a_float_list3_fine(self):
		a = [25.25 + i for i in range(10)]
		A = full_array(a, 9)
		a.remove(34.25)
		self.assertTrue(np.array_equal(A, np.asarray(a)))

	def test_a_int_list3_fine(self):
		a = [25 + i for i in range(10)]
		A = full_array(a, 9)
		a.remove(34)
		self.assertTrue(np.array_equal(A, np.asarray(a)))

	def test_a_float_array3_fine(self):
		a = [25.25 + i for i in range(10)]
		A = full_array(np.asarray(a), 9)
		a.remove(34.25)
		self.assertTrue(np.array_equal(A, np.asarray(a)))

	def test_a_int_array3_fine(self):
		a = [25 + i for i in range(10)]
		A = full_array(np.asarray(a), 9)
		a.remove(34)
		self.assertTrue(np.array_equal(A, np.asarray(a)))

class NoLimits:
	@classmethod
	def function(cls):
		def evaluate(D, x): return 0
		return evaluate

class MyBenchmark(Benchmark):
	def __init__(self):
		Benchmark.__init__(self, -10, 10)

	def function(self):
		def evaluate(D, x): return np.sum(x ** 2)
		return evaluate

class UtilityTestCase(TestCase):
	def setUp(self):
		self.u = Utility()

	def test_get_bad_benchmark_fine(self):
		self.assertRaises(TypeError, lambda: self.u.get_benchmark('hihihihihihihihihi'))
		self.assertRaises(TypeError, lambda: self.u.get_benchmark(MyBenchmark))
		self.assertRaises(TypeError, lambda: self.u.get_benchmark(NoLimits))

class MyFakeAlgorithm:
	def __init__(self):
		pass

class MyCustomAlgorithm(Algorithm):
	pass

class AlgorithmUtilityTestCase(TestCase):
	def setUp(self):
		self.algorithm_utility = AlgorithmUtility()

	def test_get_bad_algorithm_fine(self):
		self.assertRaises(TypeError, lambda: self.algorithm_utility.get_algorithm('hihihihihihihihihi'))
		self.assertRaises(TypeError, lambda: self.algorithm_utility.get_algorithm(MyFakeAlgorithm()))

	def test_get_algorithm_fine(self):
		algorithm = MyCustomAlgorithm()
		gwo = GreyWolfOptimizer()
		self.assertEqual(algorithm, self.algorithm_utility.get_algorithm(algorithm))
		self.assertEqual(gwo, self.algorithm_utility.get_algorithm(gwo))
		self.assertTrue(isinstance(self.algorithm_utility.get_algorithm("GreyWolfOptimizer"), GreyWolfOptimizer))

class StoppingTaskBaseTestCase(TestCase):
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
	def setUp(self):
		self.D, self.nFES, self.nGEN = 10, 10, 10
		self.t = StoppingTask(D=self.D, nFES=self.nFES, nGEN=self.nGEN, benchmark=MyBenchmark())

	def test_isFeasible_fine(self):
		x = np.full(self.D, 10)
		self.assertTrue(self.t.isFeasible(x))
		x = np.full(self.D, -10)
		self.assertTrue(self.t.isFeasible(x))
		x = np.random.uniform(-10, 10, self.D)
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

class ThrowingTaskTestCase(TestCase):
	def setUp(self):
		self.D, self.nFES, self.nGEN = 10, 10, 10
		self.t = ThrowingTask(D=self.D, nFES=self.nFES, nGEN=self.nGEN, refValue=0, benchmark=MyBenchmark())

	def test_isFeasible_fine(self):
		x = np.full(self.D, 10)
		self.assertTrue(self.t.isFeasible(x))
		x = np.full(self.D, -10)
		self.assertTrue(self.t.isFeasible(x))
		x = np.random.uniform(-10, 10, self.D)
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

class LimitRepairTestCase(TestCase):
	def setUp(self):
		self.D = 10
		self.Upper, self.Lower = full_array(10, self.D), full_array(-10, self.D)
		self.met = repair.limit

	def generateIndividual(self, D, upper, lower):
		u, l = full_array(upper, D), full_array(lower, D)
		return l + np.random.rand(D) * (u - l)

	def test_limit_repair_good_solution_fine(self):
		x = self.generateIndividual(self.D, self.Upper, self.Lower)
		x = self.met(x, self.Lower, self.Upper)
		self.assertFalse((x > self.Upper).any())
		self.assertFalse((x < self.Lower).any())

	def test_limit_repair_bad_upper_solution_fine(self):
		x = self.generateIndividual(self.D, 12, 11)
		x = self.met(x, self.Lower, self.Upper)
		self.assertFalse((x > self.Upper).any())
		self.assertFalse((x < self.Lower).any())

	def test_limit_repair_bad_lower_soluiton_fine(self):
		x = self.generateIndividual(self.D, -11, -12)
		x = self.met(x, self.Lower, self.Upper)
		self.assertFalse((x > self.Upper).any())
		self.assertFalse((x < self.Lower).any())

	def test_limit_repair_bad_upper_lower_soluiton_fine(self):
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
		self.met = repair.random

class ReflectRepairTestCase(LimitRepairTestCase):
	def setUp(self):
		LimitRepairTestCase.setUp(self)
		self.met = repair.reflect

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
