# encoding=utf8

import logging
from queue import Queue
from threading import Thread
from unittest import TestCase

from numpy import random as rnd, full, inf, array_equal, apply_along_axis, asarray

from NiaPy.util import objects2array
from NiaPy.task.task import Task, StoppingTask
from NiaPy.algorithms.algorithm import Individual, Algorithm
from NiaPy.benchmarks import Benchmark, Sphere

logging.basicConfig()
logger = logging.getLogger('NiaPy.test')
logger.setLevel('INFO')

class MyBenchmark(Benchmark):
	r"""Testing benchmark class.

	Date:
		April 2019

	Author:
		Klemen Berkovič

	See Also:
		* :class:`NiaPy.benchmarks.Benchmark`
	"""
	def __init__(self):
		Benchmark.__init__(self, -5.12, 5.12)

	def function(self):
		return Sphere(self.Lower, self.Upper).function()

class IndividualTestCase(TestCase):
	r"""Test case for testing Individual class.

	Date:
		April 2019

	Author:
		Klemen Berkovič

	See Also:
		* :class:`NiaPy.algorithms.Individual`
	"""
	def setUp(self):
		self.D = 20
		self.x, self.task = rnd.uniform(-100, 100, self.D), StoppingTask(D=self.D, nFES=230, nGEN=inf, benchmark=MyBenchmark())
		self.s1, self.s2, self.s3 = Individual(x=self.x, e=False), Individual(task=self.task, rand=rnd), Individual(task=self.task)

	def test_generateSolutin_fine(self):
		self.assertTrue(self.task.isFeasible(self.s2))
		self.assertTrue(self.task.isFeasible(self.s3))

	def test_evaluate_fine(self):
		self.s1.evaluate(self.task)
		self.assertAlmostEqual(self.s1.f, self.task.eval(self.x))

	def test_repair_fine(self):
		s = Individual(x=full(self.D, 100))
		self.assertFalse(self.task.isFeasible(s.x))

	def test_eq_fine(self):
		self.assertFalse(self.s1 == self.s2)
		self.assertTrue(self.s1 == self.s1)
		s = Individual(x=self.s1.x)
		self.assertTrue(s == self.s1)

	def test_str_fine(self):
		self.assertEqual(str(self.s1), '%s -> %s' % (self.x, inf))

	def test_getitem_fine(self):
		for i in range(self.D): self.assertEqual(self.s1[i], self.x[i])

	def test_len_fine(self):
		self.assertEqual(len(self.s1), len(self.x))

def init_pop_numpy(task, NP, **kwargs):
	r"""Custom population initialization function for numpy individual type.

	Args:
		task (Task): Optimization task.
		np (int): Population size.
		**kwargs (Dict[str, Any]): Additional arguments.

	Returns:
		Tuple[numpy.ndarray, numpy.ndarray[float]):
			1. Initialized population.
			2. Initialized populations fitness/function values.
	"""
	pop = full((NP, task.D), 0.0)
	fpop = apply_along_axis(task.eval, 1, pop)
	return pop, fpop

def init_pop_individual(task, NP, itype, **kwargs):
	r"""Custom population initialization function for numpy individual type.

	Args:
		task (Task): Optimization task.
		np (int): Population size.
		itype (Individual): Type of individual in population.
		**kwargs (Dict[str, Any]): Additional arguments.

	Returns:
		Tuple[numpy.ndarray, numpy.ndarray[float]):
			1. Initialized population.
			2. Initialized populations fitness/function values.
	"""
	pop = objects2array([itype(x=full(task.D, 0.0), task=task) for _ in range(NP)])
	return pop, asarray([x.f for x in pop])

class AlgorithBaseTestCase(TestCase):
	r"""Test case for testing Algorithm class.

	Date:
		April 2019

	Author:
		Klemen Berkovič

	Attributes:
		seed (int): Starting seed of random generator.
		rnd (mtrand.RandomState): Random generator.
		a (Algorithm): Algorithm to use for testing.

	See Also:
		* :class:`NiaPy.algorithms.Individual`
	"""
	def setUp(self):
		self.seed = 1
		self.rnd = rnd.RandomState(self.seed)
		self.a = Algorithm(seed=self.seed)

	def test_algorithm_info_fine(self):
		r"""Check if method works fine."""
		i = Algorithm.algorithmInfo()
		self.assertIsNotNone(i)

	def test_algorithm_getParameters_fine(self):
		r"""Check if method works fine."""
		algo = Algorithm()
		params = algo.getParameters()
		self.assertIsNotNone(params)

	def test_type_parameters_fine(self):
		d = Algorithm.typeParameters()
		self.assertIsNotNone(d)

	def test_init_population_numpy_fine(self):
		r"""Test if custome generation initialization works ok."""
		a = Algorithm(NP=10, InitPopFunc=init_pop_numpy)
		t = Task(D=20, benchmark=MyBenchmark())
		self.assertTrue(array_equal(full((10, t.D), 0.0), a.initPopulation(t)[0]))

	def test_init_population_individual_fine(self):
		r"""Test if custome generation initialization works ok."""
		a = Algorithm(NP=10, InitPopFunc=init_pop_individual, itype=Individual)
		t = Task(D=20, benchmark=MyBenchmark())
		i = Individual(x=full(t.D, 0.0), task=t)
		pop, fpop, d = a.initPopulation(t)
		for e in pop: self.assertEqual(i, e)

	def test_setParameters(self):
		self.a.setParameters(t=None, a=20)
		self.assertRaises(AttributeError, lambda: self.assertEqual(self.a.a, None))

	def test_randint_fine(self):
		o = self.a.randint(Nmax=20, Nmin=10, D=[10, 10])
		self.assertEqual(o.shape, (10, 10))
		self.assertTrue(array_equal(self.rnd.randint(10, 20, (10, 10)), o))
		o = self.a.randint(Nmax=20, Nmin=10, D=(10, 5))
		self.assertEqual(o.shape, (10, 5))
		self.assertTrue(array_equal(self.rnd.randint(10, 20, (10, 5)), o))
		o = self.a.randint(Nmax=20, Nmin=10, D=10)
		self.assertEqual(o.shape, (10,))
		self.assertTrue(array_equal(self.rnd.randint(10, 20, 10), o))

	def test_randn_fine(self):
		a = self.a.randn([1, 2])
		self.assertEqual(a.shape, (1, 2))
		self.assertTrue(array_equal(self.rnd.randn(1, 2), a))
		a = self.a.randn(1)
		self.assertEqual(len(a), 1)
		self.assertTrue(array_equal(self.rnd.randn(1), a))
		a = self.a.randn(2)
		self.assertEqual(len(a), 2)
		self.assertTrue(array_equal(self.rnd.randn(2), a))
		a = self.a.randn()
		self.assertIsInstance(a, float)
		self.assertTrue(array_equal(self.rnd.randn(), a))

	def test_uniform_fine(self):
		a = self.a.uniform(-10, 10, [10, 10])
		self.assertEqual(a.shape, (10, 10))
		self.assertTrue(array_equal(self.rnd.uniform(-10, 10, (10, 10)), a))
		a = self.a.uniform(4, 10, (4, 10))
		self.assertEqual(len(a), 4)
		self.assertEqual(len(a[0]), 10)
		self.assertTrue(array_equal(self.rnd.uniform(4, 10, (4, 10)), a))
		a = self.a.uniform(1, 4, 2)
		self.assertEqual(len(a), 2)
		self.assertTrue(array_equal(self.rnd.uniform(1, 4, 2), a))
		a = self.a.uniform(10, 100)
		self.assertIsInstance(a, float)
		self.assertEqual(self.rnd.uniform(10, 100), a)

	def test_normal_fine(self):
		a = self.a.normal(-10, 10, [10, 10])
		self.assertEqual(a.shape, (10, 10))
		self.assertTrue(array_equal(self.rnd.normal(-10, 10, (10, 10)), a))
		a = self.a.normal(4, 10, (4, 10))
		self.assertEqual(len(a), 4)
		self.assertEqual(len(a[0]), 10)
		self.assertTrue(array_equal(self.rnd.normal(4, 10, (4, 10)), a))
		a = self.a.normal(1, 4, 2)
		self.assertEqual(len(a), 2)
		self.assertTrue(array_equal(self.rnd.normal(1, 4, 2), a))
		a = self.a.normal(10, 100)
		self.assertIsInstance(a, float)
		self.assertEqual(self.rnd.normal(10, 100), a)

class TestingTask(StoppingTask, TestCase):
	r"""Testing task.

	Date:
		April 2019

	Author:
		Klemen Berkovič

	See Also:
		* :class:`NiaPy.util.StoppingTask`
	"""
	def names(self):
		r"""Get names of benchmark.

		Returns:
			 List[str]: List of task names.
		"""
		return self.benchmark.Name

	def eval(self, A):
		r"""Check if is algorithm trying to evaluate solution out of bounds."""
		self.assertTrue(self.isFeasible(A), 'Solution %s is not in feasible space!!!' % A)
		return StoppingTask.eval(self, A)

class AlgorithmTestCase(TestCase):
	r"""Base class for testing other algorithms.

	Date:
		April 2019

	Author:
		Klemen Berkovič

	Attributes:
		D (List[int]): Dimension of problem.
		nGEN (int): Number of generations/iterations.
		nFES (int): Number of function evaluations.
		seed (int): Starting seed of random generator.

	See Also:
		* :class:`NiaPy.algorithms.Algorithm`
	"""
	def setUp(self):
		r"""Setup basic parameters of the algorithm run."""
		self.D, self.nGEN, self.nFES, self.seed = [10, 40], 1000, 1000, 1
		self.algo = Algorithm

	def test_algorithm_type_parameters(self):
		r"""Test if type parametes for algorithm work fine."""
		tparams = self.algo.typeParameters()
		self.assertIsNotNone(tparams)

	def test_algorithm_info_fine(self):
		r"""Test if algorithm info works fine."""
		info = self.algo.algorithmInfo()
		self.assertIsNotNone(info)

	def test_algorithm_get_parameters_fine(self):
		r"""Test if algorithms parameters values are fine."""
		params = self.algo().getParameters()
		self.assertIsNotNone(params)

	def setUpTasks(self, D, bech='griewank', nFES=None, nGEN=None):
		r"""Setup optimization tasks for testing.

		Args:
			D (int): Dimension of the problem.
			bech (Optional[str, class]): Optimization problem to use.
			nFES (int): Number of fitness/objective function evaluations.
			nGEN (int): Number of generations.

		Returns:
			Tuple[Taks, Taks]: Two testing tasks.
		"""
		task1, task2 = TestingTask(D=D, nFES=self.nFES if nFES is None else nFES, nGEN=self.nGEN if nGEN is None else nGEN, benchmark=bech), TestingTask(D=D, nFES=self.nFES if nFES is None else nFES, nGEN=self.nGEN if nGEN is None else nGEN, benchmark=bech)
		return task1, task2

	def test_algorithm_run(self, a=None, b=None, benc='griewank', nFES=None, nGEN=None):
		r"""Run main testing of algorithm.

		Args:
			a (Algorithm): First instance of algorithm.
			b (Algorithm): Second instance of algorithm.
			benc (Union[Benchmark, str]): Benchmark to use for testing.
			nFES (int): Number of function evaluations.
			nGEN (int): Number of algorithm generations/iterations.
		"""
		if a is None or b is None: return
		for D in self.D:
			task1, task2 = self.setUpTasks(D, benc, nFES=nFES)
			q = Queue(maxsize=2)
			thread1, thread2 = Thread(target=lambda a, t, q: q.put(a.run(t)), args=(a, task1, q)), Thread(target=lambda a, t, q: q.put(a.run(t)), args=(b, task2, q))
			thread1.start(), thread2.start()
			thread1.join(), thread2.join()
			x, y = q.get(block=True), q.get(block=True)
			self.assertFalse(a.bad_run() or b.bad_run(), "Something went wrong at runtime of the algorithm --> %s" % a.exception)
			self.assertIsNotNone(x), self.assertIsNotNone(y)
			logger.info('%s\n%s -> %s\n%s -> %s' % (task1.names(), x[0], x[1], y[0], y[1]))
			self.assertAlmostEqual(task1.benchmark.function()(D, x[0].x if isinstance(x[0], Individual) else x[0]), x[1], msg='Best individual fitness values does not mach the given one')
			self.assertAlmostEqual(task1.x_f, x[1], msg='While running the algorithm, algorithm got better individual with fitness: %s' % task1.x_f)
			self.assertTrue(array_equal(x[0], y[0]), 'Results can not be reproduced, check usages of random number generator')
			self.assertAlmostEqual(x[1], y[1], msg='Results can not be reproduced or bad function value')
			self.assertTrue(self.nFES >= task1.Evals), self.assertEqual(task1.Evals, task2.Evals)
			self.assertTrue(self.nGEN >= task1.Iters), self.assertEqual(task1.Iters, task2.Iters)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
