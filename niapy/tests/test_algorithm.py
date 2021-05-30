# encoding=utf8

import logging
from queue import Queue
from threading import Thread
from unittest import TestCase

import numpy as np
from numpy.random import default_rng

from niapy.algorithms.algorithm import Individual, Algorithm
from niapy.problems import Problem
from niapy.task import Task
from niapy.util import objects_to_array

logging.basicConfig()
logger = logging.getLogger('niapy.test')
logger.setLevel('INFO')


class MyProblem(Problem):
    r"""Testing problem class.

    Date:
        April 2019

    Author:
        Klemen Berkovič

    See Also:
        * :class:`niapy.problems.Problem`

    """

    def __init__(self, dimension=10, *_args, **_kwargs):
        super().__init__(dimension, -5.12, 5.12)

    def _evaluate(self, x):
        return np.sum(x ** 2)


class IndividualTestCase(TestCase):
    r"""Test case for testing Individual class.

    Date:
        April 2019

    Author:
        Klemen Berkovič

    See Also:
        * :class:`niapy.algorithms.Individual`

    """

    def setUp(self):
        self.dimension = 20
        rng = default_rng()
        self.x = rng.uniform(-100, 100, self.dimension)
        self.task = Task(max_evals=230, max_iters=np.inf, problem=MyProblem(self.dimension))
        self.s1 = Individual(x=self.x, e=False)
        self.s2 = Individual(task=self.task, rng=rng)
        self.s3 = Individual(task=self.task)

    def test_generate_solution(self):
        self.assertTrue(self.task.is_feasible(self.s2))
        self.assertTrue(self.task.is_feasible(self.s3))

    def test_evaluate(self):
        self.s1.evaluate(self.task)
        self.assertAlmostEqual(self.s1.f, self.task.eval(self.x))

    def test_repair(self):
        s = Individual(x=np.full(self.dimension, 100))
        self.assertFalse(self.task.is_feasible(s.x))

    def test_eq(self):
        self.assertFalse(self.s1 == self.s2)
        self.assertTrue(self.s1 == self.s1)
        s = Individual(x=self.s1.x)
        self.assertTrue(s == self.s1)

    def test_str(self):
        self.assertEqual(str(self.s1), '%s -> %s' % (self.x, np.inf))

    def test_getitem(self):
        for i in range(self.dimension):
            self.assertEqual(self.s1[i], self.x[i])

    def test_len(self):
        self.assertEqual(len(self.s1), len(self.x))


def init_pop_numpy(task, population_size, **_kwargs):
    r"""Custom population initialization function for numpy individual type.

    Args:
        task (Task): Optimization task.
        population_size (int): Population size.

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray[float]):
            1. Initialized population.
            2. Initialized populations fitness/function values.

    """
    pop = np.zeros((population_size, task.dimension))
    fpop = np.apply_along_axis(task.eval, 1, pop)
    return pop, fpop


def init_pop_individual(task, population_size, individual_type, **_kwargs):
    r"""Custom population initialization function for numpy individual type.

    Args:
        task (Task): Optimization task.
        population_size (int): Population size.
        individual_type (Type[Individual]): Type of individual in population.

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray[float]):
            1. Initialized population.
            2. Initialized populations fitness/function values.

    """
    pop = objects_to_array([individual_type(x=np.zeros(task.dimension), task=task) for _ in range(population_size)])
    return pop, np.asarray([x.f for x in pop])


class AlgorithmBaseTestCase(TestCase):
    r"""Test case for testing Algorithm class.

    Date:
        April 2019

    Author:
        Klemen Berkovič

    Attributes:
        seed (int): Starting seed of random generator.
        rng (numpy.random.Generator): Random generator.
        a (Algorithm): Algorithm to use for testing.

    See Also:
        * :class:`niapy.algorithms.Individual`

    """

    def setUp(self):
        self.seed = 1
        self.rng = default_rng(self.seed)
        self.a = Algorithm(seed=self.seed)

    def test_algorithm_info(self):
        r"""Check if method works fine."""
        i = Algorithm.info()
        self.assertIsNotNone(i)

    def test_algorithm_getParameters(self):
        r"""Check if method works fine."""
        algo = Algorithm()
        params = algo.get_parameters()
        self.assertIsNotNone(params)

    def test_init_population_numpy(self):
        r"""Test if custom generation initialization works ok."""
        a = Algorithm(population_size=10, initialization_function=init_pop_numpy)
        t = Task(problem=MyProblem(dimension=20))
        self.assertTrue(np.array_equal(np.zeros((10, t.dimension)), a.init_population(t)[0]))

    def test_init_population_individual(self):
        r"""Test if custom generation initialization works ok."""
        a = Algorithm(population_size=10, initialization_function=init_pop_individual, individual_type=Individual)
        t = Task(problem=MyProblem(dimension=20))
        i = Individual(x=np.zeros(t.dimension), task=t)
        pop, fpop, d = a.init_population(t)
        for e in pop:
            self.assertEqual(i, e)

    def test_set_parameters(self):
        self.a.set_parameters(t=None, a=20)
        self.assertRaises(AttributeError, lambda: self.assertEqual(self.a.a, None))

    def test_integers(self):
        o = self.a.integers(low=10, high=20, size=[10, 10])
        self.assertEqual(o.shape, (10, 10))
        self.assertTrue(np.array_equal(self.rng.integers(10, 20, (10, 10)), o))
        o = self.a.integers(low=10, high=20, size=(10, 5))
        self.assertEqual(o.shape, (10, 5))
        self.assertTrue(np.array_equal(self.rng.integers(10, 20, (10, 5)), o))
        o = self.a.integers(low=10, high=20, size=10)
        self.assertEqual(o.shape, (10,))
        self.assertTrue(np.array_equal(self.rng.integers(10, 20, 10), o))

    def test_standard_normal(self):
        a = self.a.standard_normal([1, 2])
        self.assertEqual(a.shape, (1, 2))
        self.assertTrue(np.array_equal(self.rng.standard_normal((1, 2)), a))
        a = self.a.standard_normal(1)
        self.assertEqual(len(a), 1)
        self.assertTrue(np.array_equal(self.rng.standard_normal(1), a))
        a = self.a.standard_normal(2)
        self.assertEqual(len(a), 2)
        self.assertTrue(np.array_equal(self.rng.standard_normal(2), a))
        a = self.a.standard_normal()
        self.assertIsInstance(a, float)
        self.assertTrue(np.array_equal(self.rng.standard_normal(), a))

    def test_uniform(self):
        a = self.a.uniform(-10, 10, [10, 10])
        self.assertEqual(a.shape, (10, 10))
        self.assertTrue(np.array_equal(self.rng.uniform(-10, 10, (10, 10)), a))
        a = self.a.uniform(4, 10, (4, 10))
        self.assertEqual(len(a), 4)
        self.assertEqual(len(a[0]), 10)
        self.assertTrue(np.array_equal(self.rng.uniform(4, 10, (4, 10)), a))
        a = self.a.uniform(1, 4, 2)
        self.assertEqual(len(a), 2)
        self.assertTrue(np.array_equal(self.rng.uniform(1, 4, 2), a))
        a = self.a.uniform(10, 100)
        self.assertIsInstance(a, float)
        self.assertEqual(self.rng.uniform(10, 100), a)

    def test_normal(self):
        a = self.a.normal(-10, 10, [10, 10])
        self.assertEqual(a.shape, (10, 10))
        self.assertTrue(np.array_equal(self.rng.normal(-10, 10, (10, 10)), a))
        a = self.a.normal(4, 10, (4, 10))
        self.assertEqual(len(a), 4)
        self.assertEqual(len(a[0]), 10)
        self.assertTrue(np.array_equal(self.rng.normal(4, 10, (4, 10)), a))
        a = self.a.normal(1, 4, 2)
        self.assertEqual(len(a), 2)
        self.assertTrue(np.array_equal(self.rng.normal(1, 4, 2), a))
        a = self.a.normal(10, 100)
        self.assertIsInstance(a, float)
        self.assertEqual(self.rng.normal(10, 100), a)


class TestingTask(Task, TestCase):
    r"""Testing task.

    Date:
        April 2019

    Author:
        Klemen Berkovič

    See Also:
        * :class:`niapy.task.Task`

    """

    def eval(self, x):
        r"""Check if is algorithm trying to evaluate solution out of bounds."""
        self.assertTrue(self.is_feasible(x), 'Solution %s is not in feasible space!!!' % x)
        return super().eval(x)


class AlgorithmTestCase(TestCase):
    r"""Base class for testing other algorithms.

    Date:
        April 2019

    Author:
        Klemen Berkovič

    Attributes:
        dimensions (List[int]): Dimension of problem.
        max_iters (int): Number of generations/iterations.
        max_evals (int): Number of function evaluations.
        seed (int): Starting seed of random generator.

    See Also:
        * :class:`niapy.algorithms.Algorithm`

    """

    def setUp(self):
        r"""Setup basic parameters of the algorithm run."""
        self.dimensions = (10, 40)
        self.max_iters = 1000
        self.max_evals = 1000
        self.seed = 1
        self.algo = Algorithm

    def test_algorithm_info(self):
        r"""Test if algorithm info works fine."""
        info = self.algo.info()
        self.assertIsNotNone(info)

    def test_algorithm_get_parameters(self):
        r"""Test if algorithms parameters values are fine."""
        params = self.algo().get_parameters()
        self.assertIsNotNone(params)

    def setUpTasks(self, dimension, problem='griewank', max_evals=None, max_iters=None):
        r"""Setup optimization tasks for testing.

        Args:
            dimension (int): Dimension of the problem.
            problem (Optional[str, class]): Optimization problem to use.
            max_evals (int): Number of fitness/objective function evaluations.
            max_iters (int): Number of generations.

        Returns:
            Tuple[Task, Task]: Two testing tasks.

        """
        if isinstance(problem, Problem):
            cls = problem.__class__
            problem = cls(dimension=dimension)
            dimension = None
        max_evals = self.max_evals if max_evals is None else max_evals
        max_iters = self.max_iters if max_iters is None else max_iters
        task1 = TestingTask(dimension=dimension, max_evals=max_evals, max_iters=max_iters, problem=problem)
        task2 = TestingTask(dimension=dimension, max_evals=max_evals, max_iters=max_iters, problem=problem)
        return task1, task2

    def test_algorithm_run(self, a=None, b=None, problem='griewank', max_evals=None, max_iters=None):
        r"""Run main testing of algorithm.

        Args:
            a (Algorithm): First instance of algorithm.
            b (Algorithm): Second instance of algorithm.
            problem (Union[Problem, str]): Problem to use for testing.
            max_evals (int): Number of function evaluations.
            max_iters (int): Number of algorithm generations/iterations.

        """
        if a is None or b is None:
            return
        for D in self.dimensions:
            task1, task2 = self.setUpTasks(D, problem, max_evals=max_evals, max_iters=max_iters)
            # x = a.run(task1) # For debugging purposes
            # y = b.run(task2) # For debugging purposes
            q = Queue(maxsize=2)
            thread1 = Thread(target=lambda algorithm, task, queue: queue.put(algorithm.run(task)), args=(a, task1, q))
            thread2 = Thread(target=lambda algorithm, task, queue: queue.put(algorithm.run(task)), args=(b, task2, q))
            thread1.start()
            thread2.start()
            thread1.join()
            thread2.join()
            x = q.get(block=True)
            y = q.get(block=True)
            if a.bad_run():
                raise a.exception
            self.assertFalse(a.bad_run() or b.bad_run(),
                             "Something went wrong at runtime of the algorithm --> %s" % a.exception)
            self.assertIsNotNone(x)
            self.assertIsNotNone(y)
            logger.info('%s\n%s -> %s\n%s -> %s' % (task1.problem.name(), x[0], x[1], y[0], y[1]))
            self.assertAlmostEqual(task1.problem.evaluate(x[0].x if isinstance(x[0], Individual) else x[0]),
                                   x[1], msg='Best individual fitness values does not mach the given one')
            self.assertAlmostEqual(task1.x_f, x[1],
                                   msg='While running the algorithm, algorithm got better individual with fitness: %s' % task1.x_f)
            self.assertTrue(np.array_equal(x[0], y[0]),
                            'Results can not be reproduced, check usages of random number generator')
            self.assertAlmostEqual(x[1], y[1], msg='Results can not be reproduced or bad function value')
            self.assertTrue(self.max_evals if max_evals is None else max_evals >= task1.evals)
            self.assertEqual(task1.evals, task2.evals)
            self.assertTrue(self.max_iters if max_iters is None else max_iters >= task1.iters)
            self.assertEqual(task1.iters, task2.iters)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
