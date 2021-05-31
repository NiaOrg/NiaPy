# encoding=utf8
import logging
import multiprocessing
import threading

import numpy as np
from numpy.random import default_rng

from niapy.util.array import objects_to_array

logging.basicConfig()
logger = logging.getLogger('niapy.util.utility')
logger.setLevel('INFO')

__all__ = [
    'Algorithm',
    'Individual',
    'default_individual_init',
    'default_numpy_init'
]


def default_numpy_init(task, population_size, rng, **_kwargs):
    r"""Initialize starting population that is represented with `numpy.ndarray` with shape `(population_size, task.dimension)`.

    Args:
        task (Task): Optimization task.
        population_size (int): Number of individuals in population.
        rng (numpy.random.Generator): Random number generator.

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray[float]]:
            1. New population with shape `(population_size, task.D)`.
            2. New population function/fitness values.

    """
    pop = rng.uniform(task.lower, task.upper, (population_size, task.dimension))
    fpop = np.apply_along_axis(task.eval, 1, pop)
    return pop, fpop


def default_individual_init(task, population_size, rng, individual_type=None, **_kwargs):
    r"""Initialize `population_size` individuals of type `individual_type`.

    Args:
        task (Task): Optimization task.
        population_size (int): Number of individuals in population.
        rng (numpy.random.Generator): Random number generator.
        individual_type (Optional[Individual]): Class of individual in population.

    Returns:
        Tuple[numpy.ndarray[Individual], numpy.ndarray[float]:
            1. Initialized individuals.
            2. Initialized individuals function/fitness values.

    """
    pop = objects_to_array([individual_type(task=task, rng=rng, e=True) for _ in range(population_size)])
    return pop, np.asarray([x.f for x in pop])


class Algorithm:
    r"""Class for implementing algorithms.

    Date:
        2018

    Author
        Klemen Berkovič

    License:
        MIT

    Attributes:
        Name (List[str]): List of names for algorithm.
        rng (numpy.random.Generator): Random generator.
        population_size (int): Population size.
        initialization_function (Callable[[int, Task, numpy.random.Generator, Dict[str, Any]], Tuple[numpy.ndarray, numpy.ndarray[float]]]):
            Population initialization function.
        individual_type (Optional[Type[Individual]]): Type of individuals used in population, default value is None for Numpy arrays.

    """

    Name = ['Algorithm', 'AAA']

    def __init__(self, population_size=50, initialization_function=default_numpy_init, individual_type=None,
                 seed=None, *args, **kwargs):
        r"""Initialize algorithm and create name for an algorithm.

        Args:
            population_size (Optional[int]): Population size.
            initialization_function (Optional[Callable[[int, Task, numpy.random.Generator, Dict[str, Any]], Tuple[numpy.ndarray, numpy.ndarray[float]]]]):
                Population initialization function.
            individual_type (Optional[Type[Individual]]): Individual type used in population, default is Numpy array.
            seed (Optional[int]): Starting seed for random generator.

        See Also:
            * :func:`niapy.algorithms.Algorithm.set_parameters`

        """
        self.population_size = population_size
        self.initialization_function = initialization_function
        self.individual_type = individual_type
        self.rng = default_rng(seed)
        self.exception = None

    @staticmethod
    def info():
        r"""Get algorithm information.

        Returns:
            str: Bit item.

        """
        return '''Basic algorithm. No implementation!!!'''

    def set_parameters(self, population_size=50, initialization_function=default_numpy_init, individual_type=None,
                       *args, **kwargs):
        r"""Set the parameters/arguments of the algorithm.

        Args:
            population_size (Optional[int]): Population size.
            initialization_function (Optional[Callable[[int, Task, numpy.random.Generator, Dict[str, Any]], Tuple[numpy.ndarray, numpy.ndarray[float]]]]):
                Population initialization function.
            individual_type (Optional[Type[Individual]]): Individual type used in population, default is Numpy array.

        See Also:
            * :func:`niapy.algorithms.default_numpy_init`
            * :func:`niapy.algorithms.default_individual_init`

        """
        self.population_size = population_size
        self.initialization_function = initialization_function
        self.individual_type = individual_type

    def get_parameters(self):
        r"""Get parameters of the algorithm.

        Returns:
            Dict[str, Any]:
                * Parameter name (str): Represents a parameter name
                * Value of parameter (Any): Represents the value of the parameter

        """
        return {
            'population_size': self.population_size,
            'initialization_function': self.initialization_function,
            'individual_type': self.individual_type
        }

    def random(self, size=None):
        r"""Get random distribution of shape size in range from 0 to 1.

        Args:
            size (Union[None, int, Iterable[int]]): Shape of returned random distribution.

        Returns:
            Union[numpy.ndarray[float], float]: Random number or numbers :math:`\in [0, 1]`.

        """
        return self.rng.random(size)

    def uniform(self, low, high, size=None):
        r"""Get uniform random distribution of shape size in range from "low" to "high".

        Args:
            low (Union[float, Iterable[float]]): Lower bound.
            high (Union[float, Iterable[float]]): Upper bound.
            size (Union[None, int, Iterable[int]]): Shape of returned uniform random distribution.

        Returns:
            Union[numpy.ndarray[float], float]: Array of numbers :math:`\in [\mathit{Lower}, \mathit{Upper}]`.

        """
        return self.rng.uniform(low, high, size)

    def normal(self, loc, scale, size=None):
        r"""Get normal random distribution of shape size with mean "loc" and standard deviation "scale".

        Args:
            loc (float): Mean of the normal random distribution.
            scale (float): Standard deviation of the normal random distribution.
            size (Union[int, Iterable[int]]): Shape of returned normal random distribution.

        Returns:
            Union[numpy.ndarray[float], float]: Array of numbers.

        """
        return self.rng.normal(loc, scale, size)

    def standard_normal(self, size=None):
        r"""Get standard normal distribution of shape size.

        Args:
            size (Union[int, Iterable[int]]): Shape of returned standard normal distribution.

        Returns:
            Union[numpy.ndarray[float], float]: Random generated numbers or one random generated number :math:`\in [0, 1]`.

        """
        return self.rng.standard_normal(size)

    def integers(self, low, high=None, size=None, skip=None):
        r"""Get discrete uniform (integer) random distribution of D shape in range from "low" to "high".

        Args:
            low (Union[int, Iterable[int]]): Lower integer bound.
                If high = None low is 0 and this value is used as high
            high (Union[int, Iterable[int]]): One above upper integer bound.
            size (Union[None, int, Iterable[int]]): shape of returned discrete uniform random distribution.
            skip (Union[None, int, Iterable[int], numpy.ndarray[int]]): numbers to skip.

        Returns:
            Union[int, numpy.ndarray[int]]: Random generated integer number.

        """
        r = self.rng.integers(low, high, size)
        return r if skip is None or r not in skip else self.integers(low, high, size, skip)

    @staticmethod
    def get_best(population, population_fitness, best_x=None, best_fitness=np.inf):
        r"""Get the best individual for population.

        Args:
            population (numpy.ndarray): Current population.
            population_fitness (numpy.ndarray): Current populations fitness/function values of aligned individuals.
            best_x (Optional[numpy.ndarray]): Best individual.
            best_fitness (float): Fitness value of best individual.

        Returns:
            Tuple[numpy.ndarray, float]:
                1. Coordinates of best solution.
                2. beset fitness/function value.

        """
        ib = np.argmin(population_fitness)
        if isinstance(population_fitness, (float, int)) and best_fitness >= population_fitness:
            best_x, best_fitness = population, population_fitness
        elif isinstance(population_fitness, (np.ndarray, list)) and best_fitness >= population_fitness[ib]:
            best_x, best_fitness = population[ib], population_fitness[ib]
        return (best_x.x.copy() if isinstance(best_x, Individual) else best_x.copy()), best_fitness

    def init_population(self, task):
        r"""Initialize starting population of optimization algorithm.

        Args:
            task (Task): Optimization task.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray, Dict[str, Any]]:
                1. New population.
                2. New population fitness values.
                3. Additional arguments.

        See Also:
            * :func:`niapy.algorithms.Algorithm.set_parameters`

        """
        pop, fpop = self.initialization_function(task=task, population_size=self.population_size, rng=self.rng,
                                                 individual_type=self.individual_type)
        return pop, fpop, {}

    def run_iteration(self, task, population, population_fitness, best_x, best_fitness, **params):
        r"""Core functionality of algorithm.

        This function is called on every algorithm iteration.

        Args:
            task (Task): Optimization task.
            population (numpy.ndarray): Current population coordinates.
            population_fitness (numpy.ndarray): Current population fitness value.
            best_x (numpy.ndarray): Current generation best individuals coordinates.
            best_fitness (float): current generation best individuals fitness value.
            **params (Dict[str, Any]): Additional arguments for algorithms.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, float, Dict[str, Any]]:
                1. New populations coordinates.
                2. New populations fitness values.
                3. New global best position/solution
                4. New global best fitness/objective value
                5. Additional arguments of the algorithm.

        See Also:
            * :func:`niapy.algorithms.Algorithm.iteration_generator`

        """
        return population, population_fitness, best_x, best_fitness, params

    def iteration_generator(self, task):
        r"""Run the algorithm for a single iteration and return the best solution.

        Args:
            task (Task): Task with bounds and objective function for optimization.

        Returns:
            Generator[Tuple[numpy.ndarray, float], None, None]: Generator getting new/old optimal global values.

        Yields:
            Tuple[numpy.ndarray, float]:
                1. New population best individuals coordinates.
                2. Fitness value of the best solution.

        See Also:
            * :func:`niapy.algorithms.Algorithm.init_population`
            * :func:`niapy.algorithms.Algorithm.run_iteration`

        """
        pop, fpop, params = self.init_population(task)
        xb, fxb = self.get_best(pop, fpop)
        if task.stopping_condition():
            yield xb, fxb
        while True:
            pop, fpop, xb, fxb, params = self.run_iteration(task, pop, fpop, xb, fxb, **params)
            yield xb, fxb

    def run_task(self, task):
        r"""Start the optimization.

        Args:
            task (Task): Task with bounds and objective function for optimization.

        Returns:
            Tuple[numpy.ndarray, float]:
                1. Best individuals components found in optimization process.
                2. Best fitness value found in optimization process.

        See Also:
            * :func:`niapy.algorithms.Algorithm.iteration_generator`

        """
        algo, xb, fxb = self.iteration_generator(task), None, np.inf
        while not task.stopping_condition():
            xb, fxb = next(algo)
            task.next_iter()
        return xb, fxb

    def run(self, task):
        r"""Start the optimization.

        Args:
            task (Task): Optimization task.

        Returns:
            Tuple[numpy.ndarray, float]:
                1. Best individuals components found in optimization process.
                2. Best fitness value found in optimization process.

        See Also:
            * :func:`niapy.algorithms.Algorithm.run_task`

        """
        try:
            r = self.run_task(task)
            return r[0], r[1] * task.optimization_type.value
        except BaseException as e:
            if threading.current_thread() == threading.main_thread() and multiprocessing.current_process().name == 'MainProcess':
                raise e
            self.exception = e
            return None, None

    def bad_run(self):
        r"""Check if some exceptions where thrown when the algorithm was running.

        Returns:
            bool: True if some error where detected at runtime of the algorithm, otherwise False

        """
        return self.exception is not None


class Individual:
    r"""Class that represents one solution in population of solutions.

    Date:
        2018

    Author:
        Klemen Berkovič

    License:
        MIT

    Attributes:
        x (numpy.ndarray): Coordinates of individual.
        f (float): Function/fitness value of individual.

    """

    def __init__(self, x=None, task=None, e=True, rng=None, **kwargs):
        r"""Initialize new individual.

        Parameters:
            task (Optional[Task]): Optimization task.
            rand (Optional[numpy.random.Generator]): Random generator.
            x (Optional[numpy.ndarray]): Individuals components.
            e (Optional[bool]): True to evaluate the individual on initialization. Default value is True.

        """
        self.f = task.optimization_type.value * np.inf if task is not None else np.inf
        if x is not None:
            self.x = x if isinstance(x, np.ndarray) else np.asarray(x)
        elif task is not None:
            self.generate_solution(task, default_rng(rng))
        if e and task is not None:
            self.evaluate(task, rng)

    def generate_solution(self, task, rng):
        r"""Generate new solution.

        Generate new solution for this individual and set it to ``self.x``.
        This method uses ``rng`` for getting random numbers.
        For generating random components ``rng`` and ``task`` is used.

        Args:
            task (Task): Optimization task.
            rng (numpy.random.Generator): Random numbers generator object.

        """
        self.x = rng.uniform(task.lower, task.upper, task.dimension)

    def evaluate(self, task, rng=None):
        r"""Evaluate the solution.

        Evaluate solution ``this.x`` with the help of task.
        Task is used for repairing the solution and then evaluating it.

        Args:
            task (Task): Objective function object.
            rng (Optional[numpy.random.Generator]): Random generator.

        See Also:
            * :func:`niapy.task.Task.repair`

        """
        self.x = task.repair(self.x, rng=rng)
        self.f = task.eval(self.x)

    def copy(self):
        r"""Return a copy of self.

        Method returns copy of ``this`` object so it is safe for editing.

        Returns:
            Individual: Copy of self.

        """
        return Individual(x=self.x.copy(), e=False, f=self.f)

    def __eq__(self, other):
        r"""Compare the individuals for equalities.

        Args:
            other (Union[Any, numpy.ndarray]): Object that we want to compare this object to.

        Returns:
            bool: `True` if equal or `False` if no equal.

        """
        if isinstance(other, np.ndarray):
            for e in other:
                if self == e:
                    return True
            return False
        return np.array_equal(self.x, other.x) and self.f == other.f

    def __str__(self):
        r"""Print the individual with the solution and objective value.

        Returns:
            str: String representation of self.

        """
        return '%s -> %s' % (self.x, self.f)

    def __getitem__(self, i):
        r"""Get the value of i-th component of the solution.

        Args:
            i (int): Position of the solution component.

        Returns:
            Any: Value of ith component.

        """
        return self.x[i]

    def __setitem__(self, i, v):
        r"""Set the value of i-th component of the solution to v value.

        Args:
            i (int): Position of the solution component.
            v (Any): Value to set to i-th component.

        """
        self.x[i] = v

    def __len__(self):
        r"""Get the length of the solution or the number of components.

        Returns:
            int: Number of components.

        """
        return len(self.x)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
