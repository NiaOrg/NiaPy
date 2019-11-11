# encoding=utf8
import logging

from numpy import random as rand, inf, ndarray, asarray, array_equal, argmin, apply_along_axis

from NiaPy.util import FesException, GenException, TimeException, RefException
from NiaPy.util.utility import objects2array

logging.basicConfig()
logger = logging.getLogger('NiaPy.util.utility')
logger.setLevel('INFO')

__all__ = [
	'Algorithm',
	'Individual',
	'defaultIndividualInit',
	'defaultNumPyInit'
]

def defaultNumPyInit(task, NP, rnd=rand, **kwargs):
	r"""Initialize starting population that is represented with `numpy.ndarray` with shape `{NP, task.D}`.

	Args:
		task (Task): Optimization task.
		NP (int): Number of individuals in population.
		rnd (Optional[mtrand.RandomState]): Random number generator.
		kwargs (Dict[str, Any]): Additional arguments.

	Returns:
		Tuple[numpy.ndarray, numpy.ndarray[float]]:
			1. New population with shape `{NP, task.D}`.
			2. New population function/fitness values.
	"""
	pop = task.Lower + rnd.rand(NP, task.D) * task.bRange
	fpop = apply_along_axis(task.eval, 1, pop)
	return pop, fpop

def defaultIndividualInit(task, NP, rnd=rand, itype=None, **kwargs):
	r"""Initialize `NP` individuals of type `itype`.

	Args:
		task (Task): Optimization task.
		NP (int): Number of individuals in population.
		rnd (Optional[mtrand.RandomState]): Random number generator.
		itype (Optional[Individual]): Class of individual in population.
		kwargs (Dict[str, Any]): Additional arguments.

	Returns:
		Tuple[numpy.ndarray[Individual], numpy.ndarray[float]:
			1. Initialized individuals.
			2. Initialized individuals function/fitness values.
	"""
	pop = objects2array([itype(task=task, rnd=rnd, e=True) for _ in range(NP)])
	return pop, asarray([x.f for x in pop])

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
		Rand (mtrand.RandomState): Random generator.
		NP (int): Number of inidividuals in populatin.
		InitPopFunc (Callable[[int, Task, mtrand.RandomState, Dict[str, Any]], Tuple[numpy.ndarray, numpy.ndarray[float]]]): Idividual initialization function.
		itype (Individual): Type of individuals used in population, default value is None for Numpy arrays.
	"""
	Name = ['Algorithm', 'AAA']
	Rand = rand.RandomState(None)
	NP = 50
	InitPopFunc = defaultNumPyInit
	itype = None

	@staticmethod
	def typeParameters():
		r"""Return functions for checking values of parameters.

		Return:
			Dict[str, Callable]:
				* NP (Callable[[int], bool]): Check if number of individuals is :math:`\in [0, \infty]`.
		"""
		return {'NP': lambda x: isinstance(x, int) and x > 0}

	def __init__(self, **kwargs):
		r"""Initialize algorithm and create name for an algorithm.

		Args:
			seed (int): Starting seed for random generator.

		See Also:
			* :func:`NiaPy.algorithms.Algorithm.setParameters`
		"""
		self.Rand, self.exception = rand.RandomState(kwargs.pop('seed', None)), None
		self.setParameters(**kwargs)

	@staticmethod
	def algorithmInfo():
		r"""Get algorithm information.

		Returns:
			str: Bit item.
		"""
		return '''Basic algorithm. No implementation!!!'''

	def setParameters(self, NP=50, InitPopFunc=defaultNumPyInit, itype=None, **kwargs):
		r"""Set the parameters/arguments of the algorithm.

		Args:
			NP (Optional[int]): Number of individuals in population :math:`\in [1, \infty]`.
			InitPopFunc (Optional[Callable[[int, Task, mtrand.RandomState, Dict[str, Any]], Tuple[numpy.ndarray, numpy.ndarray[float]]]]): Type of individuals used by algorithm.
			itype (Optional[Any]): Individual type used in population, default is Numpy array.
			**kwargs (Dict[str, Any]): Additional arguments.

		See Also:
			* :func:`NiaPy.algorithms.defaultNumPyInit`
			* :func:`NiaPy.algorithms.defaultIndividualInit`
		"""
		self.NP, self.InitPopFunc, self.itype = NP, InitPopFunc, itype

	def getParameters(self):
		r"""Get parameters of the algorithm.

		Returns:
			Dict[str, Any]:
			* Parameter name (str): Represents a parameter name
			* Value of parameter (Any): Represents the value of the parameter
		"""
		return {
			'NP': self.NP,
			'InitPopFunc': self.InitPopFunc,
			'itype': self.itype
		}

	def rand(self, D=1):
		r"""Get random distribution of shape D in range from 0 to 1.

		Args:
			D (numpy.ndarray[int]): Shape of returned random distribution.

		Returns:
			Union[numpy.ndarray[float], float]: Random number or numbers :math:`\in [0, 1]`.
		"""
		if isinstance(D, (ndarray, list)): return self.Rand.rand(*D)
		elif D > 1: return self.Rand.rand(D)
		else: return self.Rand.rand()

	def uniform(self, Lower, Upper, D=None):
		r"""Get uniform random distribution of shape D in range from "Lower" to "Upper".

		Args:
			Lower (Iterable[float]): Lower bound.
			Upper (Iterable[float]): Upper bound.
			D (Union[int, Iterable[int]]): Shape of returned uniform random distribution.

		Returns:
			Union[numpy.ndarray[float], float]: Array of numbers :math:`\in [\mathit{Lower}, \mathit{Upper}]`.
		"""
		return self.Rand.uniform(Lower, Upper, D) if D is not None else self.Rand.uniform(Lower, Upper)

	def normal(self, loc, scale, D=None):
		r"""Get normal random distribution of shape D with mean "loc" and standard deviation "scale".

		Args:
			loc (float): Mean of the normal random distribution.
			scale (float): Standard deviation of the normal random distribution.
			D (Union[int, Iterable[int]]): Shape of returned normal random distribution.

		Returns:
			Union[numpy.ndarray[float], float]: Array of numbers.
		"""
		return self.Rand.normal(loc, scale, D) if D is not None else self.Rand.normal(loc, scale)

	def randn(self, D=None):
		r"""Get standard normal distribution of shape D.

		Args:
			D (Union[int, Iterable[int]]): Shape of returned standard normal distribution.

		Returns:
			Union[numpy.ndarray[float], float]: Random generated numbers or one random generated number :math:`\in [0, 1]`.
		"""
		if D is None: return self.Rand.randn()
		elif isinstance(D, int): return self.Rand.randn(D)
		return self.Rand.randn(*D)

	def randint(self, Nmax, D=1, Nmin=0, skip=None):
		r"""Get discrete uniform (integer) random distribution of D shape in range from "Nmin" to "Nmax".

		Args:
			Nmin (int): Lower integer bound.
			Nmax (int): One above upper integer bound.
			D (Union[int, Iterable[int]]): shape of returned discrete uniform random distribution.
			skip (Union[int, Iterable[int], numpy.ndarray[int]]): numbers to skip.

		Returns:
			Union[int, numpy.ndarrayj[int]]: Random generated integer number.
		"""
		r = None
		if isinstance(D, (list, tuple, ndarray)): r = self.Rand.randint(Nmin, Nmax, D)
		elif D > 1: r = self.Rand.randint(Nmin, Nmax, D)
		else: r = self.Rand.randint(Nmin, Nmax)
		return r if skip is None or r not in skip else self.randint(Nmax, D, Nmin, skip)

	def getBest(self, X, X_f, xb=None, xb_f=inf):
		r"""Get the best individual for population.

		Args:
			X (numpy.ndarray): Current population.
			X_f (numpy.ndarray): Current populations fitness/function values of aligned individuals.
			xb (numpy.ndarray): Best individual.
			xb_f (float): Fitness value of best individual.

		Returns:
			Tuple[numpy.ndarray, float]:
				1. Coordinates of best solution.
				2. beset fitness/function value.
		"""
		ib = argmin(X_f)
		if isinstance(X_f, (float, int)) and xb_f >= X_f: xb, xb_f = X, X_f
		elif isinstance(X_f, (ndarray, list)) and xb_f >= X_f[ib]: xb, xb_f = X[ib], X_f[ib]
		return (xb.x.copy() if isinstance(xb, Individual) else xb.copy()), xb_f

	def initPopulation(self, task):
		r"""Initialize starting population of optimization algorithm.

		Args:
			task (Task): Optimization task.

		Returns:
			Tuple[numpy.ndarray, numpy.ndarray, Dict[str, Any]]:
				1. New population.
				2. New population fitness values.
				3. Additional arguments.

		See Also:
			* :func:`NiaPy.algorithms.Algorithm.setParameters`
		"""
		pop, fpop = self.InitPopFunc(task=task, NP=self.NP, rnd=self.Rand, itype=self.itype)
		return pop, fpop, {}

	def runIteration(self, task, pop, fpop, xb, fxb, **dparams):
		r"""Core functionality of algorithm.

		This function is called on every algorithm iteration.

		Args:
			task (Task): Optimization task.
			pop (numpy.ndarray): Current population coordinates.
			fpop (numpy.ndarray): Current population fitness value.
			xb (numpy.ndarray): Current generation best individuals coordinates.
			xb_f (float): current generation best individuals fitness value.
			**dparams (Dict[str, Any]): Additional arguments for algorithms.

		Returns:
			Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, float, Dict[str, Any]]:
				1. New populations coordinates.
				2. New populations fitness values.
				3. New global best position/solution
				4. New global best fitness/objective value
				5. Additional arguments of the algorithm.

		See Also:
			* :func:`NiaPy.algorithms.Algorithm.runYield`
		"""
		return pop, fpop, xb, fxb, dparams

	def runYield(self, task):
		r"""Run the algorithm for a single iteration and return the best solution.

		Args:
			task (Task): Task with bounds and objective function for optimization.

		Returns:
			Generator[Tuple[numpy.ndarray, float], None, None]: Generator getting new/old optimal global values.

		Yield:
			Tuple[numpy.ndarray, float]:
				1. New population best individuals coordinates.
				2. Fitness value of the best solution.

		See Also:
			* :func:`NiaPy.algorithms.Algorithm.initPopulation`
			* :func:`NiaPy.algorithms.Algorithm.runIteration`
		"""
		pop, fpop, dparams = self.initPopulation(task)
		xb, fxb = self.getBest(pop, fpop)
		yield xb, fxb
		while True:
			pop, fpop, xb, fxb, dparams = self.runIteration(task, pop, fpop, xb, fxb, **dparams)
			yield xb, fxb

	def runTask(self, task):
		r"""Start the optimization.

		Args:
			task (Task): Task with bounds and objective function for optimization.

		Returns:
			Tuple[numpy.ndarray, float]:
				1. Best individuals components found in optimization process.
				2. Best fitness value found in optimization process.

		See Also:
			* :func:`NiaPy.algorithms.Algorithm.runYield`
		"""
		algo, xb, fxb = self.runYield(task), None, inf
		while not task.stopCond():
			xb, fxb = next(algo)
			task.nextIter()
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
			* :func:`NiaPy.algorithms.Algorithm.runTask`
		"""
		try:
			# task.start()
			r = self.runTask(task)
			return r[0], r[1] * task.optType.value
		except (FesException, GenException, TimeException, RefException): return task.x, task.x_f * task.optType.value
		except Exception as e: self.exception = e
		return None, None

	def bad_run(self):
		r"""Check if some exeptions where thrown when the algorithm was running.

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
	x = None
	f = inf

	def __init__(self, x=None, task=None, e=True, rnd=rand, **kwargs):
		r"""Initialize new individual.

		Parameters:
			task (Optional[Task]): Optimization task.
			rand (Optional[mtrand.RandomState]): Random generator.
			x (Optional[numpy.ndarray]): Individuals components.
			e (Optional[bool]): True to evaluate the individual on initialization. Default value is True.
			**kwargs (Dict[str, Any]): Additional arguments.
		"""
		self.f = task.optType.value * inf if task is not None else inf
		if x is not None: self.x = x if isinstance(x, ndarray) else asarray(x)
		else: self.generateSolution(task, rnd)
		if e and task is not None: self.evaluate(task, rnd)

	def generateSolution(self, task, rnd=rand):
		r"""Generate new solution.

		Generate new solution for this individual and set it to ``self.x``.
		This method uses ``rnd`` for getting random numbers.
		For generating random components ``rnd`` and ``task`` is used.

		Args:
			task (Task): Optimization task.
			rnd (Optional[mtrand.RandomState]): Random numbers generator object.
		"""
		if task is not None: self.x = task.Lower + task.bRange * rnd.rand(task.D)

	def evaluate(self, task, rnd=rand):
		r"""Evaluate the solution.

		Evaluate solution ``this.x`` with the help of task.
		Task is used for reparing the solution and then evaluating it.

		Args:
			task (Task): Objective function object.
			rnd (Optional[mtrand.RandomState]): Random generator.

		See Also:
			* :func:`NiaPy.util.Task.repair`
		"""
		self.x = task.repair(self.x, rnd=rnd)
		self.f = task.eval(self.x)

	def copy(self):
		r"""Return a copy of self.

		Method returns copy of ``this`` object so it is safe for editing.

		Returns:
			Individual: Copy of self.
		"""
		return Individual(x=self.x.copy(), f=self.f, e=False)

	def __eq__(self, other):
		r"""Compare the individuals for equalities.

		Args:
			other (Union[Any, numpy.ndarray]): Object that we want to compare this object to.

		Returns:
			bool: `True` if equal or `False` if no equal.
		"""
		if isinstance(other, ndarray):
			for e in other:
				if self == e: return True
			return False
		return array_equal(self.x, other.x) and self.f == other.f

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
