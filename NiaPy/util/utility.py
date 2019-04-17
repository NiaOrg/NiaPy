# encoding=utf8
# pylint: disable=line-too-long, mixed-indentation, bad-continuation,multiple-statements, unused-argument, no-self-use, trailing-comma-tuple, logging-not-lazy, no-else-return, dangerous-default-value, assignment-from-no-return, superfluous-parens

"""Implementation of benchmarks utility function."""
import logging
# from datetime import datetime
from enum import Enum

from numpy import ndarray, asarray, full, empty, inf, dot, where, random as rand, fabs, ceil, amin, amax
from matplotlib import pyplot as plt, animation as anim

from NiaPy.benchmarks import Rastrigin, Rosenbrock, Griewank, Sphere, Ackley, Schwefel, Schwefel221, Schwefel222, Whitley, Alpine1, Alpine2, HappyCat, Ridge, ChungReynolds, Csendes, Pinter, Qing, Quintic, Salomon, SchumerSteiglitz, Step, Step2, Step3, Stepint, SumSquares, StyblinskiTang, BentCigar, Discus, Elliptic, ExpandedGriewankPlusRosenbrock, HGBat, Katsuura, ExpandedSchaffer, ModifiedSchwefel, Weierstrass, Michalewichz, Levy, Sphere2, Sphere3, Trid, Perm, Zakharov, DixonPrice, Powell, CosineMixture, Infinity, SchafferN2, SchafferN4
from NiaPy.util.exception import FesException, GenException, RefException  # TimeException,

logging.basicConfig()
logger = logging.getLogger('NiaPy.util.utility')
logger.setLevel('INFO')

__all__ = [
	'Utility',
	'limitRepair',
	'limitInversRepair',
	'wangRepair',
	'randRepair',
	'Task',
	'CountingTask',
	'StoppingTask',
	'ThrowingTask',
	'TaskConvPrint',
	'TaskConvPlot',
	'TaskConvSave',
	'fullArray',
	'objects2array',
	'TaskComposition',
	'OptimizationType',
	'ScaledTask'
]

def fullArray(a, D):
	r"""Fill or create array of length D, from value or value form a.

	Arguments:
		a (Union[int, float, numpy.ndarray], Iterable[Any]): Input values for fill.
		D (int): Length of new array.

	Returns:
		numpy.ndarray: Array filled with passed values or value.
	"""
	A = []
	if isinstance(a, (int, float)):	A = full(D, a)
	elif isinstance(a, (ndarray, list, tuple)):
		if len(a) == D: A = a if isinstance(a, ndarray) else asarray(a)
		elif len(a) > D: A = a[:D] if isinstance(a, ndarray) else asarray(a[:D])
		else:
			for i in range(int(ceil(float(D) / len(a)))): A.extend(a[:D if (D - i * len(a)) >= len(a) else D - i * len(a)])
			A = asarray(A)
	return A

def objects2array(objs):
	r"""Convert `Iterable` array or list to `NumPy` array.

	Args:
		objs (Iterable[Any]): Array or list to convert.

	Returns:
		numpy.ndarray: Array of objects.
	"""
	a = empty(len(objs), dtype=object)
	for i, e in enumerate(objs): a[i] = e
	return a

class Utility:
	r"""Base class with string mappings to benchmarks.

	Attributes:
		classes (Dict[str, Benchmark]): Mapping from stings to benchmark.
	"""
	def __init__(self):
		r"""

		"""
		self.classes = {
			'ackley': Ackley,
			'alpine1': Alpine1,
			'alpine2': Alpine2,
			'bentcigar': BentCigar,
			'chungReynolds': ChungReynolds,
			'cosinemixture': CosineMixture,
			'csendes': Csendes,
			'discus': Discus,
			'dixonprice': DixonPrice,
			'conditionedellptic': Elliptic,
			'elliptic': Elliptic,
			'expandedgriewankplusrosenbrock': ExpandedGriewankPlusRosenbrock,
			'expandedschaffer': ExpandedSchaffer,
			'griewank': Griewank,
			'happyCat': HappyCat,
			'hgbat': HGBat,
			'infinity': Infinity,
			'katsuura': Katsuura,
			'levy': Levy,
			'michalewicz': Michalewichz,
			'modifiedscwefel': ModifiedSchwefel,
			'perm': Perm,
			'pinter': Pinter,
			'powell': Powell,
			'qing': Qing,
			'quintic': Quintic,
			'rastrigin': Rastrigin,
			'ridge': Ridge,
			'rosenbrock': Rosenbrock,
			'salomon': Salomon,
			'schaffer2': SchafferN2,
			'schaffer4': SchafferN4,
			'schumerSteiglitz': SchumerSteiglitz,
			'schwefel': Schwefel,
			'schwefel221': Schwefel221,
			'schwefel222': Schwefel222,
			'sphere': Sphere,
			'sphere2': Sphere2,
			'sphere3': Sphere3,
			'step': Step,
			'step2': Step2,
			'step3': Step3,
			'stepint': Stepint,
			'styblinskiTang': StyblinskiTang,
			'sumSquares': SumSquares,
			'trid': Trid,
			'weierstrass': Weierstrass,
			'whitley': Whitley,
			'zakharov': Zakharov
		}

	def get_benchmark(self, benchmark):
		r"""Get the optimization problem.

		Arguments:
			benchmark (Union[str, Benchmark]): String or class that represents the optimization problem.

		Returns:
			Benchmark: Optimization function with limits.
		"""
		if not isinstance(benchmark, str) and not callable(benchmark): return benchmark
		elif benchmark in self.classes: return self.classes[benchmark]()
		else: raise TypeError('Passed benchmark is not defined!')

	@classmethod
	def __raiseLowerAndUpperNotDefined(cls):
		r"""Trow exception if lower and upper bounds are not defined in benchmark.

		Raises:
			TypeError: Type error.
		"""
		raise TypeError('Upper and Lower value must be defined!')

class OptimizationType(Enum):
	r"""Enum representing type of optimization.

	Attributes:
		MINIMIZATION (int): Represents minimization problems and is default optimization type of all algorithms.
		MAXIMIZATION (int): Represents maximization problems.
	"""
	MINIMIZATION = 1.0
	MAXIMIZATION = -1.0

def limitRepair(x, Lower, Upper, **kwargs):
	r"""Repair solution and put the solution in the random position inside of the bounds of problem.

	Arguments:
		x (numpy.ndarray): Solution to check and repair if needed.
		Lower (numpy.ndarray): Lower bounds of search space.
		Upper (numpy.ndarray): Upper bounds of search space.
		kwargs (Dict[str, Any]): Additional arguments.

	Returns:
		numpy.ndarray: Solution in search space.
	"""
	ir = where(x < Lower)
	x[ir] = Lower[ir]
	ir = where(x > Upper)
	x[ir] = Lower[ir]
	return x

def limitInversRepair(x, Lower, Upper, **kwargs):
	r"""Repair solution and put the solution in the random position inside of the bounds of problem.

	Arguments:
		x (numpy.ndarray): Solution to check and repair if needed.
		Lower (numpy.ndarray): Lower bounds of search space.
		Upper (numpy.ndarray): Upper bounds of search space.
		kwargs (Dict[str, Any]): Additional arguments.

	Returns:
		numpy.ndarray: Solution in search space.
	"""
	ir = where(x < Lower)
	x[ir] = Upper[ir]
	ir = where(x > Upper)
	x[ir] = Lower[ir]
	return x

def wangRepair(x, Lower, Upper, **kwargs):
	r"""Repair solution and put the solution in the random position inside of the bounds of problem.

	Arguments:
		x (numpy.ndarray): Solution to check and repair if needed.
		Lower (numpy.ndarray): Lower bounds of search space.
		Upper (numpy.ndarray): Upper bounds of search space.
		kwargs (Dict[str, Any]): Additional arguments.

	Returns:
		numpy.ndarray: Solution in search space.
	"""
	ir = where(x < Lower)
	x[ir] = amin([Upper[ir], 2 * Lower[ir] - x[ir]], axis=0)
	ir = where(x > Upper)
	x[ir] = amax([Lower[ir], 2 * Upper[ir] - x[ir]], axis=0)
	return x

def randRepair(x, Lower, Upper, rnd=rand, **kwargs):
	r"""Repair solution and put the solution in the random position inside of the bounds of problem.

	Arguments:
		x (numpy.ndarray): Solution to check and repair if needed.
		Lower (numpy.ndarray): Lower bounds of search space.
		Upper (numpy.ndarray): Upper bounds of search space.
		rnd (mtrand.RandomState): Random generator.
		kwargs (Dict[str, Any]): Additional arguments.

	Returns:
		numpy.ndarray: Fixed solution.
	"""
	ir = where(x < Lower)
	x[ir] = rnd.uniform(Lower[ir], Upper[ir])
	ir = where(x > Upper)
	x[ir] = rnd.uniform(Lower[ir], Upper[ir])
	return x

def reflectRepair(x, Lower, Upper, **kwargs):
	r"""Repair solution and put the solution in search space with reflection of how much the solution violates a bound.

	Args:
		x (numpy.ndarray): Solution to be fixed.
		Lower (numpy.ndarray): Lower bounds of search space.
		Upper (numpy.ndarray): Upper bounds of search space.
		kwargs (Dict[str, Any]): Additional arguments.

	Returns:
		numpy.ndarray: Fix solution.
	"""
	ir = where(x > Upper)
	x[ir] = Lower[ir] + x[ir] % (Upper[ir] - Lower[ir])
	ir = where(x < Lower)
	x[ir] = Lower[ir] + x[ir] % (Upper[ir] - Lower[ir])
	return x

class Task(Utility):
	r"""Class representing problem to solve with optimization.

	Date:
		2019

	Author:
		Klemen Berkovič

	Attributes:
		D (int): Dimension of the problem.
		Lower (numpy.ndarray): Lower bounds of the problem.
		Upper (numpy.ndarray): Upper bounds of the problem.
		bRange (numpy.ndarray): Search range between upper and lower limits.
		optType (OptimizationType): Optimization type to use.

	See Also:
		* :class:`NiaPy.util.Utility`
	"""
	D = 0
	benchmark = None
	Lower, Upper, bRange = inf, inf, inf
	optType = OptimizationType.MINIMIZATION

	def __init__(self, D=0, optType=OptimizationType.MINIMIZATION, benchmark=None, Lower=None, Upper=None, frepair=limitRepair, **kwargs):
		r"""Initialize task class for optimization.

		Arguments:
			D (Optional[int]): Number of dimensions.
			optType (Optional[OptimizationType]): Set the type of optimization.
			benchmark (Union[str, Benchmark]): Problem to solve with optimization.
			Lower (Optional[numpy.ndarray]): Lower limits of the problem.
			Upper (Optional[numpy.ndarray]): Upper limits of the problem.
			frepair (Optional[Callable[[numpy.ndarray, numpy.ndarray, numpy.ndarray, Dict[str, Any]], numpy.ndarray]]): Function for reparing individuals components to desired limits.

		See Also:
			* `func`:NiaPy.util.Utility.__init__`
			* `func`:NiaPy.util.Utility.repair`
		"""
		Utility.__init__(self)
		# dimension of the problem
		self.D = D
		# set optimization type
		self.optType = optType
		# set optimization function
		self.benchmark = self.get_benchmark(benchmark) if benchmark is not None else None
		if self.benchmark is not None: self.Fun = self.benchmark.function() if self.benchmark is not None else None
		# set Lower limits
		if Lower is not None: self.Lower = fullArray(Lower, self.D)
		elif Lower is None and benchmark is not None: self.Lower = fullArray(self.benchmark.Lower, self.D)
		else: self.Lower = fullArray(0, self.D)
		# set Upper limits
		if Upper is not None: self.Upper = fullArray(Upper, self.D)
		elif Upper is None and benchmark is not None: self.Upper = fullArray(self.benchmark.Upper, self.D)
		else: self.Upper = fullArray(0, self.D)
		# set range
		self.bRange = self.Upper - self.Lower
		# set repair function
		self.frepair = frepair

	def dim(self):
		r"""Get the number of dimensions.

		Returns:
			int: Dimension of problem optimizing.
		"""
		return self.D

	def bcLower(self):
		r"""Get the array of lower bound constraint.

		Returns:
			numpy.ndarray: Lower bound.
		"""
		return self.Lower

	def bcUpper(self):
		r"""Get the array of upper bound constraint.

		Returns:
			numpy.ndarray: Upper bound.
		"""
		return self.Upper

	def bcRange(self):
		r"""Get the range of bound constraint.

		Returns:
			numpy.ndarray: Range between lower and upper bound.
		"""
		return self.Upper - self.Lower

	def repair(self, x, rnd=rand):
		r"""Repair solution and put the solution in the random position inside of the bounds of problem.

		Arguments:
			x (numpy.ndarray): Solution to check and repair if needed.
			rnd (mtrand.RandomState): Random number generator.

		Returns:
			numpy.ndarray: Fixed solution.

		See Also:
			* :func:`NiaPy.util.limitRepair`
			* :func:`NiaPy.util.limitInversRepair`
			* :func:`NiaPy.util.wangRepair`
			* :func:`NiaPy.util.randRepair`
			* :func:`NiaPy.util.reflectRepair`
		"""
		return self.frepair(x, self.Lower, self.Upper, rnd=rnd)

	def nextIter(self):
		r"""Increments the number of algorithm iterations."""

	def start(self):
		r"""Start stopwatch."""

	def eval(self, A):
		r"""Evaluate the solution A.

		Arguments:
			A (numpy.ndarray): Solution to evaluate.

		Returns:
			float: Fitness/function values of solution.
		"""
		return self.Fun(self.D, A) * self.optType.value

	def isFeasible(self, A):
		r"""Check if the solution is feasible.

		Arguments:
			A (Union[numpy.ndarray, Individual]): Solution to check for feasibility.

		Returns:
			bool: `True` if solution is in feasible space else `False`.
		"""
		return False not in (A >= self.Lower) and False not in (A <= self.Upper)

	def stopCond(self):
		r"""Check if optimization task should stop.

		Returns:
			bool: `True` if stopping condition is meet else `False`.
		"""
		return False

class CountingTask(Task):
	r"""Optimization task with added counting of function evaluations and algorithm iterations/generations.

	Attributes:
		Iters (int): Number of algorithm iterations/generations.
		Evals (int): Number of function evaluations.

	See Also:
		* :class:`NiaPy.util.Task`
	"""

	def __init__(self, **kwargs):
		r"""Initialize counting task.

		Args:
			**kwargs (Dict[str, Any]): Additional arguments.

		See Also:
			* :func:`NiaPy.util.Task.__init__`
		"""
		Task.__init__(self, **kwargs)
		self.Iters, self.Evals = 0, 0

	def eval(self, A):
		r"""Evaluate the solution A.

		This function increments function evaluation counter `self.Evals`.

		Arguments:
			A (numpy.ndarray): Solutions to evaluate.

		Returns:
			float: Fitness/function values of solution.

		See Also:
			* :func:`NiaPy.util.Task.eval`
		"""
		r = Task.eval(self, A)
		self.Evals += 1
		return r

	def evals(self):
		r"""Get the number of evaluations made.

		Returns:
			int: Number of evaluations made.
		"""
		return self.Evals

	def iters(self):
		r"""Get the number of algorithm iteratins made.

		Returns:
			int: Number of generations/iterations made by algorithm.
		"""
		return self.Iters

	def nextIter(self):
		r"""Increases the number of algorithm iterations made.

		This function increments number of algorithm iterations/generations counter `self.Iters`.
		"""
		self.Iters += 1

class StoppingTask(CountingTask):
	r"""Optimization task with implemented checking for stopping criterias.

	Attributes:
		nGEN (int): Maximum number of algorithm iterations/generations.
		nFES (int): Maximum number of function evaluations.
		refValue (float): Reference function/fitness values to reach in optimization.
		x (numpy.ndarray): Best found individual.
		x_f (float): Best found individual function/fitness value.

	See Also:
		* :class:`NiaPy.util.CountingTask`
	"""

	def __init__(self, nFES=inf, nGEN=inf, refValue=None, **kwargs):
		r"""Initialize task class for optimization.

		Arguments:
			nFES (Optional[int]): Number of function evaluations.
			nGEN (Optional[int]): Number of generations or iterations.
			refValue (Optional[float]): Reference value of function/fitness function.

		See Also:
			* :func:`NiaPy.util.CountingTask.__init__`
		"""
		CountingTask.__init__(self, **kwargs)
		self.refValue = (-inf if refValue is None else refValue)
		self.x, self.x_f = None, inf
		self.nFES, self.nGEN = nFES, nGEN

	def eval(self, A):
		r"""Evaluate solution.

		Args:
			A (numpy.ndarray): Solution to evaluate.

		Returns:
			float: Fitness/function value of solution.

		See Also:
			* :func:`NiaPy.util.StoppingTask.stopCond`
			* :func:`NiaPy.util.CountingTask.eval`
		"""
		if self.stopCond(): return inf * self.optType.value
		x_f = CountingTask.eval(self, A)
		if x_f < self.x_f: self.x_f = x_f
		return x_f

	def stopCond(self):
		r"""Check if stopping condition reached.

		Returns:
			bool: `True` if number of function evaluations or number of algorithm iterations/generations or reference values is reach else `False`
		"""
		return (self.Evals >= self.nFES) or (self.Iters >= self.nGEN) or (self.refValue > self.x_f)

	def stopCondI(self):
		r"""Check if stopping condition reached and increase number of iterations.

		Returns:
			bool: `True` if number of function evaluations or number of algorithm iterations/generations or reference values is reach else `False`.

		See Also:
			* :func:`NiaPy.util.StoppingTask.stopCond`
			* :func:`NiaPy.util.CountingTask.nextIter`
		"""
		r = self.stopCond()
		CountingTask.nextIter(self)
		return r

class ThrowingTask(StoppingTask):
	r"""Task that throw exceptions when stopping condition is meet.

	See Also:
		* :class:`NiaPy.util.StoppingTask`
	"""
	def __init__(self, **kwargs):
		r"""Initialize optimization task.

		Args:
			**kwargs (Dict[str, Any]): Additional arguments.

		See Also:
			* :func:`NiaPy.util.StoppingTask.__init__`
		"""
		StoppingTask.__init__(self, **kwargs)

	def stopCondE(self):
		r"""Throw exception for the given stopping condition.

		Raises:
			* FesException: Thrown when the number of function/fitness evaluations is reached.
			* GenException: Thrown when the number of algorithms generations/iterations is reached.
			* RefException: Thrown when the reference values is reached.
			* TimeException: Thrown when algorithm exceeds time run limit.
		"""
		# dtime = datetime.now() - self.startTime
		if self.Evals >= self.nFES: raise FesException()
		if self.Iters >= self.nGEN: raise GenException()
		# if self.runTime is not None and self.runTime >= dtime: raise TimeException()
		if self.refValue >= self.x_f: raise RefException()

	def eval(self, A):
		r"""Evaluate solution.

		Args:
			A (numpy.ndarray): Solution to evaluate.

		Returns:
			float: Function/fitness values of solution.

		See Also:
			* :func:`NiaPy.util.ThrowingTask.stopCondE`
			* :func:`NiaPy.util.StoppingTask.eval`
		"""
		self.stopCondE()
		return StoppingTask.eval(self, A)

class MoveTask(StoppingTask):
	def __init__(self, o=None, fo=None, M=None, fM=None, optF=None, **kwargs):
		r"""Initialize task class for optimization.

		Arguments:
			o (numpy.ndarray[Union[float, int]]): Array for shifting.
			of (Callable[numpy.ndarray[Union[float, int]]]): Function applied on shifted input.
			M (numpy.ndarray[Union[float, int]]): Matrix for rotating.
			fM (Callable[numpy.ndarray[Union[float, int]]]): Function applied after rotating.

		See Also:
			* :func:`NiaPy.util.StoppingTask.__init__`
		"""
		StoppingTask.__init__(self, **kwargs)
		self.o = o if isinstance(o, ndarray) or o is None else asarray(o)
		self.M = M if isinstance(M, ndarray) or M is None else asarray(M)
		self.fo, self.fM, self.optF = fo, fM, optF

	def eval(self, A):
		r"""Evaluate the solution.

		Args:
			A (numpy.ndarray): Solution to evaluate

		Returns:
			float: Fitness/function value of solution.

		See Also:
			* :func:`NiaPy.util.StoppingTask.stopCond`
			* :func:`NiaPy.util.StoppingTask.eval`
		"""
		if self.stopCond(): return inf * self.optType.value
		X = A - self.o if self.o is not None else A
		X = self.fo(X) if self.fo is not None else X
		X = dot(X, self.M) if self.M is not None else X
		X = self.fM(X) if self.fM is not None else X
		r = StoppingTask.eval(self, X) + (self.optF if self.optF is not None else 0)
		if r <= self.x_f: self.x, self.x_f = A, r
		return r

class ScaledTask(Task):
	r"""Scaled task.

	Attributes:
		_task (Task): Optimization task with evaluation function.
		Lower (numpy.ndarray): Scaled lower limit of search space.
		Upper (numpy.ndarray): Scaled upper limit of search space.

	See Also:
		* :class:`NiaPy.util.Task`
	"""
	def __init__(self, task, Lower, Upper, **kwargs):
		r"""Initialize scaled task.

		Args:
			task (Task): Optimization task to scale to new bounds.
			Lower (Union[float, int, numpy.ndarray]): New lower bounds.
			Upper (Union[float, int, numpy.ndarray]): New upper bounds.
			**kwargs (Dict[str, Any]): Additional arguments.

		See Also:
			* :func:`NiaPy.util.fullArray`
		"""
		Task.__init__(self)
		self._task = task
		self.D = self._task.D
		self.Lower, self.Upper = fullArray(Lower, self.D), fullArray(Upper, self.D)
		self.bRange = fabs(Upper - Lower)

	def stopCond(self):
		r"""Test for stopping condition.

		This function uses `self._task` for checking the stopping criteria.

		Returns:
			bool: `True` if stopping condition is meet else `False`.
		"""
		return self._task.stopCond()

	def stopCondI(self):
		r"""Test for stopping condition and increments the number of algorithm generations/iterations.

		This function uses `self._task` for checking the stopping criteria.

		Returns:
			bool: `True` if stopping condition is meet else `False`.
		"""
		return self._task.stopCondI()

	def eval(self, A):
		r"""Evaluate solution.

		Args:
			A (numpy.ndarray): Solution for calculating function/fitness value.

		Returns:
			float: Function values of solution.
		"""
		return self._task.eval(A)

	def evals(self):
		r"""Get the number of function evaluations.

		Returns:
			int: Number of function evaluations.
		"""
		return self._task.evals()

	def iters(self):
		r"""Get the number of algorithms generations/iterations.

		Returns:
			int: Number of algorithms generations/iterations.
		"""
		return self._task.iters()

	def nextIter(self):
		r"""Increment the number of iterations/generations.

		Function uses `self._task` to increment number of generations/iterations.
		"""
		self._task.nextIter()

class TaskConvPrint(StoppingTask):
	r"""Task class with printing out new global best solutions found.

	Attributes:
		xb (numpy.ndarray): Global best solution.
		xb_f (float): Global best function/fitness values.

	See Also:
		* :class:`NiaPy.util.StoppingTask`
	"""
	def __init__(self, **kwargs):
		r"""Initialize TaskConvPrint class.

		Args:
			**kwargs (Dict[str, Any]): Additional arguments.

		See Also:
			* :func:`NiaPy.util.StoppingTask.__init__`
		"""
		StoppingTask.__init__(self, **kwargs)
		self.xb, self.xb_f = None, inf

	def eval(self, A):
		r"""Evaluate solution.

		Args:
			A (nupy.ndarray): Solution to evaluate.

		Returns:
			float: Function/Fitness values of solution.

		See Also:
			* :func:`NiaPy.util.StoppingTask.eval`
		"""
		x_f = StoppingTask.eval(self, A)
		if self.x_f != self.xb_f:
			self.xb, self.xb_f = A, x_f
			logger.info('nFES:%d nGEN:%d => %s -> %s' % (self.Evals, self.Iters, self.xb, self.xb_f * self.optType.value))
		return x_f

class TaskConvSave(StoppingTask):
	r"""Task class with logging of function evaluations need to reach some function vale.

	Attributes:
		evals (List[int]): List of ints representing when the new global best was found.
		x_f_vals (List[float]): List of floats representing function/fitness values found.

	See Also:
		* :class:`NiaPy.util.StoppingTask`
	"""
	def __init__(self, **kwargs):
		r"""Initialize TaskConvSave class.

		Args:
			**kwargs (Dict[str, Any]): Additional arguments.

		See Also:
			* :func:`NiaPy.util.StoppingTask.__init__`
		"""
		StoppingTask.__init__(self, **kwargs)
		self.evals = []
		self.x_f_vals = []

	def eval(self, A):
		r"""Evaluate solution.

		Args:
			A (numpy.ndarray): Individual/solution to evaluate.

		Returns:
			float: Function/fitness values of individual.

		See Also:
			* :func:`SNiaPy.util.toppingTask.eval`
		"""
		x_f = StoppingTask.eval(self, A)
		if x_f <= self.x_f:
			self.evals.append(self.Evals)
			self.x_f_vals.append(x_f)
		return x_f

	def return_conv(self):
		r"""Get values of x and y axis for plotting covariance graph.

		Returns:
			Tuple[List[int], List[float]]:
				1. List of ints of function evaluations.
				2. List of ints of function/fitness values.
		"""
		return self.evals, self.x_f_vals

class TaskConvPlot(TaskConvSave):
	r"""Task class with ability of showing convergence graph.

	Attributes:
		iters (List[int]): List of ints representing when the new global best was found.
		x_fs (List[float]): List of floats representing function/fitness values found.

	See Also:
		* :class:`NiaPy.util.StoppingTask`
	"""
	def __init__(self, **kwargs):
		r"""TODO.

		Args:
			**kwargs (Dict[str, Any]): Additional arguments.

		See Also:
			* :func:`NiaPy.util.StoppingTask.__init__`
		"""
		StoppingTask.__init__(self, **kwargs)
		self.fig = plt.figure()
		self.ax = self.fig.subplots(nrows=1, ncols=1)
		self.ax.set_xlim(0, self.nFES)
		self.line, = self.ax.plot(self.iters, self.x_fs, animated=True)
		self.ani = anim.FuncAnimation(self.fig, self.updatePlot, blit=True)
		self.showPlot()

	def eval(self, A):
		r"""Evaluate solution.

		Args:
			A (numpy.ndarray): Solution to evaluate.

		Returns:
			float: Fitness/function values of solution.
		"""
		x_f = StoppingTask.eval(self, A)
		if not self.x_f_vals: self.x_f_vals.append(x_f)
		elif x_f < self.x_f_vals[-1]: self.x_f_vals.append(x_f)
		else: self.x_f_vals.append(self.x_f_vals[-1])
		self.evals.append(self.Evals)
		return x_f

	def showPlot(self):
		r"""Animation updating function."""
		plt.show(block=False)
		plt.pause(0.001)

	def updatePlot(self, frame):
		r"""Update mathplotlib figure.

		Args:
			frame (): TODO

		Returns:
			Tuple[List[float], Any]:
				1. Line
		"""
		if self.x_f_vals:
			max_fs, min_fs = self.x_f_vals[0], self.x_f_vals[-1]
			self.ax.set_ylim(min_fs + 1, max_fs + 1)
			self.line.set_data(self.evals, self.x_f_vals)
		return self.line,

class TaskComposition(MoveTask):
	def __init__(self, benchmarks=None, rho=None, lamb=None, bias=None, **kwargs):
		r"""Initialize of composite function problem.

		Arguments:
			benchmarks (List[Benchmark]): Optimization function to use in composition
			delta (numpy.ndarray[float]): TODO
			lamb (numpy.ndarray[float]): TODO
			bias (numpy.ndarray[float]): TODO

		See Also:
			* :func:`NiaPy.util.MoveTask.__init__`

		TODO:
			Class is a work in progress.
		"""
		MoveTask.__init__(self, **kwargs)

	def eval(self, A):
		r"""TODO.

		Args:
			A:

		Returns:
			float:

		Todo:
			Usage of multiple functions on the same time
		"""
		return inf

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
