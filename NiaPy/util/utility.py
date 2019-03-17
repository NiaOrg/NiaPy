# encoding=utf8
# pylint: disable=line-too-long, mixed-indentation, bad-continuation,multiple-statements, unused-argument, no-self-use, trailing-comma-tuple, logging-not-lazy, no-else-return, dangerous-default-value, assignment-from-no-return, superfluous-parens

"""Implementation of benchmarks utility function."""
import logging
# from datetime import datetime
from enum import Enum
from numpy import ndarray, asarray, full, inf, dot, where, random as rand, fabs, ceil, amin, amax
from matplotlib import pyplot as plt, animation as anim
from NiaPy.benchmarks import Rastrigin, Rosenbrock, Griewank, Sphere, Ackley, Schwefel, Schwefel221, Schwefel222, Whitley, Alpine1, Alpine2, HappyCat, Ridge, ChungReynolds, Csendes, Pinter, Qing, Quintic, Salomon, SchumerSteiglitz, Step, Step2, Step3, Stepint, SumSquares, StyblinskiTang, BentCigar, Discus, Elliptic, ExpandedGriewankPlusRosenbrock, HGBat, Katsuura, ExpandedSchaffer, ModifiedSchwefel, Weierstrass, Michalewichz, Levy, Sphere2, Sphere3, Trid, Perm, Zakharov, DixonPrice, Powell, CosineMixture, Infinity, SchafferN2, SchafferN4
from NiaPy.util.exception import FesException, GenException, RefException #TimeException,

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
	'TaskConvPrint',
	'TaskConvPlot',
	'TaskConvSave',
	'fullArray',
	'TaskComposition',
	'OptimizationType',
	'ScaledTask'
]

def fullArray(a, D):
	r"""Fill or create array of length D, from value or value form a.

	Arguments:
	a {integer} or {real} or {list} or {ndarray} -- Input values for fill
	D {integer} -- Length of new array
	"""
	A = []
	if isinstance(a, (int, float)):	A = full(D, a)
	elif isinstance(a, (ndarray, list)):
		if len(a) == D: A = a if isinstance(a, ndarray) else asarray(a)
		elif len(a) > D: A = a[:D] if isinstance(a, ndarray) else asarray(a[:D])
		else:
			for i in range(int(ceil(float(D) / len(a)))): A.extend(a[:D if (D - i * len(a)) >= len(a) else D - i * len(a)])
			A = asarray(A)
	return A

class Utility:
	def __init__(self):
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
		benchmark {string} or {class} -- String or class that represents the optimization problem
		"""
		if not isinstance(benchmark, str) and not callable(benchmark): return benchmark
		elif benchmark in self.classes: return self.classes[benchmark]()
		else: raise TypeError('Passed benchmark is not defined!')

	@classmethod
	def __raiseLowerAndUpperNotDefined(cls): raise TypeError('Upper and Lower value must be defined!')

class OptimizationType(Enum):
	MINIMIZATION = 1.0
	MAXIMIZATION = -1.0

def limitRepair(x, Lower, Upper, **kwargs):
	r"""Repair solution and put the solution in the random position inside of the bounds of problem.

	Arguments:
	x {array} -- solution to check and repair if needed
	"""
	ir = where(x < Lower)
	x[ir] = Lower[ir]
	ir = where(x > Upper)
	x[ir] = Lower[ir]
	return x

def limitInversRepair(x, Lower, Upper, **kwargs):
	r"""Repair solution and put the solution in the random position inside of the bounds of problem.

	Arguments:
	x {array} -- solution to check and repair if needed
	"""
	ir = where(x < Lower)
	x[ir] = Upper[ir]
	ir = where(x > Upper)
	x[ir] = Lower[ir]
	return x

def wangRepair(x, Lower, Upper, **kwargs):
	r"""Repair solution and put the solution in the random position inside of the bounds of problem.

	Arguments:
	x {array} -- solution to check and repair if needed
	"""
	ir = where(x < Lower)
	x[ir] = amin([Upper[ir], 2 * Lower[ir] - x[ir]], axis=0)
	ir = where(x > Upper)
	x[ir] = amax([Lower[ir], 2 * Upper[ir] - x[ir]], axis=0)
	return x

def randRepair(x, Lower, Upper, rnd=rand, **kwargs):
	r"""Repair solution and put the solution in the random position inside of the bounds of problem.

	Arguments:
	x {array} -- solution to check and repair if needed
	Lower {array}
	Upper {array}
	rnd {function}
	"""
	ir = where(x < Lower)
	x[ir] = rnd.uniform(Lower[ir], Upper[ir])
	ir = where(x > Upper)
	x[ir] = rnd.uniform(Lower[ir], Upper[ir])
	return x

class Task(Utility):
	D = 0
	benchmark = None
	Lower, Upper, bRange = inf, inf, inf
	optType = OptimizationType.MINIMIZATION

	def __init__(self, D=0, optType=OptimizationType.MINIMIZATION, benchmark=None, Lower=None, Upper=None, frepair=randRepair, **kwargs):
		r"""Initialize task class for optimization.

		Arguments:
		D {integer} -- Number of dimensions
		optType {OptimizationType} -- Set the type of optimization
		benchmark {class} or {string} -- Problem to solve
		Lower {array} or {real} -- Lower limits of the problem
		Upper {array} or {real} -- Upper limits of the problem
		frepair {function} -- Function for reparing individuals components to desired limits
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
		r"""Get the number of dimensions."""
		return self.D

	def bcLower(self):
		r"""Get the array of lower bound constraint."""
		return self.Lower

	def bcUpper(self):
		r"""Get the array of upper bound constraint."""
		return self.Upper

	def bcRange(self):
		r"""Get the range of bound constraint."""
		return self.Upper - self.Lower

	def repair(self, x, rnd=rand):
		r"""Repair solution and put the solution in the random position inside of the bounds of problem.

		Arguments:
		x {array} -- solution to check and repair if needed
		"""
		return self.frepair(x, self.Lower, self.Upper, rnd)

	def nextIter(self):
		r"""Increments the number of algorithm iterations."""
		pass

	def start(self):
		r"""Start stopwatch."""
		pass

	def eval(self, A):
		r"""Evaluate the solution A.

		Arguments:
		A {array} -- Solution to evaluate
		"""
		return self.Fun(self.D, A) * self.optType.value

	def isFeasible(self, A):
		r"""Check if the solution is feasible.

		Arguments:
		A {array} -- Solution to check for feasibility
		"""
		return not False in (A > self.Lower) and not False in (A < self.Upper)

	def stopCond(self):
		r"""TODO."""
		return False

class CountingTask(Task):
	Iters, Evals = 0, 0

	def __init__(self, **kwargs):
		Task.__init__(self, **kwargs)

	def eval(self, A):
		r"""Evaluate the solution A.

		Arguments:
		A {array} -- Solutions to evaluate
		"""
		self.Evals += 1
		return Task.eval(self, A)

	def evals(self):
		r"""Get the number of evaluations made."""
		return self.Evals

	def iters(self):
		r"""Get the number of algorithm iteratins made."""
		return self.Iters

	def nextIter(self):
		r"""Increases the number of algorithm iterations made."""
		self.Iters += 1

	def stopCondI(self):
		r"""Check if stoping condition and increment number of generations/iterations."""
		self.Iters += 1
		return Task.stopCondI(self)

class StopingTask(CountingTask):
	nGEN, nFES = inf, inf
	refValue, x, x_f = inf, None, inf

	def __init__(self, nFES=inf, nGEN=inf, refValue=None, **kwargs):
		r"""Initialize task class for optimization.

		Arguments:
		nFES {integer} -- Number of function evaluations
		nGEN {integer} -- Number of generations or iterations
		"""
		CountingTask.__init__(self, **kwargs)
		self.refValue = (-inf if refValue is None else refValue)
		self.x_f = inf
		self.nFES, self.nGEN = nFES, nGEN

	def eval(self, A):
		if self.stopCond(): return inf * self.optType.value
		x_f = CountingTask.eval(self, A)
		if x_f < self.x_f: self.x_f = x_f
		return x_f

	def stopCond(self):
		r"""Check if stopping condition reached."""
		return (self.Evals >= self.nFES) or (self.Iters >= self.nGEN) or (self.refValue > self.x_f)

	def stopCondI(self):
		r"""Check if stopping condition reached and increase number of iterations."""
		CountingTask.stopCondI(self)
		return self.stopCond()

class ThrowingTask(StopingTask):
	def __init__(self, **kwargs):
		StopingTask.__init__(self, **kwargs)

	def stopCondE(self):
		r"""Throw exception for the given stopping condition."""
		# dtime = datetime.now() - self.startTime
		if self.Evals >= self.nFES: raise FesException()
		if self.Iters >= self.nGEN: raise GenException()
		# if self.runTime is not None and self.runTime >= dtime: raise TimeException()
		if self.refValue >= self.x_f: raise RefException()

	def eval(self, A):
		self.stopCondE()
		return StopingTask.eval(self, A)

class MoveTask(StopingTask):
	def __init__(self, o=None, fo=None, M=None, fM=None, optF=None, **kwargs):
		r"""Initialize task class for optimization.

		Arguments:
		o {array} -- Array for shifting
		of {function} -- Function applied on shifted input
		M {matrix} -- Matrix for rotating
		fM {function} -- Function applied after rotating
		"""
		StopingTask.__init__(self, **kwargs)
		self.o = o if isinstance(o, ndarray) or o is None else asarray(o)
		self.M = M if isinstance(M, ndarray) or M is None else asarray(M)
		self.fo, self.fM, self.optF = fo, fM, optF

	def eval(self, A):
		if self.stopCond(): return inf * self.optType.value
		X = A - self.o if self.o is not None else A
		X = self.fo(X) if self.fo is not None else X
		X = dot(X, self.M) if self.M is not None else X
		X = self.fM(X) if self.fM is not None else X
		r = StopingTask.eval(self, X) + (self.optF if self.optF is not None else 0)
		if r <= self.x_f: self.x, self.x_f = A, r
		return r

class ScaledTask(Task):
	def __init__(self, task, Lower, Upper, **kwargs):
		Task.__init__(self)
		self._task = task
		self.D = self._task.D
		self.Lower, self.Upper = fullArray(Lower, self.D), fullArray(Upper, self.D)
		self.bRange = fabs(Upper - Lower)

	def stopCond(self): return self._task.stopCond()

	def stopCondI(self): return self._task.stopCondI()

	def eval(self, A): return self._task.eval(A)

	def evals(self): return self._task.evals()

	def iters(self): return self._task.iters()

	def nextIter(self): self._task.nextIter()

	def isFeasible(self, A): return self._task.isFeasible(A)

class ScaledTaskE(ScaledTask):
	def __init__(self, **kwargs):
		ScaledTask.__init__(self, **kwargs)

	def eval(self, A):
		self.stopCond()
		return ScaledTask.eval(self, A)

class TaskConvPrint(StopingTask):
	xb, xb_f = inf, None

	def __init__(self, **kwargs):
		StopingTask.__init__(self, **kwargs)

	def eval(self, A):
		x_f = StopingTask.eval(self, A)
		if not self.x_f == self.xb_f:
			self.xb, self.xb_f = A, x_f
			logger.info('nFES:%d nGEN:%d => %s -> %s' % (self.Evals, self.Iters, self.xb, self.xb_f * self.optType.value))
		return x_f

class TaskConvSave(StopingTask):
	def __init__(self, **kwargs):
		StopingTask.__init__(self, **kwargs)
		self.evals = []
		self.x_f_vals = []

	def eval(self, A):
		x_f = StopingTask.eval(self, A)
		if x_f <= self.x_f:
			self.evals.append(self.Evals)
			self.x_f_vals.append(x_f)
		return x_f

	def return_conv(self): return self.evals, self.x_f_vals

class TaskConvPlot(StopingTask):
	def __init__(self, **kwargs):
		StopingTask.__init__(self, **kwargs)
		self.x_fs, self.iters = [], []
		self.fig = plt.figure()
		self.ax = self.fig.subplots(nrows=1, ncols=1)
		self.ax.set_xlim(0, self.nFES)
		self.line, = self.ax.plot(self.iters, self.x_fs, animated=True)
		self.ani = anim.FuncAnimation(self.fig, self.updatePlot, blit=True)
		self.showPlot()

	def eval(self, A):
		x_f = StopingTask.eval(self, A)
		if not self.x_fs: self.x_fs.append(x_f)
		elif x_f < self.x_fs[-1]: self.x_fs.append(x_f)
		else: self.x_fs.append(self.x_fs[-1])
		self.iters.append(self.Evals)
		return x_f

	def showPlot(self):
		plt.show(block=False)
		plt.pause(0.001)

	def updatePlot(self, frame):
		if self.x_fs:
			max_fs, min_fs = self.x_fs[0], self.x_fs[-1]
			self.ax.set_ylim(min_fs + 1, max_fs + 1)
			self.line.set_data(self.iters, self.x_fs)
		return self.line,

class TaskComposition(MoveTask):
	def __init__(self, benchmarks=None, rho=None, lamb=None, bias=None, **kwargs):
		r"""Initialize of composite function problem.

		Arguments:
		benchmarks {array} of {problems} -- optimization function to use in composition
		delta {array} of {real} --
		lamb {array} of {real} --
		bias {array} of {real} --
		"""
		MoveTask.__init__(self, **kwargs)

	def eval(self, A):
		# TODO Usage of multiple functions on the same time
		return inf

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
