# encoding=utf8
# pylint: disable=mixed-indentation, line-too-long, bad-continuation, multiple-statements, singleton-comparison
"""Implementation of benchmarks utility function."""

from numpy import ndarray, asarray, full, inf, dot
from . import Rastrigin, Rosenbrock, Griewank, Sphere, Ackley, Schwefel, Schwefel221, Schwefel222, Whitley, Alpine1, Alpine2, HappyCat, Ridge, ChungReynolds, Csendes, Pinter, Qing, Quintic, Salomon, SchumerSteiglitz, Step, Step2, Step3, Stepint, SumSquares, StyblinskiTang, BentCigar, Discus, Elliptic, ExpandedGriewankPlusRosenbrock, HGBat, Katsuura, ExpandedScaffer, ModifiedSchwefel, Weierstrass

__all__ = ['Utility', 'Task']

class Utility(object):
	def __init__(self):
		self.classes = {
			'ackley': Ackley,
			'alpine1': Alpine1,
			'alpine2': Alpine2,
			'bentcigar': BentCigar,
			'chungReynolds': ChungReynolds,
			'csendes': Csendes,
			'discus': Discus,
			'conditionedellptic': Elliptic,
			'elliptic': Elliptic,
			'expandedgriewankplusrosenbrock': ExpandedGriewankPlusRosenbrock,
			'expandedscaffer': ExpandedScaffer,
			'griewank': Griewank,
			'happyCat': HappyCat,
			'hgbat': HGBat,
			'katsuura': Katsuura,
			'modifiedscwefel': ModifiedSchwefel,
			'pinter': Pinter,
			'quing': Qing,
			'quintic': Quintic,
			'rastrigin': Rastrigin,
			'ridge': Ridge,
			'rosenbrock': Rosenbrock,
			'salomon': Salomon,
			'schumerSteiglitz': SchumerSteiglitz,
			'schwefel': Schwefel,
			'schwefel221': Schwefel221,
			'schwefel222': Schwefel222,
			'sphere': Sphere,
			'step': Step,
			'step2': Step2,
			'step3': Step3,
			'stepint': Stepint,
			'styblinskiTang': StyblinskiTang,
			'sumSquares': SumSquares,
			'weierstrass': Weierstrass,
			'whitley': Whitley
		}

	def get_benchmark(self, benchmark):
		if not isinstance(benchmark, ''.__class__):
			if not callable(benchmark): return benchmark
			else:	raise TypeError('Passed benchmark is not defined!')
		else:
			if benchmark in self.classes:	return self.classes[benchmark]()
			else:	raise TypeError('Passed benchmark is not defined!')

	@classmethod
	def __raiseLowerAndUpperNotDefined(cls): raise TypeError('Upper and Lower value must be defined!')

class Task(Utility):
	def __init__(self, D, nFES, nGEN, benchmark=None, o=None, fo=None, M=None, fM=None, optF=None):
		r"""Initialize task class for optimization.

		Arguments:
		D {integer} -- Number of dimensions
		nFES {integer} -- Number of function evaluations
		nGEN {integer} -- Number of generation or iterations
		benchmark {class} or {string} -- Problem to solve
		o {array} -- Array for shifting
		of {function} -- Function applied on shifted input
		M {matrix} -- Matrix for rotating
		fM {function} -- Function applied after rotating
		optF {real} -- Value added to benchmark function return
		"""
		super(Task, self).__init__()
		self.benchmark = self.get_benchmark(benchmark) if benchmark != None else None
		self.D = D  # dimension of the problem
		self.Iters, self.nGEN = 0, nGEN if nGEN != None else 10000
		self.Evals, self.nFES = 0, nFES
		self.Fun = self.benchmark.function() if benchmark != None else None
		self.o = o if isinstance(o, ndarray) or o == None else asarray(o)
		self.M = M if isinstance(M, ndarray) or M == None else asarray(M)
		self.fo, self.fM, self.optF = fo, fM, optF
		self.__initBounds()

	def __initBounds(self):
		Lower, Upper = self.benchmark.Lower, self.benchmark.Upper
		if isinstance(Lower, (int, float)): self.Lower = full(self.D, Lower)
		else: self.Lower = Lower if isinstance(Lower, ndarray) else asarray(Lower)
		if isinstance(Upper, (int, float)): self.Upper = full(self.D, Upper)
		else: self.Upper = Upper if isinstance(Upper, ndarray) else asarray(Upper)
		self.bRange = self.Upper - self.Lower

	def stopCond(self): return self.Evals >= self.nFES or (False if self.nGEN == None else self.Iters >= self.nGEN)

	def eval(self, A):
		self.Evals += 1
		if self.stopCond() or not self.isFisible(A): return inf
		X = A - self.o if self.o != None else A
		X = self.fo(X) if self.fo != None else X
		X = dot(X, self.M) if self.M != None else X
		X = self.fM(X) if self.fM != None else X
		return self.Fun(self.D, X) + (self.optF if self.optF != None else 0)

	def nextIter(self): self.Iters += 1

	def isFisible(self, A): return (False if True in (A < self.Lower) else True) or (False if True in (A > self.Upper) else True)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
