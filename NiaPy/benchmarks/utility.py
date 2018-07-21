# encoding=utf8
# pylint: disable=mixed-indentation, line-too-long, bad-continuation, multiple-statements, singleton-comparison
"""Implementation of benchmarks utility function."""

import numpy as np
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
		if not isinstance(benchmark, ''.__class__): return benchmark
		else:
			if benchmark in self.classes:	return self.classes[benchmark]()
			else:	raise TypeError('Passed benchmark is not defined!')

	@classmethod
	def __raiseLowerAndUpperNotDefined(cls): raise TypeError('Upper and Lower value must be defined!')

class Task(Utility):
	def __init__(self, D, nFES, nGEN, benchmark=None, o=None, M=None):
		r"""__init__(self, D, nFES, nGEN, benchmark=None, o, M)
		D {integer} --
		nFES {integer} --
		nGEN {integer} --
		benchmark {class} or {string} --
		o {array} --
		M {matrix} --
		"""
		super().__init__()
		self.benchmark = self.get_benchmark(benchmark) if benchmark != None else None
		self.D = D  # dimension of the problem
		self.Iters, self.nGEN = 0, nGEN
		self.Evals, self.nFES = 0, nFES
		self.Fun = self.benchmark.function() if benchmark != None else None
		self.o = o if isinstance(o, np.ndarray) or o == None else np.asarray(o)
		self.M = M if isinstance(M, np.ndarray) or M == None else np.asarray(M)
		self.__initBounds()

	def __initBounds(self):
		Lower, Upper = self.benchmark.Lower, self.benchmark.Upper
		if isinstance(Lower, (int, float)): self.Lower = np.full(self.D, Lower)
		else: self.Lower = Lower if isinstance(Lower, np.ndarray) else np.asarray(Lower)
		if isinstance(Upper, (int, float)): self.Upper = np.full(self.D, Upper)
		else: self.Upper = Upper if isinstance(Upper, np.ndarray) else np.asarray(Upper)
		self.bRange = self.Upper - self.Lower

	def stopCond(self): return self.Evals >= self.nFES or (False if self.nGEN == None else self.Iters >= self.nGEN)

	def eval(self, A):
		# TODO add solustions to storage for anaysis
		if self.stopCond(): return np.inf
		self.Evals += 1
		# Shift
		X = A - self.o if self.o != None else A
		# rotate
		X = np.dot(X, self.M) if self.M != None else X
		return self.Fun(self.D, X)

	def nextIter(self): self.Iters += 1

	def isFisible(self, A): return False if True in (A < self.Lower) or True in (A > self.Upper) else True

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
