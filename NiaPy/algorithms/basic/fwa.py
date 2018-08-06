# encoding=utf8
# pylint: disable=mixed-indentation, trailing-whitespace, multiple-statements, attribute-defined-outside-init, logging-not-lazy, arguments-differ
import logging
from numpy import apply_along_axis, argmin, argmax, sum, sqrt, round, argsort, fabs, asarray, full, where
from NiaPy.algorithms.algorithm import Algorithm

logging.basicConfig()
logger = logging.getLogger('NiaPy.algorithms.basic')
logger.setLevel('INFO')

__all__ = ['FireworksAlgorithm', 'EnhancedFireworksAlgorithm', 'DynamicFireworksAlgorithm']

class FireworksAlgorithm(Algorithm):
	r"""Implementation of fireworks algorithm.

	**Algorithm:** Fireworks Algorithm
	**Date:** 2018
	**Authors:** Klemen Berkovič
	**License:** MIT
	**Reference URL:** https://www.springer.com/gp/book/9783662463529
	**Reference paper:** Tan, Ying. "Firework Algorithm: A Novel Swarm Intelligence Optimization Method." (2015).
	"""
	def __init__(self, **kwargs):
		if kwargs.get('name', None) == None: Algorithm.__init__(self, name=kwargs.get('name', 'FireworksAlgorithm'), sName=kwargs.get('sName', 'FWA'), **kwargs)
		else: Algorithm.__init__(self, **kwargs)

	def setParameters(self, N=40, m=40, a=1, b=2, A=40, epsilon=1e-31, **ukwargs):
		r"""Set the arguments of an algorithm.

		**Arguments**:
		N {integer} -- number of Fireworks
		m {integer} -- number of sparks
		a {integer} -- limitation of sparks
		b {integer} -- limitation of sparks
		A {real} --
		epsilon {real} -- Small number for non 0 devision
		"""
		self.N, self.m, self.a, self.b, self.A, self.epsilon = N, m, a, b, A, epsilon
		if ukwargs: logger.info('Unused arguments: %s' % (ukwargs))

	def initApmlitude(self, task):
		Ah = None
		if isinstance(self.A, (int, float)): Ah = full(task.D, self.A)
		else: Ah = self.Ainit if isinstance(self.Ah, ndarray) else asarray(self.A)
		return Ah

	def SparsksNo(self, x_f, xw_f, Ss):
		s = self.m * (xw_f - x_f + self.epsilon) / (Ss + self.epsilon)
		return round(self.b * self.m) if s > self.b * self.m and self.a < self.b < 1 else round(self.a * self.m)

	def ExplosionAmplitude(self, x_f, xb_f, A, As): return A * (x_f - xb_f - self.epsilon) / (As + self.epsilon)

	def ExplodeSpark(self, x, A, task): return self.Mapping(x + self.rand(task.D) * self.uniform(-A, A, task.D), task)

	def GaussianSpark(self, x, task): return self.Mapping(x + self.rand(task.D) * self.normal(1, 1, task.D), task)

	def Mapping(self, x, task): return task.Lower + x % (task.bRange) if not task.isFeasible(x) else x

	def R(self, x, FW): return sqrt(sum(fabs(x - FW)))

	def p(self, r, Rs): return r / Rs

	def NextGeneration(self, FW, FW_f, FWn, task):
		FWn_f = apply_along_axis(task.eval, 1, FWn)
		ib = argmin(FWn_f)
		if FWn_f[ib] < FW_f[0]: FW[0], FW_f[0] = FWn[ib], FWn_f[ib]
		R = [self.R(FWn[i], FWn) for i in range(len(FWn))]
		Rs = sum(R)
		P = [self.p(R[i], Rs) for i in range(len(FWn))]
		isort = argsort(P)[-(self.N - 1):]
		FW[1:], FW_f[1:] = asarray(FWn)[isort], FWn_f[isort]
		return FW, FW_f

	def runTask(self, task):
		FW, Ah = self.uniform(task.Lower, task.Upper, [self.N, task.D]), self.initApmlitude(task)
		FW_f = apply_along_axis(task.eval, 1, FW)
		ib = argmin(FW_f)
		FW[0], FW_f[0] = FW[ib], FW_f[ib]
		while not task.stopCond():
			iw, ib = argmax(FW_f), 0
			Ss, As = sum(FW_f[iw] - FW_f), sum(FW_f - FW_f[ib])
			S = [self.SparsksNo(FW_f[i], FW_f[iw], Ss) for i in range(self.N)]
			A = [self.ExplosionAmplitude(FW_f[i], FW_f[ib], Ah, As) for i in range(self.N)]
			FWn = [self.ExplodeSpark(FW[i], S[i], task) for i in range(self.N) for j in range(S[i])]
			for i in range(self.m): FWn.append(self.GaussianSpark(self.randint(self.N), task))
			FW, FW_t = self.NextGeneration(FW, FW_f, FWn, task)
		return FW[0], FW_f[0]

class EnhancedFireworksAlgorithm(FireworksAlgorithm):
	r"""Implementation of enganced fireworks algorithm.

	**Algorithm:** Enhanced Fireworks Algorithm
	**Date:** 2018
	**Authors:** Klemen Berkovič
	**License:** MIT
	**Reference URL:** https://ieeexplore.ieee.org/document/6557813/
	**Reference paper:** S. Zheng, A. Janecek and Y. Tan, "Enhanced Fireworks Algorithm," 2013 IEEE Congress on Evolutionary Computation, Cancun, 2013, pp. 2069-2077. doi: 10.1109/CEC.2013.6557813
	"""
	def __init__(self, **kwargs):
		if kwargs.get('name', None) == None: Algorithm.__init__(self, name=kwargs.get('name', 'EnhancedFireworksAlgorithm'), sName=kwargs.get('sName', 'EFWA'), **kwargs)
		else: Algorithm.__init__(self, **kwargs)

	def setParameters(self, Ainit=20, Afinal=5, **ukwargs):
		FireworksAlgorithm.setParameters(self, **ukwargs)
		self.Ainit, self.Afinal = Ainit, Afinal

	def initRanges(self, task):
		Ainit, Afinal = None, None
		if isinstance(self.Ainit, (int, float)): Ainit = full(task.D, self.Ainit)
		else: Ainit = self.Ainit if isinstance(self.Ainit, ndarray) else asarray(self.Ainit)
		if isinstance(self.Afinal, (int, float)): Afinal = full(task.D, self.Afinal)
		else: Afinal = self.Afinal if isinstance(self.Afinal, ndarray) else asarray(self.Afinal)
		return Ainit, Afinal, self.uAmin(Ainit, Afinal, task)

	def uAmin(self, Ainit, Afinal, task): return Ainit - sqrt(task.Evals * (2 * task.nFES - task.Evals)) * (Ainit - Afinal) / task.nFES

	def ExplosionAmplitude(self, x_f, xb_f, A_min, Ah, As, task):
		A = FireworksAlgorithm.ExplosionAmplitude(self, x_f, xb_f, Ah, As)
		ifix = where(A < A_min)
		A[ifix] = A_min[ifix]
		return A

	def GaussianSpark(self, x, xb, task): return self.Mapping(x + self.rand(task.D) * (xb - x) * self.normal(1, 1, task.D), task)

	def NextGeneration(self, FW, FW_f, FWn, task):
		FWn_f = apply_along_axis(task.eval, 1, FWn)
		ib = argmin(FWn_f)
		if FWn_f[ib] < FW_f[0]: FW[0], FW_f[0] = FWn[ib], FWn_f[ib]
		for i in range(1, self.N):
			r = self.randint(len(FWn))
			if FWn_f[r] < FW_f[i]: FW[i], FW_f[i] = FWn[r], FWn_f[r]
		return FW, FW_f

	def runTask(self, task):
		FW, Ah = self.uniform(task.Lower, task.Upper, [self.N, task.D]), self.initApmlitude(task)
		Ainit, Afinal, A_min = self.initRanges(task)
		FW_f = apply_along_axis(task.eval, 1, FW)
		ib = argmin(FW_f)
		FW[0], FW_f[0] = FW[ib], FW_f[ib]
		while not task.stopCondI():
			iw, ib = argmax(FW_f), 0
			Ss, As = sum(FW_f[iw] - FW_f), sum(FW_f - FW_f[ib])
			S = [self.SparsksNo(FW_f[i], FW_f[iw], Ss) for i in range(self.N)]
			A = [self.ExplosionAmplitude(FW_f[i], FW_f[ib], A_min, Ah, As, task) for i in range(self.N)]
			A_min = self.uAmin(Ainit, Afinal, task)
			FWn = [self.ExplodeSpark(FW[i], S[i], task) for i in range(self.N) for j in range(S[i])]
			for i in range(self.m): FWn.append(self.GaussianSpark(self.randint(self.N), FW[ib], task))
			FW, FW_f = self.NextGeneration(FW, FW_f, FWn, task)
		return FW[0], FW_f[0]

class DynamicFireworksAlgorithm(EnhancedFireworksAlgorithm):
	r"""Implementation of dynamic fireworks algorithm.

	**Algorithm:** Dynamic Fireworks Algorithm
	**Date:** 2018
	**Authors:** Klemen Berkovič
	**License:** MIT
	**Reference URL:** http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6900485&isnumber=6900223
	**Reference paper:** S. Zheng, A. Janecek, J. Li and Y. Tan, "Dynamic search in fireworks algorithm," 2014 IEEE Congress on Evolutionary Computation (CEC), Beijing, 2014, pp. 3222-3229. doi: 10.1109/CEC.2014.6900485
	"""
	def __init__(self, **kwargs): FireworksAlgorithm.__init__(self, name='DynamicFireworksAlgorithm', sName='dynFWA', **kwargs)

	def runTask(self, task):
		FW, Ah = self.uniform(task.Lower, task.Upper, [self.N, task.D]), self.initApmlitude(task)
		Ainit, Afinal, A_min = self.initRanges(task)
		FW_f = apply_along_axis(task.eval, 1, FW)
		ib = argmin(FW_f)
		FW[0], FW_f[0] = FW[ib], FW_f[ib]
		while not task.stopCondI():
			iw, ib = argmax(FW_f), 0
			Ss, As = sum(FW_f[iw] - FW_f), sum(FW_f - FW_f[ib])
			S = [self.SparsksNo(FW_f[i], FW_f[iw], Ss) for i in range(self.N)]
			A = [self.ExplosionAmplitude(FW_f[i], FW_f[ib], A_min, Ah, As, task) for i in range(self.N)]
			A_min = self.uAmin(Ainit, Afinal, task)
			FWn = [self.ExplodeSpark(FW[i], S[i], task) for i in range(self.N) for j in range(S[i])]
			for i in range(self.m): FWn.append(self.GaussianSpark(self.randint(self.N), FW[ib], task))
			FW, FW_f = self.NextGeneration(FW, FW_f, FWn, task)
		return FW[0], FW_f[0]

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
