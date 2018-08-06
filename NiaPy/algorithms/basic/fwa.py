# encoding=utf8
# pylint: disable=mixed-indentation, trailing-whitespace, multiple-statements, attribute-defined-outside-init, logging-not-lazy, arguments-differ
import logging
from numpy import apply_along_axis, argmin, argmax, sum, sqrt, round, argsort, fabs, asarray, full, where
from NiaPy.algorithms.algorithm import Algorithm

logging.basicConfig()
logger = logging.getLogger('NiaPy.algorithms.basic')
logger.setLevel('INFO')

__all__ = ['FireworksAlgorithm', 'EnhancedFireworksAlgorithm', 'DynamicFireworksAlgorithm', 'BareBonesFireworksAlgorithm']

class BareBonesFireworksAlgorithm(Algorithm):
	r"""Implementation of bare bone fireworks algorithm.

	**Algorithm:** Bare Bones Fireworks Algorithm
	**Date:** 2018
	**Authors:** Klemen Berkovič
	**License:** MIT
	**Reference URL:**
	https://www.sciencedirect.com/science/article/pii/S1568494617306609
	**Reference paper:**
	Junzhi Li, Ying Tan, The bare bones fireworks algorithm: A minimalist global optimizer, Applied Soft Computing, Volume 62, 2018, Pages 454-462, ISSN 1568-4946, https://doi.org/10.1016/j.asoc.2017.10.046.
	"""
	def __init__(self, **kwargs): Algorithm.__init__(self, name='BareBonesFireworksAlgorithm', sName='BBFA', **kwargs)

	def setParameters(self, n=10, C_a=1.5, C_r=0.5, **ukwargs):
		r"""Set the arguments of an algorithm.

		**Arguments**:
		n {integer} -- number of sparks $\in [1, \infty)$
		C_a {real} -- amplification coefficient $\in [1, \infty)$
		C_r {real} -- reduction coefficient $\in (0, 1)$
		"""
		self.n, self.C_a, self.C_r = n, C_a, C_r
		if ukwargs: logger.info('Unused arguments: %s' % (ukwargs))

	def runTask(self, task):
		x, A = self.uniform(task.Lower, task.Upper, task.D), task.bRange
		x_fit = task.eval(x)
		while not task.stopCond():
			S = self.uniform(x - A, x + A, [self.n, task.D])
			S_fit = apply_along_axis(task.eval, 1, S)
			iS = argmin(S_fit)
			if S_fit[iS] < x_fit: x, x_fit, A = S[iS], S_fit[iS], self.C_a * A
			else: A = self.C_r * A
		return x, x_fit

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

	def initAmplitude(self, task):
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

	def Mapping(self, x, task):
		ir = where(x > task.Upper)
		x[ir] = task.Lower[ir] + x[ir] % task.bRange[ir]
		ir = where(x < task.Lower)
		x[ir] = task.Lower[ir] + x[ir] % task.bRange[ir]
		return x

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
		FW, Ah = self.uniform(task.Lower, task.Upper, [self.N, task.D]), self.initAmplitude(task)
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
		FW, Ah = self.uniform(task.Lower, task.Upper, [self.N, task.D]), self.initAmplitude(task)
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

	def setParameters(self, A_cf=20, C_a=1.4, C_r=0.47, **ukwargs):
		FireworksAlgorithm.setParameters(self, **ukwargs)
		self.A_cf, self.C_a, self.C_r = A_cf, C_a, C_r

	def initAmplitude(self, task): return FireworksAlgorithm.initAmplitude(self, task)

	def ExplosionAmplitude(self, x_f, xb_f, A, As): return FireworksAlgorithm.ExplosionAmplitude(self, x_f, xb_f, A, As)

	def Mapping(self, x, task):
		ir = where(x > task.Upper)
		x[ir] = self.uniform(task.Lower[ir], task.Upper[ir])
		ir = where(x < task.Lower)
		x[ir] = self.uniform(task.Lower[ir], task.Upper[ir])
		return x

	def NextGeneration(self, FW, FW_f, FWn, task):
		r"""Elitism Random Selection."""
		FWn_f = apply_along_axis(task.eval, 1, FWn)
		for i in range(self.N):
			r = self.randint(len(FWn))
			if FWn_f[r] < FW_f[i]: FW[i], FW_f[i] = FWn[r], FWn_f[r]
		return FW, FW_f

	def runTask(self, task):
		FW, Ah = self.uniform(task.Lower, task.Upper, [self.N, task.D]), self.initAmplitude(task)
		FW_f = apply_along_axis(task.eval, 1, FW)
		ib = argmin(FW_f)
		xb, xb_f = FW[ib], FW_f[ib]
		iw, ib = argmax(FW_f), argmin(FW_f)
		while not task.stopCondI():
			Ss, As = sum(FW_f[iw] - FW_f), sum(FW_f - FW_f[ib])
			S, sb = [self.SparsksNo(FW_f[i], FW_f[iw], Ss) for i in range(self.N)], self.SparsksNo(xb_f, FW_f[iw], Ss)
			A = [self.ExplosionAmplitude(FW_f[i], FW_f[ib], Ah, As) for i in range(self.N)]
			FWn, xbn = [self.ExplodeSpark(FW[i], S[i], task) for i in range(self.N) for j in range(S[i])], self.ExplodeSpark(xb, sb, task)
			FW, FW_f = self.NextGeneration(FW, FW_f, FWn, task)
			iw, ib = argmax(FW_f), argmin(FW_f)
			if xb_f > FW_f[ib]: xb, xb_f = FW[ib], FW_f[ib]
			xbn_f = task.eval(self.Mapping(xbn, task))
			if xbn_f < xb_f: xb, xb_f = xbn, xbn_f
		return xb, xb_f

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
