# encoding=utf8
# pylint: disable=mixed-indentation, trailing-whitespace, multiple-statements, attribute-defined-outside-init, logging-not-lazy, arguments-differ, line-too-long, redefined-builtin, no-self-use, singleton-comparison, unused-argument, bad-continuation
import logging
from numpy import apply_along_axis, argmin, argmax, sum, sqrt, round, argsort, fabs, asarray, where
from NiaPy.algorithms.algorithm import Algorithm
from NiaPy.util import fullArray

logging.basicConfig()
logger = logging.getLogger('NiaPy.algorithms.basic')
logger.setLevel('INFO')

__all__ = ['FireworksAlgorithm', 'EnhancedFireworksAlgorithm', 'DynamicFireworksAlgorithm', 'DynamicFireworksAlgorithmGauss', 'BareBonesFireworksAlgorithm']

class BareBonesFireworksAlgorithm(Algorithm):
	r"""Implementation of bare bone fireworks algorithm.

	Algorithm:
		Bare Bones Fireworks Algorithm

	Date:
		2018

	Authors:
		Klemen Berkovič

	License:
		MIT

	Reference URL:
		https://www.sciencedirect.com/science/article/pii/S1568494617306609

	Reference paper:
		Junzhi Li, Ying Tan, The bare bones fireworks algorithm: A minimalist global optimizer, Applied Soft Computing, Volume 62, 2018, Pages 454-462, ISSN 1568-4946, https://doi.org/10.1016/j.asoc.2017.10.046.

	Attributes:
		Name (lsit of str): List of strings representing algorithm names
		n (int): Number of spraks
		C_a (float): amplification coefficient
		C_r (float): reduction coefficient
	"""
	Name = ['BareBonesFireworksAlgorithm', 'BBFWA']
	C_a, C_r = 1.5, 0.5
	n = 10

	@staticmethod
	def typeParameters(): return {
			'n': lambda x: isinstance(x, int) and x > 0,
			'C_a': lambda x: isinstance(x, (float, int)) and x > 1,
			'C_r': lambda x: isinstance(x, (float, int)) and 0 < x < 1
	}

	def setParameters(self, n=10, C_a=1.5, C_r=0.5, **ukwargs):
		r"""Set the arguments of an algorithm.

		Arguments:
			n (int): Number of sparks $\in [1, \infty)$
			C_a (float): Amplification coefficient :math:`\in [1, \infty)`
			C_r (float): Reduction coefficient :math:`\in (0, 1)`
		"""
		self.n, self.C_a, self.C_r = n, C_a, C_r
		if ukwargs: logger.info('Unused arguments: %s' % (ukwargs))

	def initPopulation(self, task):
		r"""Initialize starting population.

		Args:
			task (Task): Optimization task.

		Returns:
			Tuple[numpy.ndarray, float, Dict[str, Any]]:
				1. Initial solution.
				2. Initial solution function/fitness value
				3. Additional arguments:
					* A (numpy.ndarray): Starting aplitude or search range.
		"""
		x, A = self.uniform(task.Lower, task.Upper, task.D), task.bRange
		x_fit = task.eval(x)
		return x, x_fit, {'A': A}

	def runIteration(self, task, x, x_fit, xb, fxb, A, **dparams):
		r"""Core function of Bare Bones Fireworks Algorithm.

		Args:
			task (Task): Optimization task.
			x (numpy.ndarray): Current solution.
			x_fit (float): Current solution fitness/function value.
			xb (numpy.ndarray): Current best solution.
			fxb (float): Current best solution fitness/function value.
			A (numpy.ndarray): Serach range.
			dparams (Dict[str, Any]): Additional parameters.

		Returns:
			Tuple[numpy.ndarray, float, Dict[str, Any]]:
				1. New solution.
				2. New solution fitness/function value.
				3. Additional arguments:
					* A (numpy.ndarray): Serach range.
		"""
		S = apply_along_axis(task.repair, 1, self.uniform(x - A, x + A, [self.n, task.D]), self.Rand)
		S_fit = apply_along_axis(task.eval, 1, S)
		iS = argmin(S_fit)
		if S_fit[iS] < x_fit: x, x_fit, A = S[iS], S_fit[iS], self.C_a * A
		else: A = self.C_r * A
		return x, x_fit, {'A': A}

class FireworksAlgorithm(Algorithm):
	r"""Implementation of fireworks algorithm.

	Algorithm:
		Fireworks Algorithm

	Date:
		2018

	Authors:
		Klemen Berkovič

	License:
		MIT

	Reference URL:
		https://www.springer.com/gp/book/9783662463529

	Reference paper:
		Tan, Ying. "Firework Algorithm: A Novel Swarm Intelligence Optimization Method." (2015).

	Attributes:
		Name (list of str): List of stirngs representing algorithm names
	"""
	Name = ['FireworksAlgorithm', 'FWA']

	@staticmethod
	def typeParameters(): return {
			'N': lambda x: isinstance(x, int) and x > 0,
			'm': lambda x: isinstance(x, int) and x > 0,
			'a': lambda x: isinstance(x, (int, float)) and x > 0,
			'b': lambda x: isinstance(x, (int, float)) and x > 0,
			'epsilon': lambda x: isinstance(x, float) and 0 < x < 1
	}

	def setParameters(self, N=40, m=40, a=1, b=2, A=40, epsilon=1e-31, **ukwargs):
		r"""Set the arguments of an algorithm.

		Arguments:
			N (int): Number of Fireworks
			m (int): Number of sparks
			a (int): Limitation of sparks
			b (int): Limitation of sparks
			A (float): --
			epsilon (float): Small number for non 0 devision
		"""
		self.N, self.m, self.a, self.b, self.A, self.epsilon = N, m, a, b, A, epsilon
		if ukwargs: logger.info('Unused arguments: %s' % (ukwargs))

	def initAmplitude(self, task):
		r"""

		Args:
			task:

		Returns:

		"""
		return fullArray(self.A, task.D)

	def SparsksNo(self, x_f, xw_f, Ss):
		r"""

		Args:
			x_f:
			xw_f:
			Ss:

		Returns:

		"""
		s = self.m * (xw_f - x_f + self.epsilon) / (Ss + self.epsilon)
		return round(self.b * self.m) if s > self.b * self.m and self.a < self.b < 1 else round(self.a * self.m)

	def ExplosionAmplitude(self, x_f, xb_f, A, As):
		r"""

		Args:
			x_f:
			xb_f:
			A:
			As:

		Returns:

		"""
		return A * (x_f - xb_f - self.epsilon) / (As + self.epsilon)

	def ExplodeSpark(self, x, A, task):
		r"""

		Args:
			x:
			A:
			task:

		Returns:

		"""
		return self.Mapping(x + self.rand(task.D) * self.uniform(-A, A, task.D), task)

	def GaussianSpark(self, x, task):
		r"""

		Args:
			x:
			task:

		Returns:

		"""
		return self.Mapping(x + self.rand(task.D) * self.normal(1, 1, task.D), task)

	def Mapping(self, x, task):
		r"""

		Args:
			x:
			task:

		Returns:

		"""
		ir = where(x > task.Upper)
		x[ir] = task.Lower[ir] + x[ir] % task.bRange[ir]
		ir = where(x < task.Lower)
		x[ir] = task.Lower[ir] + x[ir] % task.bRange[ir]
		return x

	def R(self, x, FW):
		r"""

		Args:
			x:
			FW:

		Returns:

		"""
		return sqrt(sum(fabs(x - FW)))

	def p(self, r, Rs):
		r"""

		Args:
			r:
			Rs:

		Returns:

		"""
		return r / Rs

	def NextGeneration(self, FW, FW_f, FWn, task):
		r"""

		Args:
			FW:
			FW_f:
			FWn:
			task:

		Returns:

		"""
		FWn_f = apply_along_axis(task.eval, 1, FWn)
		ib = argmin(FWn_f)
		if FWn_f[ib] < FW_f[0]: FW[0], FW_f[0] = FWn[ib], FWn_f[ib]
		R = [self.R(FWn[i], FWn) for i in range(len(FWn))]
		Rs = sum(R)
		P = [self.p(R[i], Rs) for i in range(len(FWn))]
		isort = argsort(P)[-(self.N - 1):]
		FW[1:], FW_f[1:] = asarray(FWn)[isort], FWn_f[isort]
		return FW, FW_f

	def initPopulation(self, task):
		r"""

		Args:
			task:

		Returns:

		"""
		FW, Ah = self.uniform(task.Lower, task.Upper, [self.N, task.D]), self.initAmplitude(task)
		FW_f = apply_along_axis(task.eval, 1, FW)
		return FW, FW_f, {'Ah':Ah}

	def runIteration(self, task, FW, FW_f, xb, fxb, Ah, **dparams):
		r"""

		Args:
			task:
			FW:
			FW_f:
			xb:
			fxb:
			Ah:
			**dparams:

		Returns:

		"""
		iw, ib = argmax(FW_f), 0
		Ss, As = sum(FW_f[iw] - FW_f), sum(FW_f - FW_f[ib])
		S = [self.SparsksNo(FW_f[i], FW_f[iw], Ss) for i in range(self.N)]
		A = [self.ExplosionAmplitude(FW_f[i], FW_f[ib], Ah, As) for i in range(self.N)]
		FWn = [self.ExplodeSpark(FW[i], A[i], task) for i in range(self.N) for _ in range(S[i])]
		for i in range(self.m): FWn.append(self.GaussianSpark(self.randint(self.N), task))
		FW, FW_f = self.NextGeneration(FW, FW_f, FWn, task)
		return FW, FW_f, {'Ah':Ah}

class EnhancedFireworksAlgorithm(FireworksAlgorithm):
	r"""Implementation of enganced fireworks algorithm.

	Algorithm:
		Enhanced Fireworks Algorithm

	Date:
		2018

	Authors:
		Klemen Berkovič

	License:
		MIT

	Reference URL:
		https://ieeexplore.ieee.org/document/6557813/

	Reference paper:
		S. Zheng, A. Janecek and Y. Tan, "Enhanced Fireworks Algorithm," 2013 IEEE Congress on Evolutionary Computation, Cancun, 2013, pp. 2069-2077. doi: 10.1109/CEC.2013.6557813

	Attributes:
		Name (list of str): List of strings representing algorithm names
	"""
	Name = ['EnhancedFireworksAlgorithm', 'EFWA']

	@staticmethod
	def typeParameters():
		r"""

		Returns:

		"""
		d = FireworksAlgorithm.typeParameters()
		d['Ainit'] = lambda x: isinstance(x, (float, int)) and x > 0
		d['Afinal'] = lambda x: isinstance(x, (float, int)) and x > 0
		return d

	def setParameters(self, Ainit=20, Afinal=5, **ukwargs):
		r"""

		Args:
			Ainit:
			Afinal:
			**ukwargs:

		Returns:

		"""
		FireworksAlgorithm.setParameters(self, **ukwargs)
		self.Ainit, self.Afinal = Ainit, Afinal

	def initRanges(self, task):
		r"""

		Args:
			task:

		Returns:

		"""
		Ainit, Afinal = fullArray(self.Ainit, task.D), fullArray(self.Afinal, task.D)
		return Ainit, Afinal, self.uAmin(Ainit, Afinal, task)

	def uAmin(self, Ainit, Afinal, task):
		r"""

		Args:
			Ainit:
			Afinal:
			task:

		Returns:

		"""
		return Ainit - sqrt(task.Evals * (2 * task.nFES - task.Evals)) * (Ainit - Afinal) / task.nFES

	def ExplosionAmplitude(self, x_f, xb_f, A_min, Ah, As, task):
		r"""

		Args:
			x_f:
			xb_f:
			A_min:
			Ah:
			As:
			task:

		Returns:

		"""
		A = FireworksAlgorithm.ExplosionAmplitude(self, x_f, xb_f, Ah, As)
		ifix = where(A < A_min)
		A[ifix] = A_min[ifix]
		return A

	def GaussianSpark(self, x, xb, task):
		r"""

		Args:
			x:
			xb:
			task:

		Returns:

		"""
		return self.Mapping(x + self.rand(task.D) * (xb - x) * self.normal(1, 1, task.D), task)

	def NextGeneration(self, FW, FW_f, FWn, task):
		r"""

		Args:
			FW:
			FW_f:
			FWn:
			task:

		Returns:

		"""
		FWn_f = apply_along_axis(task.eval, 1, FWn)
		ib = argmin(FWn_f)
		if FWn_f[ib] < FW_f[0]: FW[0], FW_f[0] = FWn[ib], FWn_f[ib]
		for i in range(1, self.N):
			r = self.randint(len(FWn))
			if FWn_f[r] < FW_f[i]: FW[i], FW_f[i] = FWn[r], FWn_f[r]
		return FW, FW_f

	def initPopulation(self, task):
		r"""

		Args:
			task:

		Returns:

		"""
		FW, FW_f, d = FireworksAlgorithm.initPopulation(self, task)
		Ainit, Afinal, A_min = self.initRanges(task)
		d.update({'Ainit':Ainit, 'Afinal':Afinal, 'A_min':A_min})
		return FW, FW_f, d

	def runIteration(self, task, FW, FW_f, xb, fxb, Ah, Ainit, Afinal, A_min, **dparams):
		r"""

		Args:
			task:
			FW:
			FW_f:
			xb:
			fxb:
			Ah:
			Ainit:
			Afinal:
			A_min:
			**dparams:

		Returns:

		"""
		iw, ib = argmax(FW_f), 0
		Ss, As = sum(FW_f[iw] - FW_f), sum(FW_f - FW_f[ib])
		S = [self.SparsksNo(FW_f[i], FW_f[iw], Ss) for i in range(self.N)]
		A = [self.ExplosionAmplitude(FW_f[i], FW_f[ib], A_min, Ah, As, task) for i in range(self.N)]
		A_min = self.uAmin(Ainit, Afinal, task)
		FWn = [self.ExplodeSpark(FW[i], A[i], task) for i in range(self.N) for _ in range(S[i])]
		for i in range(self.m): FWn.append(self.GaussianSpark(self.randint(self.N), FW[ib], task))
		FW, FW_f = self.NextGeneration(FW, FW_f, FWn, task)
		return FW, FW_f, {'Ah':Ah, 'Ainit':Ainit, 'Afinal':Afinal, 'A_min':A_min}

class DynamicFireworksAlgorithmGauss(EnhancedFireworksAlgorithm):
	r"""Implementation of dynamic fireworks algorithm.

	Algorithm:
		Dynamic Fireworks Algorithm

	Date:
		2018

	Authors:
		Klemen Berkovič

	License:
		MIT

	Reference URL:
		http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6900485&isnumber=6900223

	Reference paper:
		S. Zheng, A. Janecek, J. Li and Y. Tan, "Dynamic search in fireworks algorithm," 2014 IEEE Congress on Evolutionary Computation (CEC), Beijing, 2014, pp. 3222-3229. doi: 10.1109/CEC.2014.6900485

	Attributes:
		Name (list of str): List of strings representing algorithm names
	"""
	Name = ['DynamicFireworksAlgorithmGauss', 'dynFWAG']

	@staticmethod
	def typeParameters():
		d = FireworksAlgorithm.typeParameters()
		d['A_cf'] = lambda x: isinstance(x, (float, int)) and x > 0
		d['C_a'] = lambda x: isinstance(x, (float, int)) and x > 1
		d['C_r'] = lambda x: isinstance(x, (float, int)) and 0 < x < 1
		d['epsilon'] = lambda x: isinstance(x, (float, int)) and 0 < x < 1
		return d

	def setParameters(self, A_cf=20, C_a=1.2, C_r=0.9, epsilon=1e-8, **ukwargs):
		r"""

		Args:
			A_cf:
			C_a:
			C_r:
			epsilon:
			**ukwargs:

		See Also:
			:func:`FireworksAlgorithm.setParameters`
		"""
		FireworksAlgorithm.setParameters(self, **ukwargs)
		self.A_cf, self.C_a, self.C_r, self.epsilon = A_cf, C_a, C_r, epsilon

	def initAmplitude(self, task):
		r"""

		Args:
			task:

		Returns:

		"""
		return FireworksAlgorithm.initAmplitude(self, task), task.bRange

	def ExplosionAmplitude(self, x_f, xb_f, A, As):
		r"""

		Args:
			x_f:
			xb_f:
			A:
			As:

		Returns:

		"""
		return FireworksAlgorithm.ExplosionAmplitude(self, x_f, xb_f, A, As)

	def Mapping(self, x, task):
		r"""

		Args:
			x:
			task:

		Returns:

		"""
		ir = where(x > task.Upper)
		x[ir] = self.uniform(task.Lower[ir], task.Upper[ir])
		ir = where(x < task.Lower)
		x[ir] = self.uniform(task.Lower[ir], task.Upper[ir])
		return x

	def repair(self, x, d, epsilon):
		r"""

		Args:
			x:
			d:
			epsilon:

		Returns:

		"""
		ir = where(x <= epsilon)
		x[ir] = d[ir]
		return x

	def NextGeneration(self, FW, FW_f, FWn, task):
		r"""

		Args:
			FW:
			FW_f:
			FWn:
			task:

		Returns:

		"""
		FWn_f = apply_along_axis(task.eval, 1, FWn)
		ib = argmin(FWn_f)
		for i, f in enumerate(FW_f):
			r = self.randint(len(FWn))
			if FWn_f[r] < f: FW[i], FW_f[i] = FWn[r], FWn_f[r]
		FW[0], FW_f[0] = FWn[ib], FWn_f[ib]
		return FW, FW_f

	def uCF(self, xnb, xcb, xcb_f, xb, xb_f, Acf, task):
		r"""

		Args:
			xnb:
			xcb:
			xcb_f:
			xb:
			xb_f:
			Acf:
			task:

		Returns:

		"""
		xnb_f = apply_along_axis(task.eval, 1, xnb)
		ib_f = argmin(xnb_f)
		if xnb_f[ib_f] <= xb_f: xb, xb_f = xnb[ib_f], xnb_f[ib_f]
		Acf = self.repair(Acf, task.bRange, self.epsilon)
		if xb_f >= xcb_f: xb, xb_f, Acf = xcb, xcb_f, Acf * self.C_a
		else: Acf = Acf * self.C_r
		return xb, xb_f, Acf

	def initPopulation(self, task):
		r"""Initialize population.

		Args:
			task (Task): Optimization task.

		Returns:
			Tuple[numpy.ndarray, numpy.ndarray[float], Dict[str, Any]]:
				1. Initialized population.
				2. Initialized population function/fitness values.
				3. Additional arguments:
					* Ah (): TODO
					* Ab (): TODO
		"""
		FW, FW_f, _ = Algorithm.initPopulation(self, task)
		Ah, Ab = self.initAmplitude(task)
		return FW, FW_f, {'Ah':Ah, 'Ab':Ab}

	def runIteration(self, task, FW, FW_f, xb, fxb, Ah, Ab, **dparams):
		r"""Core function of DynamicFireworksAlgorithmGauss algorithm.

		Args:
			task (Task): Optimization task.
			FW (numpy.ndarray): Current population.
			FW_f (numpy.ndarray[float]): Current populations function/fitness values.
			xb (numpy.ndarray): Global best individual.
			fxb (float): Global best fitness/function value.
			Ah (): TODO
			Ab (): TODO
			**dparams (Dict[str, Any]): Additional arguments.

		Returns:
			Tuple[numpy.ndarray, numpy.ndarray[float], Dict[str, Any]]:
				1. New population.
				2. New populations fitness/function values.
				3. Additional arguments:
					* Ah (): TODO
					* Ab (): TODO
		"""
		iw, ib = argmax(FW_f), argmin(FW_f)
		Ss, As = sum(FW_f[iw] - FW_f), sum(FW_f - FW_f[ib])
		S, sb = [self.SparsksNo(FW_f[i], FW_f[iw], Ss) for i in range(len(FW))], self.SparsksNo(fxb, FW_f[iw], Ss)
		A = [self.ExplosionAmplitude(FW_f[i], FW_f[ib], Ah, As) for i in range(len(FW))]
		FWn, xnb = [self.ExplodeSpark(FW[i], A[i], task) for i in range(self.N) for _ in range(S[i])], [self.ExplodeSpark(xb, Ab, task) for _ in range(sb)]
		for i in range(self.m): FWn.append(self.GaussianSpark(self.randint(self.N), FW[ib], task))
		FW, FW_f = self.NextGeneration(FW, FW_f, FWn, task)
		iw, ib = argmax(FW_f), 0
		_, _, Ab = self.uCF(xnb, FW[ib], FW_f[ib], xb, fxb, Ab, task)
		return FW, FW_f, {'Ah':Ah, 'Ab':Ab}

class DynamicFireworksAlgorithm(DynamicFireworksAlgorithmGauss):
	r"""Implementation of dynamic fireworks algorithm.

	Algorithm:
		Dynamic Fireworks Algorithm

	Date:
		2018

	Authors:
		Klemen Berkovič

	License:
		MIT

	Reference URL:
		http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6900485&isnumber=6900223

	Reference paper:
	 	S. Zheng, A. Janecek, J. Li and Y. Tan, "Dynamic search in fireworks algorithm," 2014 IEEE Congress on Evolutionary Computation (CEC), Beijing, 2014, pp. 3222-3229. doi: 10.1109/CEC.2014.6900485

	Attributes:
		Name (list of str): List of strings representing algorithm name

	See Also:
		DynamicFireworksAlgorithmGauss
	"""
	Name = ['DynamicFireworksAlgorithm', 'dynFWA']

	def runIteration(self, task, FW, FW_f, xb, fxb, Ah, Ab, **dparams):
		r"""Core function of Dynamic Fireworks Algorithm.

		Args:
			task (Task): Optimization task
			FW (numpy.ndarray): Current population
			FW_f (numpy.ndarray[float]): Current population fitness/function values
			xb (numpy.ndarray): Current best solution
			fxb (float): Current best solution's fitness/function value
			Ah (): TODO
			Ab (): TODO
			**dparams:

		Returns:
			Tuple[numpy.ndarray, numpy.ndarray[float], Dict[str, Any]]:
				1. New population.
				2. New population function/fitness values.
				3. Additional arguments:
					* Ah (): TODO
					* Ab (): TODO
		"""
		iw, ib = argmax(FW_f), argmin(FW_f)
		Ss, As = sum(FW_f[iw] - FW_f), sum(FW_f - FW_f[ib])
		S, sb = [self.SparsksNo(FW_f[i], FW_f[iw], Ss) for i in range(len(FW))], self.SparsksNo(fxb, FW_f[iw], Ss)
		A = [self.ExplosionAmplitude(FW_f[i], FW_f[ib], Ah, As) for i in range(len(FW))]
		FWn, xnb = [self.ExplodeSpark(FW[i], A[i], task) for i in range(self.N) for _ in range(S[i])], [self.ExplodeSpark(xb, Ab, task) for _ in range(sb)]
		FW, FW_f = self.NextGeneration(FW, FW_f, FWn, task)
		iw, ib = argmax(FW_f), 0
		_, _, Ab = self.uCF(xnb, FW[ib], FW_f[ib], xb, fxb, Ab, task)
		return FW, FW_f, {'Ah':Ah, 'Ab':Ab}

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
