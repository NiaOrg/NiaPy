# encoding=utf8
# pylint: disable=mixed-indentation, trailing-whitespace, multiple-statements, attribute-defined-outside-init, logging-not-lazy, arguments-differ
import logging
from scipy.spatial.distance import euclidean as ed
from numpy import apply_along_axis, argmin, argmax, sum, full, inf, ndarray, asarray
from NiaPy.algorithms.algorithm import Algorithm

logging.basicConfig()
logger = logging.getLogger('NiaPy.algorithms.basic')
logger.setLevel('INFO')

__all__ = ['KrillHerdV1', 'KrillHerdV2', 'KrillHerdV3', 'KrillHerdV4', 'KrillHerdV11']

class KrillHerd(Algorithm):
	r"""Implementation of krill herd algorithm.

	**Algorithm:** Krill Herd Algorithm
	**Date:** 2018
	**Authors:** Klemen Berkovič
	**License:** MIT
	**Reference URL:** http://www.sciencedirect.com/science/article/pii/S1007570412002171
	**Reference paper:** Amir Hossein Gandomi, Amir Hossein Alavi, Krill herd: A new bio-inspired optimization algorithm, Communications in Nonlinear Science and Numerical Simulation, Volume 17, Issue 12, 2012, Pages 4831-4845, ISSN 1007-5704, https://doi.org/10.1016/j.cnsns.2012.05.010.
	"""
	def __init__(self, **kwargs):
		if kwargs.get('name', None) == None: Algorithm.__init__(self, name='KrillHerd', sName='KH', **kwargs)
		else: Algorithm.__init__(self, **kwargs)

	def setParameters(self, NP=50, N_max=0.01, V_f=0.02, D_max=0.002, C_t=0.93, W_n=0.42, W_f=0.38, d_s=2.63, nn=5, Cr=0.2, Mu=0.05, epsilon=1e-31, **ukwargs):
		r"""Set the arguments of an algorithm.

		**Arguments**:
		NP {integer} -- Number of krill herds in population
		N_max {real} -- maximum induced speed
		V_f {real} -- foraging speed
		D_max {real} -- maximum diffsion speed
		C_t {real} -- constant $\in [0, 2]$
		W_n {real} or {array} -- inerta weights of the motion iduced from neighbors $\in [0, 1]$
		W_f {real} or {array} -- inerta weights of the motion iduced from fraging $\in [0, 1]$
		d_s {real} -- maximum euclidean distance for neighbors
		nn {integer} -- maximu neighbors for neighbors effect
		Cr {real} -- Crossover rate
		Mu {real} -- Mutation rate
		epsilon {real} -- Small numbers for devision
		"""
		print (self.N, ' ', NP)
		self.N, self.N_max, self.V_f, self.D_max, self.C_t, self.W_n, self.W_f, self.d_s, self.nn, self._Cr, self._Mu, self.epsilon = NP, N_max, V_f, D_max, C_t, W_n, W_f, d_s, nn, Cr, Mu, epsilon
		if ukwargs: logger.info('Unused arguments: %s' % (ukwargs))

	def initWeights(self, task):
		W_n, W_f = None, None
		if isinstance(self.W_n, (int, float)): W_n = full(task.D, self.W_n)
		else: W_n = self.W_n if isinstance(self.W_n, ndarray) else asarray(self.W_n)
		if isinstance(self.W_f, (int, float)): W_f = full(task.D, self.W_f)
		else: W_f = self.W_f if isinstance(self.W_f, ndarray) else asarray(self.W_f)
		return W_n, W_f

	def sensRange(self, ki, KH): return sum([ed(KH[ki], KH[i]) for i in range(self.N)]) / (self.nn * self.N)

	def getNeigbors(self, i, ids, KH):
		N = list()
		for j in range(self.N):
			if j != i and ids > ed(KH[i], KH[j]): N.append(j)
		return N

	def funX(self, x, y): return (y - x) / (ed(y, x) + self.epsilon)

	def funK(self, x, y, b, w): return (x - y) / (w - b)

	def induceNeigborsMotion(self, i, n, W, KH, KH_f, ikh_b, ikh_w, task):
		Ni = self.getNeigbors(i, self.sensRange(i, KH), KH)
		Nx, Nf, f_b, f_w = KH[Ni], KH_f[Ni], KH_f[ikh_b], KH_f[ikh_w]
		alpha_l = sum(asarray([self.funK(KH_f[i], j, f_b, f_w) for j in Nf]) * asarray([self.funX(KH[i], j) for j in Nx]).T)
		alpha_t = 2 * (1 + self.rand() * task.Iters / task.nGEN)
		return self.N_max * (alpha_l + alpha_t) + W * n

	def induceFragingMotion(self, i, x, x_f, f, W, KH, KH_f, ikh_b, ikh_w, task):
		beta_f = 2 * (1 - task.Iters / task.nGEN) * self.funK(KH_f[i], x_f, KH_f[ikh_b], KH_f[ikh_w]) * self.funX(KH[i], x) if KH_f[ikh_b] < KH_f[i] else 0
		beta_b = self.funK(KH_f[i], KH_f[ikh_b], KH_f[ikh_b], KH_f[ikh_w]) * self.funX(KH[i], KH[ikh_b])
		return self.V_f * (beta_f + beta_b) + W * f

	def inducePhysicalDiffusion(self, task): return self.D_max * (1 - task.Iters / task.nGEN) * self.uniform(-1, 1, task.D)

	def deltaT(self, task): return self.C_t * sum(task.bRange)

	def crossover(self, x, xo, Cr): return asarray([xo[i] if self.rand() < Cr else x[i] for i in range(len(x))])

	def mutate(self, x, x_b, Mu): return asarray([x[i] if self.rand() < Mu else x_b[i] + self.rand() for i in range(len(x))])

	def getFoodLocation(self, KH, KH_f, task):
		x_food = task.repair(asarray([sum(KH[:, i] / KH_f) for i in range(task.D)]) / sum(1 / KH_f))
		x_food_f = task.eval(x_food)
		return x_food, x_food_f

	def Mu(self, xf, yf, xf_best, xf_worst): return self._Mu / (self.funK(xf, yf, xf_best, xf_worst) + 1e-31)

	def Cr(self, xf, yf, xf_best, xf_worst): return self._Cr * self.funK(xf, yf, xf_best, xf_worst)

	def runTask(self, task):
		KH, N, F, x, x_fit = self.uniform(task.Lower, task.Upper, [self.N, task.D]), full(self.N, .0), full(self.N, .0), None, inf
		W_n, W_f = self.initWeights(task)
		while not task.stopCondI():
			KH_f = apply_along_axis(task.eval, 1, KH)
			ikh_b, ikh_w = argmin(KH_f), argmax(KH_f)
			if KH_f[ikh_b] < x_fit: x, x_fit = KH[ikh_b], KH_f[ikh_b]
			x_food, x_food_f = self.getFoodLocation(KH, KH_f, task)
			if x_food_f < x_fit: x, x_fit = x_food, x_food_f
			N = asarray([self.induceNeigborsMotion(i, N[i], W_n, KH, KH_f, ikh_b, ikh_w, task) for i in range(self.N)])
			F = asarray([self.induceFragingMotion(i, x_food, x_food_f, F[i], W_f, KH, KH_f, ikh_b, ikh_w, task) for i in range(self.N)])
			D = asarray([self.inducePhysicalDiffusion(task) for i in range(self.N)])
			KH_n = KH + (self.deltaT(task) * (N + F + D))
			Cr = asarray([self.Cr(KH_f[i], KH_f[ikh_b], KH_f[ikh_b], KH_f[ikh_w]) for i in range(self.N)])
			KH_n = asarray([self.crossover(KH_n[i], KH[i], Cr[i]) for i in range(self.N)])
			Mu = asarray([self.Mu(KH_f[i], KH_f[ikh_b], KH_f[ikh_b], KH_f[ikh_w]) for i in range(self.N)])
			KH_n = asarray([self.mutate(KH_n[i], KH[ikh_b], Mu[i]) for i in range(self.N)])
			KH = apply_along_axis(task.repair, 1, KH_n)
		return x, x_fit

class KrillHerdV4(KrillHerd):
	r"""Implementation of krill herd algorithm.

	**Algorithm:** Krill Herd Algorithm
	**Date:** 2018
	**Authors:** Klemen Berkovič
	**License:** MIT
	**Reference URL:** http://www.sciencedirect.com/science/article/pii/S1007570412002171
	**Reference paper:** Amir Hossein Gandomi, Amir Hossein Alavi, Krill herd: A new bio-inspired optimization algorithm, Communications in Nonlinear Science and Numerical Simulation, Volume 17, Issue 12, 2012, Pages 4831-4845, ISSN 1007-5704, https://doi.org/10.1016/j.cnsns.2012.05.010.
	"""
	def __init__(self, **kwargs):
		if kwargs.get('name', None) == None: KrillHerd.__init__(self, name='KrillHerdV4', sName='KHv4', **kwargs)
		else: KrillHerd.__init__(self, **kwargs)

	def setParameters(self, NP=50, N_max=0.01, V_f=0.02, D_max=0.002, C_t=0.93, W_n=0.42, W_f=0.38, d_s=2.63, **ukwargs): KrillHerd.setParameters(NP, N_max, V_f, D_max, C_t, W_n, W_f, d_s, 4, 0.2, 0.05, 1e-31, **ukwargs)

class KrillHerdV1(KrillHerdV4):
	r"""Implementation of krill herd algorithm.

	**Algorithm:** Krill Herd Algorithm
	**Date:** 2018
	**Authors:** Klemen Berkovič
	**License:** MIT
	**Reference URL:** http://www.sciencedirect.com/science/article/pii/S1007570412002171
	**Reference paper:** Amir Hossein Gandomi, Amir Hossein Alavi, Krill herd: A new bio-inspired optimization algorithm, Communications in Nonlinear Science and Numerical Simulation, Volume 17, Issue 12, 2012, Pages 4831-4845, ISSN 1007-5704, https://doi.org/10.1016/j.cnsns.2012.05.010.
	"""
	def __init__(self, **kwargs):
		if kwargs.get('name', None) == None: KrillHerd.__init__(self, name='KrillHerdV1', sName='KHv1', **kwargs)
		else: KrillHerd.__init__(self, **kwargs)

	def crossover(self, x, xo, Cr): return x

	def mutate(self, x, x_b, Mu): return x

class KrillHerdV2(KrillHerdV4):
	r"""Implementation of krill herd algorithm.

	**Algorithm:** Krill Herd Algorithm
	**Date:** 2018
	**Authors:** Klemen Berkovič
	**License:** MIT
	**Reference URL:** http://www.sciencedirect.com/science/article/pii/S1007570412002171
	**Reference paper:** Amir Hossein Gandomi, Amir Hossein Alavi, Krill herd: A new bio-inspired optimization algorithm, Communications in Nonlinear Science and Numerical Simulation, Volume 17, Issue 12, 2012, Pages 4831-4845, ISSN 1007-5704, https://doi.org/10.1016/j.cnsns.2012.05.010.
	"""
	def __init__(self, **kwargs):
		if kwargs.get('name', None) == None: KrillHerd.__init__(self, name='KrillHerdV2', sName='KHv2', **kwargs)
		else: KrillHerd.__init__(self, **kwargs)

	def mutate(self, x, x_b, Mu): return x

class KrillHerdV3(KrillHerdV4):
	r"""Implementation of krill herd algorithm.

	**Algorithm:** Krill Herd Algorithm
	**Date:** 2018
	**Authors:** Klemen Berkovič
	**License:** MIT
	**Reference URL:** http://www.sciencedirect.com/science/article/pii/S1007570412002171
	**Reference paper:** Amir Hossein Gandomi, Amir Hossein Alavi, Krill herd: A new bio-inspired optimization algorithm, Communications in Nonlinear Science and Numerical Simulation, Volume 17, Issue 12, 2012, Pages 4831-4845, ISSN 1007-5704, https://doi.org/10.1016/j.cnsns.2012.05.010.
	"""
	def __init__(self, **kwargs):
		if kwargs.get('name', None) == None: KrillHerd.__init__(self, name='KrillHerdV3', sName='KHv3', **kwargs)
		else: KrillHerd.__init__(self, **kwargs)

	def crossover(self, x, xo, Cr): return x

class KrillHerdV11(KrillHerdV4):
	r"""Implementation of krill herd algorithm.

	**Algorithm:** Krill Herd Algorithm
	**Date:** 2018
	**Authors:** Klemen Berkovič
	**License:** MIT
	**Reference URL:**
	**Reference paper:**
	"""
	def __init__(self, **kwargs):
		if kwargs.get('name', None) == None: KrillHerd.__init__(self, name='KrillHerdV11', sName='KHv11', **kwargs)
		else: KrillHerd.__init__(self, **kwargs)

	def sensRange(self, ki, HK): return 

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
