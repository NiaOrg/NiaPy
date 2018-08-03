# encoding=utf8
# pylint: disable=mixed-indentation, trailing-whitespace, multiple-statements, attribute-defined-outside-init, logging-not-lazy, arguments-differ
import logging
from scipy.spatial.distance import euclidean as ed
from numpy import apply_along_axis, argmin, argmax, sum, full, inf, ndarray, asarray
from NiaPy.algorithms.algorithm import Algorithm

logging.basicConfig()
logger = logging.getLogger('NiaPy.algorithms.basic')
logger.setLevel('INFO')

__all__ = ['KrillHerd']

class KrillHerdV4(Algorithm):
	r"""Implementation of krill herd algorithm.

	**Algorithm:** Krill Herd Algorithm
	**Date:** 2018
	**Authors:** Klemen Berkovič
	**License:** MIT
	**Reference URL:**
	**Reference paper:**
	"""
	def __init__(self, **kwargs): Algorithm.__init__(self, name='BareBonesFireworksAlgorithm', sName='BBFA', **kwargs)

	def setParameters(self, NP=50, N_max=0.01, V_f=0.02, D_max=0.002, C_t=0.93, W_n=0.42, W_f=0.38, d_s=2.63, Cr=0.2, Mu=0.05, epsilon=1e-4, **ukwargs):
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
		"""
		self.N, self.N_max, self.V_f, self.D_max, self.C_t, self.W_n, self.W_f, self.d_s, self._Cr, self._Mu, self.epsilon = NP, N_max, V_f, D_max, C_t, W_n, W_f, d_s, Cr, Mu, epsilon
		if ukwargs: logger.info('Unused arguments: %s' % (ukwargs))

	def initWeights(self, task):
		W_n, W_f = None, None
		if isinstance(self.W_n, (int, float)): W_n = full(task.D, self.W_n)
		else: W_n = self.W_n if isinstance(self.W_n, ndarray) else asarray(self.W_n)
		if isinstance(self.W_f, (int, float)): W_f = full(task.D, self.W_f)
		else: W_f = self.W_f if isinstance(self.W_f, ndarray) else asarray(self.W_f)
		return W_n, W_f

	def sensRange(self, ki, KH): return sum([ed(KH[ki], KH[i]) for i in range(self.N)]) / (5 * self.N)

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
		alpha_t = 2 * (self.rand.rand() + task.Iters / task.nGEN)
		return self.N_max * (alpha_l + alpha_t) + W * n

	def induceFragingMotion(self, i, x, x_f, f, W, KH, KH_f, ikh_b, ikh_w, task):
		beta_f = 2 * (1 - task.Iters / task.nGEN) * self.funK(KH_f[i], x_f, KH_f[ikh_b], KH_f[ikh_w]) * self.funX(KH[i], x)
		beta_b = self.funK(KH_f[i], KH_f[ikh_b], KH_f[ikh_b], KH_f[ikh_w]) * self.funX(KH[i], KH[ikh_b])
		return self.V_f * (beta_f + beta_b) + W * f

	def inducePhysicalDiffusion(self, task): return self.D_max * (1 - task.Iters / task.nGEN) * self.rand.uniform(-1, 1, task.D)

	def deltaT(self, task): return self.C_t * sum(task.bRange)

	def crossover(self, x, xo, Cr): return asarray([xo[i] if self.rand.rand() < Cr else x[i] for i in range(len(x))])

	def mutate(self, x, x_b, Mu): return asarray([x[i] if self.rand.rand() < Mu else x_b[i] + self.rand.rand() for i in range(len(x))])

	def getFoodLocation(self, KH, KH_f, task):
		x_food = asarray([sum((1 / KH_f) * KH[:, i]) for i in range(task.D)]) / sum(1 / KH_f)
		x_food_f = task.eval(x_food)
		return x_food, x_food_f

	def Mu(self, xf, yf, xf_best, xf_worst): return self._Mu / self.funK(xf, yf, xf_best, xf_worst)

	def Cr(self, xf, yf, xf_best, xf_worst): return self._Cr * self.funK(xf, yf, xf_best, xf_worst)

	def runTask(self, task):
		KH, N, F, x, x_fit = self.rand.uniform(task.Lower, task.Upper, [self.N, task.D]), full(self.N, .0), full(self.N, .0), None, inf
		W_n, W_f = self.initWeights(task)
		while not task.stopCondI():
			KH_f = apply_along_axis(task.eval, 1, KH)
			ikh_b, ikh_w = argmin(KH_f), argmax(KH_f)
			if KH_f[ikh_b] < x_fit: x, x_fit = KH[ikh_b], KH_f[ikh_b]
			N = asarray([self.induceNeigborsMotion(i, N[i], W_n, KH, KH_f, ikh_b, ikh_w, task) for i in range(self.N)])
			x_food, x_food_f = self.getFoodLocation(KH, KH_f, task)
			if x_food_f < x_fit: x, x_fit = x_food, x_food_f
			F = asarray([self.induceFragingMotion(i, x_food, x_food_f, F[i], W_f, KH, KH_f, ikh_b, ikh_w, task) for i in range(self.N)])
			D = asarray([self.inducePhysicalDiffusion(task) for i in range(self.N)])
			KH_n = KH + (self.deltaT(task) * (N + F + D))
			Cr = asarray([self.Cr(KH_f[i], KH_f[ikh_b], KH_f[ikh_b], KH_f[ikh_w]) for i in range(self.N)])
			KH_n = asarray([self.crossover(KH_n[i], KH[i], Cr) for i in range(self.N)])
			Mu = asarray([self.Mu(KH_f[i], KH_f[ikh_b], KH_f[ikh_b], KH_f[ikh_w]) for i in range(self.N)])
			KH_n = apply_along_axis(self.mutate, 1, KH_n, KH[ikh_b], Mu)
			KH = apply_along_axis(task.repair, 1, KH_n)
		return x, x_fit

class KrillHerdV1(KrillHerdV4):
	def crossover(self, x, xo, Cr): return x

	def mutate(self, x, x_b, Mu): return x

class KrillHerdV2(KrillHerdV4):
	def mutate(self, x, x_b, Mu): return x

class KrillHerdV3(KrillHerdV4):
	def crossover(self, x, xo, Cr): return x

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
