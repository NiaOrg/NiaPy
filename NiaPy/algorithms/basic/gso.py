# encoding=utf8
import logging

from scipy.spatial.distance import euclidean
from numpy import full, apply_along_axis, copy, sum, fmax, pi, where

from NiaPy.algorithms.algorithm import Algorithm

logging.basicConfig()
logger = logging.getLogger('NiaPy.algorithms.basic')
logger.setLevel('INFO')

__all__ = ['GlowwormSwarmOptimization', 'GlowwormSwarmOptimizationV1', 'GlowwormSwarmOptimizationV2', 'GlowwormSwarmOptimizationV3']

class GlowwormSwarmOptimization(Algorithm):
	r"""Implementation of glowwarm swarm optimization.

	Algorithm:
		Glowwarm Swarm Optimization Algorithm

	Date:
		2018

	Authors:
		Klemen Berkovič

	License:
		MIT

	Reference URL:
		https://www.springer.com/gp/book/9783319515946

	Reference paper:
		Kaipa, Krishnanand N., and Debasish Ghose. Glowworm swarm optimization: theory, algorithms, and applications. Vol. 698. Springer, 2017.

	Attributes:
		Name (List[str]): List of strings represeinting algorithm name.
		l0 (float): Initial luciferin quantity for each glowworm.
		nt (float): --
		rs (float): Maximum sensing range.
		rho (float): Luciferin decay constant.
		gamma (float): Luciferin enhancement constant.
		beta (float): --
		s (float): --
		Distance (Callable[[numpy.ndarray, numpy.ndarray], float]]): Measure distance between two individuals.

	See Also:
		* :class:`NiaPy.algorithms.algorithm.Algorithm`
	"""
	Name = ['GlowwormSwarmOptimization', 'GSO']

	@staticmethod
	def algorithmInfo():
		r"""Get basic information of algorithm.

		Returns:
			str: Basic information.
		"""
		return r"""Kaipa, Krishnanand N., and Debasish Ghose. Glowworm swarm optimization: theory, algorithms, and applications. Vol. 698. Springer, 2017."""

	@staticmethod
	def typeParameters():
		r"""Get dictionary with functions for checking values of parameters.

		Returns:
			Dict[str, Callable]:
				* n (Callable[[int], bool])
				* l0 (Callable[[Union[float, int]], bool])
				* nt (Callable[[Union[float, int]], bool])
				* rho (Callable[[Union[float, int]], bool])
				* gamma (Callable[[float], bool])
				* beta (Callable[[float], bool])
				* s (Callable[[float], bool])
		"""
		return {
			'n': lambda x: isinstance(x, int) and x > 0,
			'l0': lambda x: isinstance(x, (float, int)) and x > 0,
			'nt': lambda x: isinstance(x, (float, int)) and x > 0,
			'rho': lambda x: isinstance(x, float) and 0 < x < 1,
			'gamma': lambda x: isinstance(x, float) and 0 < x < 1,
			'beta': lambda x: isinstance(x, float) and x > 0,
			's': lambda x: isinstance(x, float) and x > 0
		}

	def setParameters(self, n=25, l0=5, nt=5, rho=0.4, gamma=0.6, beta=0.08, s=0.03, Distance=euclidean, **ukwargs):
		r"""Set the arguments of an algorithm.

		Arguments:
			n (Optional[int]): Number of glowworms in population.
			l0 (Optional[float]): Initial luciferin quantity for each glowworm.
			nt (Optional[float]): --
			rs (Optional]float]): Maximum sensing range.
			rho (Optional[float]): Luciferin decay constant.
			gamma (Optional[float]): Luciferin enhancement constant.
			beta (Optional[float]): --
			s (Optional[float]): --
			Distance (Optional[Callable[[numpy.ndarray, numpy.ndarray], float]]]): Measure distance between two individuals.
		"""
		ukwargs.pop('NP', None)
		Algorithm.setParameters(self, NP=n, **ukwargs)
		self.l0, self.nt, self.rho, self.gamma, self.beta, self.s, self.Distance = l0, nt, rho, gamma, beta, s, Distance

	def getParameters(self):
		r"""Get algorithms parameters values.

		Returns:
			Dict[str, Any]: TODO.
		"""
		d = Algorithm.getParameters(self)
		d.pop('NP', None)
		d.update({
			'n': self.NP,
			'l0': self.l0,
			'nt': self.nt,
			'rho': self.rho,
			'gamma': self.gamma,
			'beta': self.beta,
			's': self.s,
			'Distance': self.Distance
		})
		return d

	def getNeighbors(self, i, r, GS, L):
		r"""Get neighbours of glowworm.

		Args:
			i (int): Index of glowworm.
			r (float): Neighborhood distance.
			GS (numpy.ndarray):
			L (numpy.ndarray[float]): Luciferin value of glowworm.

		Returns:
			numpy.ndarray[int]: Indexes of neighborhood glowworms.
		"""
		N = full(self.NP, 0)
		for j, gw in enumerate(GS): N[j] = 1 if i != j and self.Distance(GS[i], gw) <= r and L[i] >= L[j] else 0
		return N

	def probabilityes(self, i, N, L):
		r"""Calculate probabilities for glowworm to movement.

		Args:
			i (int): Index of glowworm to search for probable movement.
			N (numpy.ndarray[float]):
			L (numpy.ndarray[float]):

		Returns:
			numpy.ndarray[float]: Probabilities for each glowworm in swarm.
		"""
		d, P = sum(L[where(N == 1)] - L[i]), full(self.NP, .0)
		for j in range(self.NP): P[i] = ((L[j] - L[i]) / d) if N[j] == 1 else 0
		return P

	def moveSelect(self, pb, i):
		r"""TODO.

		Args:
			pb:
			i:

		Returns:

		"""
		r, b_l, b_u = self.rand(), 0, 0
		for j in range(self.NP):
			b_l, b_u = b_u, b_u + pb[i]
			if b_l < r < b_u: return j
		return self.randint(self.NP)

	def calcLuciferin(self, L, GS_f):
		r"""TODO.

		Args:
			L:
			GS_f:

		Returns:

		"""
		return (1 - self.rho) * L + self.gamma * GS_f

	def rangeUpdate(self, R, N, rs):
		r"""TODO.

		Args:
			R:
			N:
			rs:

		Returns:

		"""
		return R + self.beta * (self.nt - sum(N))

	def initPopulation(self, task):
		r"""Initialize population.

		Args:
			task (Task): Optimization task.

		Returns:
			Tuple[numpy.ndarray, numpy.ndarray[float], Dict[str, Any]]:
				1. Initialized population of glowwarms.
				2. Initialized populations function/fitness values.
				3. Additional arguments:
					* L (numpy.ndarray): TODO.
					* R (numpy.ndarray): TODO.
					* rs (numpy.ndarray): TODO.
		"""
		GS, GS_f, d = Algorithm.initPopulation(self, task)
		rs = euclidean(full(task.D, 0), task.bRange)
		L, R = full(self.NP, self.l0), full(self.NP, rs)
		d.update({'L': L, 'R': R, 'rs': rs})
		return GS, GS_f, d

	def runIteration(self, task, GS, GS_f, xb, fxb, L, R, rs, **dparams):
		r"""Core function of GlowwormSwarmOptimization algorithm.

		Args:
			task (Task): Optimization taks.
			GS (numpy.ndarray): Current population.
			GS_f (numpy.ndarray): Current populations fitness/function values.
			xb (numpy.ndarray): Global best individual.
			fxb (float): Global best individuals function/fitness value.
			L (numpy.ndarray):
			R (numpy.ndarray):
			rs (numpy.ndarray):
			**dparams Dict[str, Any]: Additional arguments.

		Returns:
			Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, float, Dict[str, Any]]:
				1. Initialized population of glowwarms.
				2. Initialized populations function/fitness values.
				3. New global best solution
				4. New global best sloutions fitness/objective value.
				5. Additional arguments:
					* L (numpy.ndarray): TODO.
					* R (numpy.ndarray): TODO.
					* rs (numpy.ndarray): TODO.
		"""
		GSo, Ro = copy(GS), copy(R)
		L = self.calcLuciferin(L, GS_f)
		N = [self.getNeighbors(i, Ro[i], GSo, L) for i in range(self.NP)]
		P = [self.probabilityes(i, N[i], L) for i in range(self.NP)]
		j = [self.moveSelect(P[i], i) for i in range(self.NP)]
		for i in range(self.NP): GS[i] = task.repair(GSo[i] + self.s * ((GSo[j[i]] - GSo[i]) / (self.Distance(GSo[j[i]], GSo[i]) + 1e-31)), rnd=self.Rand)
		for i in range(self.NP): R[i] = max(0, min(rs, self.rangeUpdate(Ro[i], N[i], rs)))
		GS_f = apply_along_axis(task.eval, 1, GS)
		xb, fxb = self.getBest(GS, GS_f, xb, fxb)
		return GS, GS_f, xb, fxb, {'L': L, 'R': R, 'rs': rs}

class GlowwormSwarmOptimizationV1(GlowwormSwarmOptimization):
	r"""Implementation of glowwarm swarm optimization.

	Algorithm:
		Glowwarm Swarm Optimization Algorithm

	Date:
		2018

	Authors:
		Klemen Berkovič

	License:
		MIT

	Reference URL:
		https://www.springer.com/gp/book/9783319515946

	Reference paper:
		Kaipa, Krishnanand N., and Debasish Ghose. Glowworm swarm optimization: theory, algorithms, and applications. Vol. 698. Springer, 2017.

	Attributes:
		Name (List[str]): List of strings representing algorithm names.
		alpha (float): --

	See Also:
		* :class:`NiaPy.algorithms.basic.GlowwormSwarmOptimization`
	"""
	Name = ['GlowwormSwarmOptimizationV1', 'GSOv1']

	def setParameters(self, **kwargs):
		r"""Set default parameters of the algorithm.

		Args:
			**kwargs (dict): Additional arguments.
		"""
		GlowwormSwarmOptimization.setParameters(self, **kwargs)

	def calcLuciferin(self, L, GS_f):
		r"""TODO.

		Args:
			L:
			GS_f:

		Returns:

		"""
		return fmax(0, (1 - self.rho) * L + self.gamma * GS_f)

	def rangeUpdate(self, R, N, rs):
		r"""TODO.

		Args:
			R:
			N:
			rs:

		Returns:

		"""
		return rs / (1 + self.beta * (sum(N) / (pi * rs ** 2)))

class GlowwormSwarmOptimizationV2(GlowwormSwarmOptimization):
	r"""Implementation of glowwarm swarm optimization.

	Algorithm:
		Glowwarm Swarm Optimization Algorithm

	Date:
		2018

	Authors:
		Klemen Berkovič

	License:
		MIT

	Reference URL:
		https://www.springer.com/gp/book/9783319515946

	Reference paper:
		Kaipa, Krishnanand N., and Debasish Ghose. Glowworm swarm optimization: theory, algorithms, and applications. Vol. 698. Springer, 2017.

	Attributes:
		Name (List[str]): List of strings representing algorithm names.
		alpha (float): --

	See Also:
		* :class:`NiaPy.algorithms.basic.GlowwormSwarmOptimization`
	"""
	Name = ['GlowwormSwarmOptimizationV2', 'GSOv2']

	def setParameters(self, alpha=0.2, **kwargs):
		r"""Set core parameters for GlowwormSwarmOptimizationV2 algorithm.

		Args:
			alpha (Optional[float]): --
			**kwargs (Dict[str, Any]): Additional arguments.

		See Also:
			* :func:`NiaPy.algorithms.basic.GlowwormSwarmOptimization.setParameters`
		"""
		GlowwormSwarmOptimization.setParameters(self, **kwargs)
		self.alpha = alpha

	def rangeUpdate(self, P, N, rs):
		r"""TODO.

		Args:
			P:
			N:
			rs:

		Returns:
			float: TODO
		"""
		return self.alpha + (rs - self.alpha) / (1 + self.beta * sum(N))

class GlowwormSwarmOptimizationV3(GlowwormSwarmOptimization):
	r"""Implementation of glowwarm swarm optimization.

	Algorithm:
		Glowwarm Swarm Optimization Algorithm

	Date:
		2018

	Authors:
		Klemen Berkovič

	License:
		MIT

	Reference URL:
		https://www.springer.com/gp/book/9783319515946

	Reference paper:
		Kaipa, Krishnanand N., and Debasish Ghose. Glowworm swarm optimization: theory, algorithms, and applications. Vol. 698. Springer, 2017.

	Attributes:
		Name (List[str]): List of strings representing algorithm names.
		beta1 (float): --

	See Also:
		* :class:`NiaPy.algorithms.basic.GlowwormSwarmOptimization`
	"""
	Name = ['GlowwormSwarmOptimizationV3', 'GSOv3']

	def setParameters(self, beta1=0.2, **kwargs):
		r"""Set core parameters for GlowwormSwarmOptimizationV3 algorithm.

		Args:
			beta1 (Optional[float]): --
			**kwargs (Dict[str, Any]): Additional arguments.

		See Also:
			* :func:`NiaPy.algorithms.basic.GlowwormSwarmOptimization.setParameters`
		"""
		GlowwormSwarmOptimization.setParameters(self, **kwargs)
		self.beta1 = beta1

	def rangeUpdate(self, R, N, rs):
		r"""TODO.

		Args:
			R:
			N:
			rs:

		Returns:

		"""
		return R + (self.beta * sum(N)) if sum(N) < self.nt else (-self.beta1 * sum(N))

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
