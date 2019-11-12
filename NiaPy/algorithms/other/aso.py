# encoding=utf8
import logging
from scipy.spatial.distance import euclidean
from numpy import apply_along_axis, argmin, full, inf, where, asarray, random as rand, sort, exp
from NiaPy.algorithms.algorithm import Algorithm
from NiaPy.util import fullArray

logging.basicConfig()
logger = logging.getLogger('NiaPy.algorithms.other')
logger.setLevel('INFO')

__all__ = ['AnarchicSocietyOptimization', 'Elitism', 'Sequential', 'Crossover']

def Elitism(x, xpb, xb, xr, MP_c, MP_s, MP_p, F, CR, task, rnd=rand):
	r"""Select the best of all three strategies.

	Args:
		x (numpy.ndarray): individual position.
		xpb (numpy.ndarray): individuals best position.
		xb (numpy.ndarray): current best position.
		xr (numpy.ndarray): random individual.
		MP_c (float): Fickleness index value.
		MP_s (float): External irregularity index value.
		MP_p (float): Internal irregularity index value.
		F (float): scale factor.
		CR (float): crossover factor.
		task (Task): optimization task.
		rnd (mtrand.randomstate): random number generator.

	Returns:
		Tuple[numpy.ndarray, float]:
			1. New position of individual
			2. New positions fitness/function value
	"""
	xn = [task.repair(MP_C(x, F, CR, MP_c, rnd), rnd=rnd), task.repair(MP_S(x, xr, xb, CR, MP_s, rnd), rnd=rnd), task.repair(MP_P(x, xpb, CR, MP_p, rnd), rnd=rnd)]
	xn_f = apply_along_axis(task.eval, 1, xn)
	ib = argmin(xn_f)
	return xn[ib], xn_f[ib]

def Sequential(x, xpb, xb, xr, MP_c, MP_s, MP_p, F, CR, task, rnd=rand):
	r"""Sequentialy combines all three strategies.

	Args:
		x (numpy.ndarray): individual position.
		xpb (numpy.ndarray): individuals best position.
		xb (numpy.ndarray): current best position.
		xr (numpy.ndarray): random individual.
		MP_c (float): Fickleness index value.
		MP_s (float): External irregularity index value.
		MP_p (float): Internal irregularity index value.
		F (float): scale factor.
		CR (float): crossover factor.
		task (Task): optimization task.
		rnd (mtrand.randomstate): random number generator.

	Returns:
		tuple[numpy.ndarray, float]:
			1. new position
			2. new positions function/fitness value
	"""
	xn = task.repair(MP_S(MP_P(MP_C(x, F, CR, MP_c, rnd), xpb, CR, MP_p, rnd), xr, xb, CR, MP_s, rnd), rnd=rnd)
	return xn, task.eval(xn)

def Crossover(x, xpb, xb, xr, MP_c, MP_s, MP_p, F, CR, task, rnd=rand):
	r"""Create a crossover over all three strategies.

	Args:
		x (numpy.ndarray): individual position.
		xpb (numpy.ndarray): individuals best position.
		xb (numpy.ndarray): current best position.
		xr (numpy.ndarray): random individual.
		MP_c (float): Fickleness index value.
		MP_s (float): External irregularity index value.
		MP_p (float): Internal irregularity index value.
		F (float): scale factor.
		CR (float): crossover factor.
		task (Task): optimization task.
		rnd (mtrand.randomstate): random number generator.

	Returns:
		Tuple[numpy.ndarray, float]:
			1. new position
			2. new positions function/fitness value
	"""
	xns = [task.repair(MP_C(x, F, CR, MP_c, rnd), rnd=rnd), task.repair(MP_S(x, xr, xb, CR, MP_s, rnd), rnd=rnd), task.repair(MP_P(x, xpb, CR, MP_p, rnd), rnd=rnd)]
	x = asarray([xns[rnd.randint(len(xns))][i] if rnd.rand() < CR else x[i] for i in range(len(x))])
	return x, task.eval(x)

def MP_C(x, F, CR, MP, rnd=rand):
	r"""Get bew position based on fickleness.

	Args:
		x (numpy.ndarray): Current individuals position.
		F (float): Scale factor.
		CR (float): Crossover probability.
		MP (float): Fickleness index value
		rnd (mtrand.RandomState): Random number generator

	Returns:
		numpy.ndarray: New position
	"""
	if MP < 0.5:
		b = sort(rnd.choice(len(x), 2, replace=False))
		x[b[0]:b[1]] = x[b[0]:b[1]] + F * rnd.normal(0, 1, b[1] - b[0])
		return x
	return asarray([x[i] + F * rnd.normal(0, 1) if rnd.rand() < CR else x[i] for i in range(len(x))])

def MP_S(x, xr, xb, CR, MP, rnd=rand):
	r"""Get new position based on external irregularity.

	Args:
		x (numpy.ndarray): Current individuals position.
		xr (numpy.ndarray): Random individuals position.
		xb (numpy.ndarray): Global best individuals position.
		CR (float): Crossover probability.
		MP (float): External irregularity index.
		rnd (mtrand.RandomState): Random number generator.

	Returns:
		numpy.ndarray: New position.
	"""
	if MP < 0.25:
		b = sort(rnd.choice(len(x), 2, replace=False))
		x[b[0]:b[1]] = xb[b[0]:b[1]]
		return x
	elif MP < 0.5: return asarray([xb[i] if rnd.rand() < CR else x[i] for i in range(len(x))])
	elif MP < 0.75:
		b = sort(rnd.choice(len(x), 2, replace=False))
		x[b[0]:b[1]] = xr[b[0]:b[1]]
		return x
	return asarray([xr[i] if rnd.rand() < CR else x[i] for i in range(len(x))])

def MP_P(x, xpb, CR, MP, rnd=rand):
	r"""Get new position based on internal irregularity.

	Args:
		x (numpy.ndarray): Current individuals position.
		xpb (numpy.ndarray): Current individuals personal best position.
		CR (float): Crossover probability.
		MP (float): Internal irregularity index value.
		rnd (mtrand.RandomState): Random number generator.

	Returns:
		numpy.ndarray: Current individuals new position.
	"""
	if MP < 0.5:
		b = sort(rnd.choice(len(x), 2, replace=False))
		x[b[0]:b[1]] = xpb[b[0]:b[1]]
		return x
	return asarray([xpb[i] if rnd.rand() < CR else x[i] for i in range(len(x))])

class AnarchicSocietyOptimization(Algorithm):
	r"""Implementation of Anarchic Society Optimization algorithm.

	Algorithm:
		Anarchic Society Optimization algorithm

	Date:
		2018

	Authors:
		Klemen Berkovič

	License:
		MIT

	Reference paper:
		Ahmadi-Javid, Amir. "Anarchic Society Optimization: A human-inspired method." Evolutionary Computation (CEC), 2011 IEEE Congress on. IEEE, 2011.

	Attributes:
		Name (list of str): List of stings representing name of algorithm.
		alpha (List[float]): Factor for fickleness index function :math:`\in [0, 1]`.
		gamma (List[float]): Factor for external irregularity index function :math:`\in [0, \infty)`.
		theta (List[float]): Factor for internal irregularity index function :math:`\in [0, \infty)`.
		d (Callable[[float, float], float]): function that takes two arguments that are function values and calcs the distance between them.
		dn (Callable[[numpy.ndarray, numpy.ndarray], float]): function that takes two arguments that are points in function landscape and calcs the distance between them.
		nl (float): Normalized range for neighborhood search :math:`\in (0, 1]`.
		F (float): Mutation parameter.
		CR (float): Crossover parameter :math:`\in [0, 1]`.
		Combination (Callable[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, float, float, float, float, float, float, Task, mtrand.RandomState]): Function for combining individuals to get new position/individual.

	See Also:
		* :class:`NiaPy.algorithms.Algorithm`
	"""
	Name = ['AnarchicSocietyOptimization', 'ASO']

	@staticmethod
	def typeParameters():
		r"""Get dictionary with functions for checking values of parameters.

		Returns:
			Dict[str, Callable]:
				* alpha (Callable): TODO
				* gamma (Callable): TODO
				* theta (Callable): TODO
				* nl (Callable): TODO
				* F (Callable[[Union[float, int]], bool]): TODO
				* CR (Callable[[Union[float, int]], bool]): TODO

		See Also:
			* :func:`NiaPy.algorithms.Algorithm.typeParameters`
		"""
		d = Algorithm.typeParameters()
		d.update({
			'alpha': lambda x: True,
			'gamma': lambda x: True,
			'theta': lambda x: True,
			'nl': lambda x: True,
			'F': lambda x: isinstance(x, (int, float)) and x > 0,
			'CR': lambda x: isinstance(x, float) and 0 <= x <= 1
		})
		return d

	def setParameters(self, NP=43, alpha=(1, 0.83), gamma=(1.17, 0.56), theta=(0.932, 0.832), d=euclidean, dn=euclidean, nl=1, F=1.2, CR=0.25, Combination=Elitism, **ukwargs):
		r"""Set the parameters for the algorith.

		Arguments:
			alpha (Optional[List[float]]): Factor for fickleness index function :math:`\in [0, 1]`.
			gamma (Optional[List[float]]): Factor for external irregularity index function :math:`\in [0, \infty)`.
			theta (Optional[List[float]]): Factor for internal irregularity index function :math:`\in [0, \infty)`.
			d (Optional[Callable[[float, float], float]]): function that takes two arguments that are function values and calcs the distance between them.
			dn (Optional[Callable[[numpy.ndarray, numpy.ndarray], float]]): function that takes two arguments that are points in function landscape and calcs the distance between them.
			nl (Optional[float]): Normalized range for neighborhood search :math:`\in (0, 1]`.
			F (Optional[float]): Mutation parameter.
			CR (Optional[float]): Crossover parameter :math:`\in [0, 1]`.
			Combination (Optional[Callable[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, float, float, float, float, float, float, Task, mtrand.RandomState]]): Function for combining individuals to get new position/individual.

		See Also:
			* :func:`NiaPy.algorithms.Algorithm.setParameters`
			* Combination methods:
				* :func:`NiaPy.algorithms.other.Elitism'
				* :func:`NiaPy.algorithms.other.Crossover`
				* :func:`NiaPy.algorithms.other.Sequential`
		"""
		Algorithm.setParameters(self, NP=NP, **ukwargs)
		self.alpha, self.gamma, self.theta, self.d, self.dn, self.nl, self.F, self.CR, self.Combination = alpha, gamma, theta, d, dn, nl, F, CR, Combination

	def init(self, task):
		r"""Initialize dynamic parameters of algorithm.

		Args:
			task (Task): Optimization task.

		Returns:
			Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
				1. Array of `self.alpha` propagated values
				2. Array of `self.gamma` propagated values
				3. Array of `self.theta` propagated values
		"""
		return fullArray(self.alpha, self.NP), fullArray(self.gamma, self.NP), fullArray(self.theta, self.NP)

	def FI(self, x_f, xpb_f, xb_f, alpha):
		r"""Get fickleness index.

		Args:
			x_f (float): Individuals fitness/function value.
			xpb_f (float): Individuals personal best fitness/function value.
			xb_f (float): Current best found individuals fitness/function value.
			alpha (float): TODO.

		Returns:
			float: Fickleness index.
		"""
		return 1 - alpha * xb_f / x_f - (1 - alpha) * xpb_f / x_f

	def EI(self, x_f, xnb_f, gamma):
		r"""Get external irregularity index.

		Args:
			x_f (float): Individuals fitness/function value.
			xnb_f (float): Individuals new fitness/function value.
			gamma (float): TODO.

		Returns:
			float: External irregularity index.
		"""
		return 1 - exp(-gamma * self.d(x_f, xnb_f))

	def II(self, x_f, xpb_f, theta):
		r"""Get internal irregularity index.

		Args:
			x_f (float): Individuals fitness/function value.
			xpb_f (float): Individuals personal best fitness/function value.
			theta (float): TODO.

		Returns:
			float: Internal irregularity index
		"""
		return 1 - exp(-theta * self.d(x_f, xpb_f))

	def getBestNeighbors(self, i, X, X_f, rs):
		r"""Get neighbors of individual.

		Mesurment of distance for neighborhud is defined with `self.nl`.
		Function for calculating distances is define with `self.dn`.

		Args:
			i (int): Index of individual for hum we are looking for neighbours.
			X (numpy.ndarray): Current population.
			X_f (numpy.ndarray[float]): Current population fitness/function values.
			rs (numpy.ndarray[float]): Distance between individuals.

		Returns:
			numpy.ndarray[int]: Indexes that represent individuals closest to `i`-th individual.
		"""
		nn = asarray([self.dn(X[i], X[j]) / rs for j in range(len(X))])
		return argmin(X_f[where(nn <= self.nl)])

	def uBestAndPBest(self, X, X_f, Xpb, Xpb_f):
		r"""Update personal best solution of all individuals in population.

		Args:
			X (numpy.ndarray): Current population.
			X_f (numpy.ndarray[float]): Current population fitness/function values.
			Xpb (numpy.ndarray): Current population best positions.
			Xpb_f (numpy.ndarray[float]): Current populations best positions fitness/function values.

		Returns:
			Tuple[numpy.ndarray, numpy.ndarray[float], numpy.ndarray, float]:
				1. New personal best positions for current population.
				2. New personal best positions function/fitness values for current population.
				3. New best individual.
				4. New best individual fitness/function value.
		"""
		ix_pb = where(X_f < Xpb_f)
		Xpb[ix_pb], Xpb_f[ix_pb] = X[ix_pb], X_f[ix_pb]
		return Xpb, Xpb_f

	def initPopulation(self, task):
		r"""Initialize first population and additional arguments.

		Args:
			task (Task): Optimization task

		Returns:
			Tuple[numpy.ndarray, numpy.ndarray, dict]:
				1. Initialized population
				2. Initialized population fitness/function values
				3. Dict[str, Any]:
					* Xpb (numpy.ndarray): Initialized populations best positions.
					* Xpb_f (numpy.ndarray): Initialized populations best positions function/fitness values.
					* alpha (numpy.ndarray):
					* gamma (numpy.ndarray):
					* theta (numpy.ndarray):
					* rs (float): Distance of search space.

		See Also:
			* :func:`NiaPy.algorithms.algorithm.Algorithm.initPopulation`
			* :func:`NiaPy.algorithms.other.aso.AnarchicSocietyOptimization.init`
		"""
		X, X_f, d = Algorithm.initPopulation(self, task)
		alpha, gamma, theta = self.init(task)
		Xpb, Xpb_f = self.uBestAndPBest(X, X_f, full([self.NP, task.D], 0.0), full(self.NP, task.optType.value * inf))
		d.update({'Xpb': Xpb, 'Xpb_f': Xpb_f, 'alpha': alpha, 'gamma': gamma, 'theta': theta, 'rs': self.d(task.Upper, task.Lower)})
		return X, X_f, d

	def runIteration(self, task, X, X_f, xb, fxb, Xpb, Xpb_f, alpha, gamma, theta, rs, **dparams):
		r"""Core function of AnarchicSocietyOptimization algorithm.

		Args:
			task (Task): Optimization task.
			X (numpy.ndarray): Current populations positions.
			X_f (numpy.ndarray): Current populations function/fitness values.
			xb (numpy.ndarray): Current global best individuals position.
			fxb (float): Current global best individual function/fitness value.
			Xpb (numpy.ndarray): Current populations best positions.
			Xpb_f (numpy.ndarray): Current population best positions function/fitness values.
			alpha (numpy.ndarray): TODO.
			gamma (numpy.ndarray):
			theta (numpy.ndarray):
			**dparams: Additional arguments.

		Returns:
			Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, float, dict]:
				1. Initialized population
				2. Initialized population fitness/function values
				3. New global best solution
				4. New global best solutions fitness/objective value
				5. Dict[str, Union[float, int, numpy.ndarray]:
					* Xpb (numpy.ndarray): Initialized populations best positions.
					* Xpb_f (numpy.ndarray): Initialized populations best positions function/fitness values.
					* alpha (numpy.ndarray):
					* gamma (numpy.ndarray):
					* theta (numpy.ndarray):
					* rs (float): Distance of search space.
		"""
		Xin = [self.getBestNeighbors(i, X, X_f, rs) for i in range(len(X))]
		MP_c, MP_s, MP_p = asarray([self.FI(X_f[i], Xpb_f[i], fxb, alpha[i]) for i in range(len(X))]), asarray([self.EI(X_f[i], X_f[Xin[i]], gamma[i]) for i in range(len(X))]), asarray([self.II(X_f[i], Xpb_f[i], theta[i]) for i in range(len(X))])
		Xtmp = asarray([self.Combination(X[i], Xpb[i], xb, X[self.randint(len(X), skip=[i])], MP_c[i], MP_s[i], MP_p[i], self.F, self.CR, task, self.Rand) for i in range(len(X))])
		X, X_f = asarray([Xtmp[i][0] for i in range(len(X))]), asarray([Xtmp[i][1] for i in range(len(X))])
		Xpb, Xpb_f = self.uBestAndPBest(X, X_f, Xpb, Xpb_f)
		xb, fxb = self.getBest(X, X_f, xb, fxb)
		return X, X_f, xb, fxb, {'Xpb': Xpb, 'Xpb_f': Xpb_f, 'alpha': alpha, 'gamma': gamma, 'theta': theta, 'rs': rs}

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
