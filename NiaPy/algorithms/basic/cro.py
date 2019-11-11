# encoding=utf8
import logging
from scipy.spatial.distance import euclidean
from numpy import apply_along_axis, argsort, where, random as rand, asarray, delete, sqrt, sum, unique, append
from NiaPy.algorithms.algorithm import Algorithm

logging.basicConfig()
logger = logging.getLogger('NiaPy.algorithms.basic')
logger.setLevel('INFO')

__all__ = ['CoralReefsOptimization']

def SexualCrossoverSimple(pop, p, task, rnd=rand, **kwargs):
	r"""Sexual reproduction of corals.

	Args:
		pop (numpy.ndarray): Current population.
		p (float): Probability in range [0, 1].
		task (Task): Optimization task.
		rnd (mtrand.RandomState): Random generator.
		**kwargs (Dict[str, Any]): Additional arguments.

	Returns:
		Tuple[numpy.ndarray, numpy.ndarray]:
			1. New population.
			2. New population function/fitness values.
	"""
	for i in range(len(pop) // 2): pop[i] = asarray([pop[i, d] if rnd.rand() < p else pop[i * 2, d] for d in range(task.D)])
	return pop, apply_along_axis(task.eval, 1, pop)

def BroodingSimple(pop, p, task, rnd=rand, **kwargs):
	r"""Brooding or internal sexual reproduction of corals.

	Args:
		pop (numpy.ndarray): Current population.
		p (float): Probability in range [0, 1].
		task (Task): Optimization task.
		rnd (mtrand.RandomState): Random generator.
		**kwargs (Dict[str, Any]): Additional arguments.

	Returns:
		Tuple[numpy.ndarray, numpy.ndarray]:
			1. New population.
			2. New population function/fitness values.
	"""
	for i in range(len(pop)): pop[i] = task.repair(asarray([pop[i, d] if rnd.rand() < p else task.Lower[d] + task.bRange[d] * rnd.rand() for d in range(task.D)]), rnd=rnd)
	return pop, apply_along_axis(task.eval, 1, pop)

def MoveCorals(pop, p, F, task, rnd=rand, **kwargs):
	r"""Move corals.

	Args:
		pop (numpy.ndarray): Current population.
		p (float): Probability in range [0, 1].
		F (float): Factor.
		task (Task): Optimization task.
		rnd (mtrand.RandomState): Random generator.
		**kwargs (Dict[str, Any]): Additional arguments.

	Returns:
		Tuple[numpy.ndarray, numpy.ndarray]:
			1. New population.
			2. New population function/fitness values.
	"""
	for i in range(len(pop)): pop[i] = task.repair(asarray([pop[i, d] if rnd.rand() < p else pop[i, d] + F * rnd.rand() for d in range(task.D)]), rnd=rnd)
	return pop, apply_along_axis(task.eval, 1, pop)

class CoralReefsOptimization(Algorithm):
	r"""Implementation of Coral Reefs Optimization Algorithm.

	Algorithm:
		Coral Reefs Optimization Algorithm

	Date:
		2018

	Authors:
		Klemen Berkovič

	License:
		MIT

	Reference Paper:
		S. Salcedo-Sanz, J. Del Ser, I. Landa-Torres, S. Gil-López, and J. A. Portilla-Figueras, “The Coral Reefs Optimization Algorithm: A Novel Metaheuristic for Efficiently Solving Optimization Problems,” The Scientific World Journal, vol. 2014, Article ID 739768, 15 pages, 2014.

	Reference URL:
		https://doi.org/10.1155/2014/739768.

	Attributes:
		Name (List[str]): List of strings representing algorithm name.
		phi (float): Range of neighborhood.
		Fa (int): Number of corals used in asexsual reproduction.
		Fb (int): Number of corals used in brooding.
		Fd (int): Number of corals used in depredation.
		k (int): Nomber of trys for larva setting.
		P_F (float): Mutation variable :math:`\in [0, \infty]`.
		P_Cr(float): Crossover rate in [0, 1].
		Distance (Callable[[numpy.ndarray, numpy.ndarray], float]): Funciton for calculating distance between corals.
		SexualCrossover (Callable[[numpy.ndarray, float, Task, mtrand.RandomState, Dict[str, Any]], Tuple[numpy.ndarray, numpy.ndarray[float]]]): Crossover function.
		Brooding (Callable[[numpy.ndarray, float, Task, mtrand.RandomState, Dict[str, Any]], Tuple[numpy.ndarray, numpy.ndarray]]): Brooding function.

	See Also:
		* :class:`NiaPy.algorithms.Algorithm`
	"""
	Name = ['CoralReefsOptimization', 'CRO']

	@staticmethod
	def typeParameters():
		r"""Get dictionary with functions for checking values of parameters.

		Returns:
			Dict[str, Callable]:
				* N (func): TODO
				* phi (func): TODO
				* Fa (func): TODO
				* Fb (func): TODO
				* Fd (func): TODO
				* k (func): TODO
		"""
		return {
			# TODO funkcije za testiranje
			'N': False,
			'phi': False,
			'Fa': False,
			'Fb': False,
			'Fd': False,
			'k': False
		}

	def setParameters(self, N=25, phi=0.4, Fa=0.5, Fb=0.5, Fd=0.3, k=25, P_Cr=0.5, P_F=0.36, SexualCrossover=SexualCrossoverSimple, Brooding=BroodingSimple, Distance=euclidean, **ukwargs):
		r"""Set the parameters of the algorithm.

		Arguments:
			N (int): population size for population initialization.
			phi (int): TODO.
			Fa (float): Value $\in [0, 1]$ for Asexual reproduction size.
			Fb (float): Value $\in [0, 1]$ for Brooding size.
			Fd (float): Value $\in [0, 1]$ for Depredation size.
			k (int): Trys for larvae setting.
			SexualCrossover (Callable[[numpy.ndarray, float, Task, mtrand.RandomState, Dict[str, Any]], Tuple[numpy.ndarray, numpy.ndarray]]): Crossover function.
			P_Cr (float): Crossover rate $\in [0, 1]$.
			Brooding (Callable[[numpy.ndarray, float, Task, mtrand.RandomState, Dict[str, Any]], Tuple[numpy.ndarray, numpy.ndarray]]): Brooding function.
			P_F (float): Crossover rate $\in [0, 1]$.
			Distance (Callable[[numpy.ndarray, numpy.ndarray], float]): Funciton for calculating distance between corals.

		See Also:
			* :func:`NiaPy.algorithms.Algorithm.setParameters`
		"""
		ukwargs.pop('NP', None)
		Algorithm.setParameters(self, NP=N, **ukwargs)
		self.phi, self.k, self.P_Cr, self.P_F = phi, k, P_Cr, P_F
		self.Fa, self.Fb, self.Fd = int(self.NP * Fa), int(self.NP * Fb), int(self.NP * Fd)
		self.SexualCrossover, self.Brooding, self.Distance = SexualCrossover, Brooding, Distance

	def getParameters(self):
		r"""Get parameters values of the algorithm.

		Returns:
			Dict[str, Any]: TODO.
		"""
		d = Algorithm.getParameters(self)
		d.update({
			'phi': self.phi,
			'k': self.k,
			'P_Cr': self.P_Cr,
			'P_F': self.P_F,
			'Fa': self.Fa,
			'Fd': self.Fd,
			'Fb': self.Fb
		})
		return d

	def asexualReprodution(self, Reef, Reef_f, xb, fxb, task):
		r"""Asexual reproduction of corals.

		Args:
			Reef (numpy.ndarray): Current population of reefs.
			Reef_f (numpy.ndarray): Current populations function/fitness values.
			task (Task): Optimization task.

		Returns:
			Tuple[numpy.ndarray, numpy.ndarray]:
				1. New population.
				2. New population fitness/funciton values.

		See Also:
			* :func:`NiaPy.algorithms.basic.CoralReefsOptimization.setting`
			* :func:`NiaPy.algorithms.basic.BroodingSimple`
		"""
		I = argsort(Reef_f)[:self.Fa]
		Reefn, Reefn_f = self.Brooding(Reef[I], self.P_F, task, rnd=self.Rand)
		xb, fxb = self.getBest(Reefn, Reefn_f, xb, fxb)
		Reef, Reef_f, xb, fxb = self.setting(Reef, Reef_f, Reefn, Reefn_f, xb, fxb, task)
		return Reef, Reef_f, xb, fxb

	def depredation(self, Reef, Reef_f):
		r"""Depredation operator for reefs.

		Args:
			Reef (numpy.ndarray): Current reefs.
			Reef_f (numpy.ndarray): Current reefs function/fitness values.

		Returns:
			Tuple[numpy.ndarray, numpy.ndarray]:
				1. Best individual
				2. Best individual fitness/function value
		"""
		I = argsort(Reef_f)[::-1][:self.Fd]
		return delete(Reef, I), delete(Reef_f, I)

	def setting(self, X, X_f, Xn, Xn_f, xb, fxb, task):
		r"""Operator for setting reefs.

		New reefs try to seatle to selected position in search space.
		New reefs are successful if theyr fitness values is better or if they have no reef ocupying same search space.

		Args:
			X (numpy.ndarray): Current population of reefs.
			X_f (numpy.ndarray): Current populations function/fitness values.
			Xn (numpy.ndarray): New population of reefs.
			Xn_f (array of float): New populations function/fitness values.
			xb (numpy.ndarray): Global best solution.
			fxb (float): Global best solutions fitness/objective value.
			task (Task): Optimization task.

		Returns:
			Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, float]:
				1. New seatled population.
				2. New seatled population fitness/function values.
		"""
		def update(A, phi, xb, fxb):
			D = asarray([sqrt(sum((A - e) ** 2, axis=1)) for e in Xn])
			I = unique(where(D < phi)[0])
			if I.any():
				Xn[I], Xn_f[I] = MoveCorals(Xn[I], self.P_F, self.P_F, task, rnd=self.Rand)
				xb, fxb = self.getBest(Xn[I], Xn_f[I], xb, fxb)
			return xb, fxb
		for i in range(self.k):
			xb, fxb = update(X, self.phi, xb, fxb)
			xb, fxb = update(Xn, self.phi, xb, fxb)
		D = asarray([sqrt(sum((X - e) ** 2, axis=1)) for e in Xn])
		I = unique(where(D >= self.phi)[0])
		return append(X, Xn[I], 0), append(X_f, Xn_f[I], 0), xb, fxb

	def runIteration(self, task, Reef, Reef_f, xb, fxb, **dparams):
		r"""Core function of Coral Reefs Optimization algorithm.

		Args:
			task (Task): Optimization task.
			Reef (numpy.ndarray): Current population.
			Reef_f (numpy.ndarray): Current population fitness/function value.
			xb (numpy.ndarray): Global best solution.
			fxb (float): Global best solution fitness/function value.
			**dparams: Additional arguments

		Returns:
			Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, float, Dict[str, Any]]:
				1. New population.
				2. New population fitness/function values.
				3. New global bset solution
				4. New global best solutions fitness/objective value
				5. Additional arguments:

		See Also:
			* :func:`NiaPy.algorithms.basic.CoralReefsOptimization.SexualCrossover`
			* :func:`NiaPy.algorithms.basic.CoralReefsOptimization.Brooding`
		"""
		I = self.Rand.choice(len(Reef), size=self.Fb, replace=False)
		Reefn_s, Reefn_s_f = self.SexualCrossover(Reef[I], self.P_Cr, task, rnd=self.Rand)
		xb, fxb = self.getBest(Reefn_s, Reefn_s_f, xb, fxb)
		Reefn_b, Reffn_b_f = self.Brooding(delete(Reef, I, 0), self.P_F, task, rnd=self.Rand)
		xb, fxb = self.getBest(Reefn_s, Reefn_s_f, xb, fxb)
		Reefn, Reefn_f, xb, fxb = self.setting(Reef, Reef_f, append(Reefn_s, Reefn_b, 0), append(Reefn_s_f, Reffn_b_f, 0), xb, fxb, task)
		Reef, Reef_f, xb, fxb = self.asexualReprodution(Reefn, Reefn_f, xb, fxb, task)
		if task.Iters % self.k == 0: Reef, Reef_f = self.depredation(Reef, Reef_f)
		return Reef, Reef_f, xb, fxb, {}

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
