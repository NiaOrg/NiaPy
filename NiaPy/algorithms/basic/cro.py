# encoding=utf8
# pylint: disable=mixed-indentation, line-too-long, singleton-comparison, multiple-statements, attribute-defined-outside-init, no-self-use, logging-not-lazy, unused-variable, arguments-differ, bad-continuation, redefined-builtin, unused-argument, consider-using-enumerate, expression-not-assigned
import logging
from scipy.spatial.distance import euclidean
from numpy import apply_along_axis, argsort, where, inf, random as rand, asarray, delete, sqrt, sum, unique, append
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
		Tuple[numpy.ndarray, numpy.ndarray[float]]:
			1. New population.
			2. New population function/fitness values.
	"""
	for i in range(len(pop) // 2): pop[i] = asarray([pop[i, d] if rnd.rand() < p else pop[i * 2, d] for d in range(task.D)])
	return pop, apply_along_axis(task.eval, 1, pop)

def BroodingSimple(pop, p, task, rnd=rand, **kwargs):
	r"""Brooading of corals.

	Args:
		pop (numpy.ndarray): Current population.
		p (float): Probability in range [0, 1].
		task (Task): Optimization task.
		rnd (mtrand.RandomState): Random generator.
		**kwargs (Dict[str, Any]): Additional arguments.

	Returns:
		Tuple[numpy.ndarray, numpy.ndarray[float]]:
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
		F (float): Factor TODO.
		task (Task): Optimization task.
		rnd (mtrand.RandomState): Random generator.
		**kwargs (Dict[str, Any]): Additional arguments.

	Returns:
		Tuple[numpy.ndarray, numpy.ndarray[float]]:
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

	Reference:
		S. Salcedo-Sanz, J. Del Ser, I. Landa-Torres, S. Gil-López, and J. A. Portilla-Figueras, “The Coral Reefs Optimization Algorithm: A Novel Metaheuristic for Efficiently Solving Optimization Problems,” The Scientific World Journal, vol. 2014, Article ID 739768, 15 pages, 2014. https://doi.org/10.1155/2014/739768.

	Attributes:
		Name (List[str]): List of strings representing algorithm name.
		phi (int): TODO.
		Fa (int): TODO.
		Fb (int): Value in [0, 1] for Brooding size.
		Fd (int): Value in [0, 1] for Depredation.
		k (int): Nomber of trys for larva setting.
		P_F (float): Mutation variable :math:`\in [0, \infty]`.
		P_Cr(float): Crossover rate in [0, 1].
		Distance (Callable[[numpy.ndarray, numpy.ndarray], float]): Funciton for calculating distance between corals.
		SexualCrossover (Callable[[numpy.ndarray, float, Task, mtrand.RandomState, Dict[str, Any]], Tuple[numpy.ndarray, numpy.ndarray[float]]]): Crossover function.
		Brooding (Callable[[numpy.ndarray, float, Task, mtrand.RandomState, Dict[str, Any]], Tuple[numpy.ndarray, numpy.ndarray[float]]]): Brooding function.

	See Also:
		:class:`NiaPy.algorithms.algorithm.Algorithm`
	"""
	Name = ['CoralReefsOptimization', 'CRO']

	@staticmethod
	def typeParameters():
		r"""TODO.

		Returns:
			dict:
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

	def setParameters(self, N=25, phi=10, Fa=0.5, Fb=0.5, Fd=0.3, k=25, P_Cr=0.5, P_F=0.36, SexualCrossover=SexualCrossoverSimple, Brooding=BroodingSimple, Distance=euclidean, **ukwargs):
		r"""Set the parameters of the algorithm.

		Arguments:
			N (int): population size for population initialization.
			phi (int): TODO.
			Fa (float): TODO.
			Fb (float): Value $\in [0, 1]$ for Brooding size.
			Fd (float): Value $\in [0, 1]$ for Depredation.
			k (int): Trys for larvae setting.
			SexualCrossover (Callable[[numpy.ndarray, float, Task, mtrand.RandomState, Dict[str, Any]], Tuple[numpy.ndarray, numpy.ndarray[float]]]): Crossover function.
			P_Cr (float): Crossover rate $\in [0, 1]$.
			Brooding (Callable[[numpy.ndarray, float, Task, mtrand.RandomState, Dict[str, Any]], Tuple[numpy.ndarray, numpy.ndarray[float]]]): Brooding function.
			P_F (float): Crossover rate $\in [0, 1]$.
			Distance (Callable[[numpy.ndarray, numpy.ndarray], float]): Funciton for calculating distance between corals.

		See Also:
			:func:`NiaPy.algorithms.algorithm.Algorithm.setParameters`
		"""
		Algorithm.setParameters(self, NP=N)
		self.phi, self.Fa, self.Fb, self.Fd, self.k, self.P_Cr, self.P_F = phi, Fa, Fb, Fd, k, P_Cr, P_F
		self.SexualCrossover, self.Brooding, self.Distance = SexualCrossover, Brooding, Distance
		if ukwargs: logger.info('Unused arguments: %s' % (ukwargs))

	def initPopulation(self, task):
		r"""Initialize starting population.

		Args:
			task (Task): Optimization task.

		Returns:
			Tuple[numpy.ndarray, numpy.ndarray[float], Dict[str, Any]]:
				1. Initialized population.
				2. Initialized population fitness/function values.
				3. Additional arguments:
					* Fa (int): TODO
					* Fb (int): TODO
					* Fd (int): TODO

		See Also:
			:func:`NiaPy.algorithms.algorithm.Algorithm.initPopulation`
		"""
		Reef, Reef_f, d = Algorithm.initPopulation(self, task)
		Fa, Fb, Fd = self.NP * self.Fa, self.NP * self.Fb, self.NP * self.Fd
		if Fa % 2 != 0: Fa += 1
		d.update({'Fa':int(Fa), 'Fb':int(Fb), 'Fd':int(Fd)})
		return Reef, Reef_f, d

	def asexualReprodution(self, Reef, Reef_f, Fa, task):
		r"""Asexual reproduction of corals.

		Args:
			Reef (numpy.ndarray): Current population of reefs.
			Reef_f (numpy.ndarray[float]): Current populations function/fitness values.
			Fa (int): Number of corals that are used in reproduction.
			task (Task): Optimization task.

		Returns:
			Tuple[numpy.ndarray, numpy.ndarray[float]]:
				1. New population.
				2. New population fitness/funciton values.

		See Also:
			* :func:`NiaPy.algorithms.basic.CoralReefsOptimization.setting`
			* :func:`NiaPy.algorithms.basic.BroodingSimple`
		"""
		I = argsort(Reef_f)[:Fa]
		Reefn, Reefn_f = self.Brooding(Reef[I], self.P_F, task, rnd=self.Rand)
		Reef, Reef_f = self.setting(Reef, Reef_f, Reefn, Reefn_f, task)
		return Reef, Reef_f

	def depredation(self, Reef, Reef_f, Fd):
		r"""Depredation operator for reefs.

		Args:
			Reef (numpy.ndarray): Current reefs.
			Reef_f (numpy.ndarray[float]): Current reefs function/fitness values.
			Fd (int): TODO

		Returns:
			Tuple[numpy.ndarray, numpy.ndarray[float]]:
				1. Best individual
				2. Best individual fitness/function value
		"""
		I = argsort(Reef_f)[::-1][:Fd]
		return delete(Reef, I), delete(Reef_f, I)

	def setting(self, X, X_f, Xn, Xn_f, task):
		r"""Settings operator for reefs.

		New reefs try to seatle to selected position in search space.
		New reefs are successful if theyr fitness values is better or if they have no reef ocupying same search space.

		Args:
			X (numpy.ndarray): Current population of reefs.
			X_f (numpy.ndarray[float]): Current populations function/fitness values.
			Xn (numpy.ndarray): New population of reefs.
			Xn_f (array of float): New populations function/fitness values.
			task (Task): Optimization task.

		Returns:
			Tuple[numpy.ndarray, numpy.ndarray[float]]:
				1. New seatled population.
				2. New seatled population fitness/function values.
		"""
		def update(A):
			D = asarray([sqrt(sum((A - e) ** 2, axis=1)) for e in Xn])
			I = unique(where(D < self.phi)[0])
			if I.any(): Xn[I], Xn_f[I] = MoveCorals(Xn[I], self.P_F, self.P_F, task, rnd=self.Rand)
		for i in range(self.k): update(X), update(Xn)
		D = asarray([sqrt(sum((X - e) ** 2, axis=1)) for e in Xn])
		I = unique(where(D >= self.phi)[0])
		return append(X, Xn[I], 0), append(X_f, Xn_f[I], 0)

	def runIteration(self, task, Reef, Reef_f, xb, fxb, Fa, Fb, Fd, **dparams):
		r"""Core function of Coral Reefs Optimization algorithm.

		Args:
			task (Task): Optimization task.
			Reef (numpy.ndarray): Current population.
			Reef_f (numpy.ndarray[float]): Current population fitness/function value.
			xb (numpy.ndarray): Global best solution.
			fxb (float): Global best solution fitness/function value.
			Fa (int): TODO
			Fb (int): TODO
			Fd (int): TODO
			**dparams: Additional arguments

		Returns:
			Tuple[numpy.ndarray, numpy.ndarray[float], Dict[str, Any]]:
				1. New population.
				2. New population fitness/function values.
				3. Additional arguments:
					* Fa (int): TODO
					* Fb (int): TODO
					* Fd (int): TODO

		See Also:
			* :func:`NiaPy.algorithms.basic.CoralReefsOptimization.SexualCrossover`
			* :func:`NiaPy.algorithms.basic.CoralReefsOptimization.Brooding`
		"""
		I = self.Rand.choice(len(Reef), size=Fb, replace=False)
		Reefn_s, Reefn_s_f = self.SexualCrossover(Reef[I], self.P_Cr, task, rnd=self.Rand)
		Reefn_b, Reffn_b_f = self.Brooding(delete(Reef, I, 0), self.P_F, task, rnd=self.Rand)
		Reefn, Reefn_f = self.setting(Reef, Reef_f, append(Reefn_s, Reefn_b, 0), append(Reefn_s_f, Reffn_b_f, 0), task)
		Reef, Reef_f = self.asexualReprodution(Reefn, Reefn_f, Fa, task)
		if task.Iters % self.k == 0: Reef, Reef_f = self.depredation(Reef, Reef_f, Fd)
		return Reef, Reef_f, {'Fa':Fa, 'Fb':Fb, 'Fd':Fd}

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
