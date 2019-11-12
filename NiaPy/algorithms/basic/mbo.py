# encoding=utf8
import logging

from numpy import argsort, sum, apply_along_axis, where, pi, ceil, isinf, array, copy, tan
from numpy.random import exponential

from NiaPy.algorithms.algorithm import Algorithm

__all__ = ['MonarchButterflyOptimization']

logging.basicConfig()
logger = logging.getLogger('NiaPy.algorithms.basic')
logger.setLevel('INFO')

class MonarchButterflyOptimization(Algorithm):
	r"""Implementation of Monarch Butterfly Optimization.

	Algorithm:
		 Monarch Butterfly Optimization

	Date:
		 2019

	Authors:
		 Jan Banko

	License:
		 MIT

	Reference paper:
		 Wang, Gai-Ge & Deb, Suash & Cui, Zhihua. (2015). Monarch Butterfly Optimization. Neural Computing and Applications. 10.1007/s00521-015-1923-y. , https://www.researchgate.net/publication/275964443_Monarch_Butterfly_Optimization.

	Attributes:
		 Name (List[str]): List of strings representing algorithm name.
		 PAR (float): Partition.
		 PER (float): Period.

	See Also:
		 * :class:`NiaPy.algorithms.Algorithm`
	"""
	Name = ['MonarchButterflyOptimization', 'MBO']

	@staticmethod
	def algorithmInfo():
		r"""Get information of the algorithm.

		Returns:
			str: Algorithm information.

		See Also:
			 * :func:`NiaPy.algorithms.algorithm.Algorithm.algorithmInfo`
		"""
		return r"""
		Description: Monarch butterfly optimization algorithm is inspired by the migration behaviour of the monarch butterflies in nature.
		Authors: Wang, Gai-Ge & Deb, Suash & Cui, Zhihua.
		Year: 2015
		Main reference: Wang, Gai-Ge & Deb, Suash & Cui, Zhihua. (2015). Monarch Butterfly Optimization. Neural Computing and Applications. 10.1007/s00521-015-1923-y. , https://www.researchgate.net/publication/275964443_Monarch_Butterfly_Optimization.
    """

	@staticmethod
	def typeParameters():
		r"""Get dictionary with functions for checking values of parameters.

		Returns:
			 Dict[str, Callable]:
				  * PAR (Callable[[float], bool]): Checks if partition parameter has a proper value.
				  * PER (Callable[[float], bool]): Checks if period parameter has a proper value.
		See Also:
			 * :func:`NiaPy.algorithms.algorithm.Algorithm.typeParameters`
		"""
		d = Algorithm.typeParameters()
		d.update({
			'PAR': lambda x: isinstance(x, float) and x > 0,
			'PER': lambda x: isinstance(x, float) and x > 0
		})
		return d

	def setParameters(self, NP=20, PAR=5.0 / 12.0, PER=1.2, **ukwargs):
		r"""Set the parameters of the algorithm.

		Args:
			 NP (Optional[int]): Population size.
			 PAR (Optional[int]): Partition.
			 PER (Optional[int]): Period.
			 ukwargs (Dict[str, Any]): Additional arguments.

		See Also:
			 * :func:`NiaPy.algorithms.Algorithm.setParameters`
		"""
		Algorithm.setParameters(self, NP=NP, **ukwargs)
		self.NP, self.PAR, self.PER, self.keep, self.BAR, self.NP1 = NP, PAR, PER, 2, PAR, int(ceil(PAR * NP))
		self.NP2 = int(NP - self.NP1)

	def getParameters(self):
		r"""Get parameters values for the algorithm.

		Returns:
			Dict[str, Any]: TODO.
		"""
		d = Algorithm.getParameters(self)
		d.update({
			'PAR': self.PAR,
			'PER': self.PER,
			'keep': self.keep,
			'BAR': self.BAR,
			'NP1': self.NP1,
			'NP2': self.NP2
		})
		return d

	def repair(self, x, lower, upper):
		r"""Truncate exceeded dimensions to the limits.

		Args:
			 x (numpy.ndarray): Individual to repair.
			 lower (numpy.ndarray): Lower limits for dimensions.
			 upper (numpy.ndarray): Upper limits for dimensions.

		Returns:
			 numpy.ndarray: Repaired individual.
		"""
		ir = where(x < lower)
		x[ir] = lower[ir]
		ir = where(x > upper)
		x[ir] = upper[ir]
		return x

	def levy(self, step_size, D):
		r"""Calculate levy flight.

		Args:
			 step_size (float): Size of the walk step.
			 D (int): Number of dimensions.

		Returns:
			 numpy.ndarray: Calculated values for levy flight.
		"""
		delataX = array([sum(tan(pi * self.uniform(0.0, 1.0, 10))) for _ in range(0, D)])
		return delataX

	def migrationOperator(self, D, NP1, NP2, Butterflies):
		r"""Apply the migration operator.

		Args:
			 D (int): Number of dimensions.
			 NP1 (int): Number of butterflies in Land 1.
			 NP2 (int): Number of butterflies in Land 2.
			 Butterflies (numpy.ndarray): Current butterfly population.

		Returns:
			 numpy.ndarray: Adjusted butterfly population.
		"""
		pop1 = copy(Butterflies[:NP1])
		pop2 = copy(Butterflies[NP1:])
		for k1 in range(0, NP1):
			for parnum1 in range(0, D):
				r1 = self.uniform(0.0, 1.0) * self.PER
				if r1 <= self.PAR:
					r2 = self.randint(Nmin=0, Nmax=NP1 - 1)
					Butterflies[k1, parnum1] = pop1[r2, parnum1]
				else:
					r3 = self.randint(Nmin=0, Nmax=NP2 - 1)
					Butterflies[k1, parnum1] = pop2[r3, parnum1]
		return Butterflies

	def adjustingOperator(self, t, max_t, D, NP1, NP2, Butterflies, best):
		r"""Apply the adjusting operator.

		Args:
			 t (int): Current generation.
			 max_t (int): Maximum generation.
			 D (int): Number of dimensions.
			 NP1 (int): Number of butterflies in Land 1.
			 NP2 (int): Number of butterflies in Land 2.
			 Butterflies (numpy.ndarray): Current butterfly population.
			 best (numpy.ndarray): The best butterfly currently.

		Returns:
			 numpy.ndarray: Adjusted butterfly population.
		"""
		pop2 = copy(Butterflies[NP1:])
		for k2 in range(NP1, NP1 + NP2):
			scale = 1.0 / ((t + 1)**2)
			step_size = ceil(exponential(2 * max_t))
			delataX = self.levy(step_size, D)
			for parnum2 in range(0, D):
				if self.uniform(0.0, 1.0) >= self.PAR:
					Butterflies[k2, parnum2] = best[parnum2]
				else:
					r4 = self.randint(Nmin=0, Nmax=NP2 - 1)
					Butterflies[k2, parnum2] = pop2[r4, 1]
					if self.uniform(0.0, 1.0) > self.BAR:
						Butterflies[k2, parnum2] += scale * \
															 (delataX[parnum2] - 0.5)
		return Butterflies

	def evaluateAndSort(self, task, Butterflies):
		r"""Evaluate and sort the butterfly population.

		Args:
			 task (Task): Optimization task
			 Butterflies (numpy.ndarray): Current butterfly population.

		Returns:
			 numpy.ndarray: Tuple[numpy.ndarray, float, numpy.ndarray]:
				  1. Best butterfly according to the evaluation.
				  2. The best fitness value.
				  3. Butterfly population.
		"""
		Fitness = apply_along_axis(task.eval, 1, Butterflies)
		indices = argsort(Fitness)
		Butterflies = Butterflies[indices]
		Fitness = Fitness[indices]

		return Fitness, Butterflies

	def initPopulation(self, task):
		r"""Initialize the starting population.

		Args:
			 task (Task): Optimization task

		Returns:
			 Tuple[numpy.ndarray, numpy.ndarray[float], Dict[str, Any]]:
				  1. New population.
				  2. New population fitness/function values.
				  3. Additional arguments:
						* dx (float): A small value used in local seeding stage.

		See Also:
			 * :func:`NiaPy.algorithms.Algorithm.initPopulation`
		"""
		Butterflies = self.uniform(task.Lower, task.Upper, [self.NP, task.D])
		Fitness, Butterflies = self.evaluateAndSort(task, Butterflies)
		return Butterflies, Fitness, {'tmp_best': Butterflies[0]}

	def runIteration(self, task, Butterflies, Evaluations, xb, fxb, tmp_best, **dparams):
		r"""Core function of Forest Optimization Algorithm.

		Args:
			 task (Task): Optimization task.
			 Butterflies (numpy.ndarray): Current population.
			 Evaluations (numpy.ndarray[float]): Current population function/fitness values.
			 xb (numpy.ndarray): Global best individual.
			 fxb (float): Global best individual fitness/function value.
			 tmp_best (numpy.ndarray): Best individual currently.
			 **dparams (Dict[str, Any]): Additional arguments.

		Returns:
			 Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, float, Dict[str, Any]]:
				  1. New population.
				  2. New population fitness/function values.
				  3. New global best solution
				  4. New global best solutions fitness/objective value
				  5. Additional arguments:
						* dx (float): A small value used in local seeding stage.
		"""
		tmpElite = copy(Butterflies[:self.keep])
		max_t = task.nGEN if isinf(task.nGEN) is False else task.nFES / self.NP
		Butterflies = apply_along_axis(self.repair, 1, self.migrationOperator(task.D, self.NP1, self.NP2, Butterflies), task.Lower, task.Upper)
		Butterflies = apply_along_axis(self.repair, 1, self.adjustingOperator(task.Iters, max_t, task.D, self.NP1, self.NP2, Butterflies, tmp_best), task.Lower, task.Upper)
		Fitness, Butterflies = self.evaluateAndSort(task, Butterflies)
		tmp_best = Butterflies[0]
		Butterflies[-self.keep:] = tmpElite
		Fitness, Butterflies = self.evaluateAndSort(task, Butterflies)
		xb, fxb = self.getBest(Butterflies, Fitness, xb, fxb)
		return Butterflies, Fitness, xb, fxb, {'tmp_best': tmp_best}

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
