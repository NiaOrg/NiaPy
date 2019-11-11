# coding=utf-8
import logging
from numpy import array, argsort, where
from NiaPy.algorithms.algorithm import Algorithm

__all__ = ['BeesAlgorithm']

logging.basicConfig()
logger = logging.getLogger('NiaPy.algorithms.basic')
logger.setLevel('INFO')


class BeesAlgorithm(Algorithm):
	r"""Implementation of Bees algorithm.

	Algorithm:
		 The Bees algorithm

	Date:
		 2019

	Authors:
		 Rok Potočnik

	License:
		 MIT

	Reference paper:
		 DT Pham, A Ghanbarzadeh, E Koc, S Otri, S Rahim, and M Zaidi. The bees algorithm-a novel tool for complex optimisation problems. In Proceedings of the 2nd Virtual International Conference on Intelligent Production Machines and Systems (IPROMS 2006), pages 454–459, 2006

	Attributes:
		 NP (Optional[int]): Number of scout bees parameter.
		 m (Optional[int]): Number of sites selected out of n visited sites parameter.
		 e (Optional[int]): Number of best sites out of m selected sitest parameter.
		 nep (Optional[int]): Number of bees recruited for best e sites parameter.
		 nsp (Optional[int]): Number of bees recruited for the other selected sites parameter.
		 ngh (Optional[float]): Initial size of patches parameter.
		 ukwargs (Dict[str, Any]): Additional arguments.

	See Also:
		 * :func:`NiaPy.algorithms.Algorithm.setParameters`

	"""
	Name = ['BeesAlgorithm', 'BEA']
	@staticmethod
	def algorithmInfo():
		return r"""
        Description: A new population-based search algorithm called the Bees Algorithm (BA) is presented. The algorithm mimics the food foraging behaviour of swarms of honey bees.
        Authors: D.T. Pham, A. Ghanbarzadeh,  E. Koç, S. Otri,  S. Rahim, M. Zaidi
        Year: 2006
        Main reference: DT Pham, A Ghanbarzadeh, E Koc, S Otri, S Rahim, and M Zaidi. The bees algorithm-a novel tool for complex optimisation problems. In Proceedings of the 2nd Virtual International Conference on Intelligent Production Machines and Systems (IPROMS 2006), pages 454–459, 2006
    """

	@staticmethod
	def typeParameters():
		r"""Get dictionary with functions for checking values of parameters.

		Returns:
			 Dict[str, Callable]:
				  * NP (Callable[[int], bool]): Checks if number of bees parameter has a proper value.
				  * m (Callable[[int], bool]): Checks if number of selected sites parameter has a proper value.
				  * e (Callable[[int], bool]): Checks if number of elite selected sites parameter has a proper value.
				  * nep (Callable[[int], bool]): Checks if number of elite bees parameter has a proper value.
				  * nsp (Callable[[int], bool]): Checks if number of other bees parameter has a proper value.
				  * ngh (Callable[[float], bool]): Checks if size of patches parameter has a proper value.

		See Also:
			 * :func:`NiaPy.algorithms.algorithm.Algorithm.typeParameters`
		"""
		d = Algorithm.typeParameters()
		d.update({
			'NP': lambda x: isinstance(x, int) and x > 0,
			'm': lambda x: isinstance(x, int) and x > 0,
			'e': lambda x: isinstance(x, int) and x > 0,
			'nep': lambda x: isinstance(x, int) and x > 0,
			'nsp': lambda x: isinstance(x, int) and x > 0,
			'ngh': lambda x: isinstance(x, float) and x > 0
		})
		return d

	def setParameters(self, NP=40, m=5, e=4, ngh=1, nep=4, nsp=2, **ukwargs):
		r"""Set the parameters of the algorithm.

		Args:
			 NP (Optional[int]): Number of scout bees parameter.
			 m (Optional[int]): Number of sites selected out of n visited sites parameter.
			 e (Optional[int]): Number of best sites out of m selected sitest parameter.
			 nep (Optional[int]): Number of bees recruited for best e sites parameter.
			 nsp (Optional[int]): Number of bees recruited for the other selected sites parameter.
			 ngh (Optional[float]): Initial size of patches parameter.
			 ukwargs (Dict[str, Any]): Additional arguments.

		See Also:
			 * :func:`NiaPy.algorithms.Algorithm.setParameters`
		"""
		Algorithm.setParameters(self, NP=NP, **ukwargs)
		self.n, self.m, self.e, self.nep, self.nsp, self.ngh = NP, m, e, nep, nsp, ngh

	def getParameters(self):
		r"""Get parameters of the algorithm.

		Returns:
			 Dict[str, Any]:
		"""
		d = Algorithm.getParameters(self)
		d.update({
			'm': self.m,
			'e': self.e,
			'ngh': self.ngh,
			'nep': self.nep,
			'nsp': self.nsp
		})
		return d

	def beeDance(self, x, task, ngh):
		r"""Bees Dance. Search for new positions.

		Args:
			 x (numpy.ndarray): One instance from the population.
			 task (Task): Optimization task
			 ngh (float): A small value for patch search.

		Returns:
			 Tuple[numpy.ndarray, float]:
				  1. New population.
				  2. New population fitness/function values.

		See Also:
			 * :func:`NiaPy.algorithms.Algorithm.initPopulation`
		"""
		ind = self.randint(task.D)
		y = array(x, copy=True)
		y[ind] = x[ind] + self.uniform(-ngh, ngh)
		y = self.repair(y, task.Lower, task.Upper)
		res = task.eval(y)
		return y, res

	def initPopulation(self, task):
		r"""Initialize the starting population.

		Args:
			 task (Task): Optimization task

		Returns:
			 Tuple[numpy.ndarray, numpy.ndarray, Dict[str, Any]]:
				  1. New population.
				  2. New population fitness/function values.

		See Also:
			 * :func:`NiaPy.algorithms.Algorithm.initPopulation`
		"""
		BeesPosition, BeesCost, _ = Algorithm.initPopulation(self, task)

		idxs = argsort(BeesCost)
		BeesCost = BeesCost[idxs]
		BeesPosition = BeesPosition[idxs, :]

		return BeesPosition, BeesCost, {'ngh': self.ngh}

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

	def runIteration(self, task, BeesPosition, BeesCost, xb, fxb, ngh, **dparams):
		r"""Core function of Forest Optimization Algorithm.

		Args:
			 task (Task): Optimization task.
			 BeesPosition (numpy.ndarray[float]): Current population.
			 BeesCost (numpy.ndarray[float]): Current population function/fitness values.
			 xb (numpy.ndarray): Global best individual.
			 fxb (float): Global best individual fitness/function value.
			 ngh (float): A small value used for patches.
			 **dparams (Dict[str, Any]): Additional arguments.

		Returns:
			 Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, float, Dict[str, Any]]:
				  1. New population.
				  2. New population fitness/function values.
				  3. New global best solution
				  4. New global best fitness/objective value
				  5. Additional arguments:
						* ngh (float): A small value used for patches.
		"""
		for ies in range(self.e):
			BestBeeCost = float('inf')
			for ieb in range(self.nep):
				NewBeePos, NewBeeCost = self.beeDance(BeesPosition[ies, :], task, ngh)
				if NewBeeCost < BestBeeCost:
					BestBeeCost = NewBeeCost
					BestBeePos = NewBeePos
			if BestBeeCost < BeesCost[ies]:
				BeesPosition[ies, :] = BestBeePos
				BeesCost[ies] = BestBeeCost
		for ies in range(self.e, self.m):
			BestBeeCost = float('inf')
			for ieb in range(self.nsp):
				NewBeePos, NewBeeCost = self.beeDance(BeesPosition[ies, :], task, ngh)
				if NewBeeCost < BestBeeCost:
					BestBeeCost = NewBeeCost
					BestBeePos = NewBeePos
			if BestBeeCost < BeesCost[ies]:
				BeesPosition[ies, :] = BestBeePos
				BeesCost[ies] = BestBeeCost
		for ies in range(self.m, self.n):
			BeesPosition[ies, :] = array(self.uniform(task.Lower, task.Upper, task.D))
			BeesCost[ies] = task.eval(BeesPosition[ies, :])
		idxs = argsort(BeesCost)
		BeesCost = BeesCost[idxs]
		BeesPosition = BeesPosition[idxs, :]
		ngh = ngh * 0.95
		return BeesPosition, BeesCost, BeesPosition[0].copy(), BeesCost[0], {'ngh': ngh}

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
