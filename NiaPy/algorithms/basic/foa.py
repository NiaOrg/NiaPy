# encoding=utf8
import logging
from numpy import where, apply_along_axis, zeros, append, ndarray, delete, arange, argmin, absolute, int32
from NiaPy.algorithms.algorithm import Algorithm
from NiaPy.util import limit_repair

__all__ = ['ForestOptimizationAlgorithm']

logging.basicConfig()
logger = logging.getLogger('NiaPy.algorithms.basic')
logger.setLevel('INFO')

class ForestOptimizationAlgorithm(Algorithm):
	r"""Implementation of Forest Optimization Algorithm.

	Algorithm:
		 Forest Optimization Algorithm

	Date:
		 2019

	Authors:
		 Luka Pečnik

	License:
		 MIT

	Reference paper:
		 Manizheh Ghaemi, Mohammad-Reza Feizi-Derakhshi, Forest Optimization Algorithm, Expert Systems with Applications, Volume 41, Issue 15, 2014, Pages 6676-6687, ISSN 0957-4174, https://doi.org/10.1016/j.eswa.2014.05.009.

	References URL:
		 Implementation is based on the following MATLAB code: https://github.com/cominsys/FOA

	Attributes:
		 Name (List[str]): List of strings representing algorithm name.
		 lt (int): Life time of trees parameter.
		 al (int): Area limit parameter.
		 lsc (int): Local seeding changes parameter.
		 gsc (int): Global seeding changes parameter.
		 tr (float): Transfer rate parameter.

	See Also:
		 * :class:`NiaPy.algorithms.Algorithm`
	"""
	Name = ['ForestOptimizationAlgorithm', 'FOA']

	@staticmethod
	def algorithmInfo():
		r"""Get algorithms information.

		Returns:
			str: Algorithm information.
		"""
		return r"""
        Description: Forest Optimization Algorithm is inspired by few trees in the forests which can survive for several decades, while other trees could live for a limited period.
        Authors: Manizheh Ghaemi, Mohammad-Reza Feizi-Derakhshi
        Year: 2014
        Main reference: Manizheh Ghaemi, Mohammad-Reza Feizi-Derakhshi, Forest Optimization Algorithm, Expert Systems with Applications, Volume 41, Issue 15, 2014, Pages 6676-6687, ISSN 0957-4174, https://doi.org/10.1016/j.eswa.2014.05.009.
        """

	@staticmethod
	def typeParameters():
		r"""Get dictionary with functions for checking values of parameters.

		Returns:
			 Dict[str, Callable]:
				  * lt (Callable[[int], bool]): Checks if life time parameter has a proper value.
				  * al (Callable[[int], bool]): Checks if area limit parameter has a proper value.
				  * lsc (Callable[[int], bool]): Checks if local seeding changes parameter has a proper value.
				  * gsc (Callable[[int], bool]): Checks if global seeding changes parameter has a proper value.
				  * tr (Callable[[float], bool]): Checks if transfer rate parameter has a proper value.

		See Also:
			 * :func:`NiaPy.algorithms.algorithm.Algorithm.typeParameters`
		"""
		d = Algorithm.typeParameters()
		d.update({
			'lt': lambda x: isinstance(x, int) and x > 0,
			'al': lambda x: isinstance(x, int) and x > 0,
			'lsc': lambda x: isinstance(x, int) and x > 0,
			'gsc': lambda x: isinstance(x, int) and x > 0,
			'tr': lambda x: isinstance(x, float) and 0 <= x <= 1,
		})
		return d

	def setParameters(self, NP=10, lt=3, al=10, lsc=1, gsc=1, tr=0.3, **ukwargs):
		r"""Set the parameters of the algorithm.

		Args:
			 NP (Optional[int]): Population size.
			 lt (Optional[int]): Life time parameter.
			 al (Optional[int]): Area limit parameter.
			 lsc (Optional[int]): Local seeding changes parameter.
			 gsc (Optional[int]): Global seeding changes parameter.
			 tr (Optional[float]): Transfer rate parameter.
			 ukwargs (Dict[str, Any]): Additional arguments.

		See Also:
			 * :func:`NiaPy.algorithms.Algorithm.setParameters`
		"""
		Algorithm.setParameters(self, NP=NP, **ukwargs)
		self.lt, self.al, self.lsc, self.gsc, self.tr = lt, al, lsc, gsc, tr

	def getParameters(self):
		r"""Get parameters values of the algorithm.

		Returns:
			Dict[str, Any]: TODO.
		"""
		d = Algorithm.getParameters(self)
		d.update({
			'lt': self.lt,
			'al': self.al,
			'lsc': self.lsc,
			'gsc': self.gsc,
			'tr': self.tr
		})
		return d

	def localSeeding(self, task, trees):
		r"""Local optimum search stage.

		Args:
			 task (Task): Optimization task.
			 trees (numpy.ndarray): Zero age trees for local seeding.

		Returns:
			 numpy.ndarray: Resulting zero age trees.
		"""
		n = trees.shape[0]
		deltas = self.uniform(-self.dx, self.dx, (n, self.lsc))
		deltas = append(deltas, zeros((n, task.D - self.lsc)), axis=1)
		perms = self.rand([deltas.shape[0], deltas.shape[1]]).argsort(1)
		deltas = deltas[arange(deltas.shape[0])[:, None], perms]
		trees += deltas
		trees = apply_along_axis(limit_repair, 1, trees, task.Lower, task.Upper)
		return trees

	def globalSeeding(self, task, candidates, size):
		r"""Global optimum search stage that should prevent getting stuck in a local optimum.

		Args:
			 task (Task): Optimization task.
			 candidates (numpy.ndarray): Candidate population for global seeding.
			 size (int): Number of trees to produce.

		Returns:
			 numpy.ndarray: Resulting trees.
		"""
		seeds = candidates[self.randint(len(candidates), D=size)]
		deltas = self.uniform(task.benchmark.Lower, task.benchmark.Upper, (size, self.gsc))
		deltas = append(deltas, zeros((size, task.D - self.gsc)), axis=1)
		perms = self.rand([deltas.shape[0], deltas.shape[1]]).argsort(1)
		deltas = deltas[arange(deltas.shape[0])[:, None], perms]
		deltas = deltas.flatten()
		seeds = seeds.flatten()
		seeds[deltas != 0] = deltas[deltas != 0]

		return seeds.reshape(size, task.D)

	def removeLifeTimeExceeded(self, trees, candidates, age):
		r"""Remove dead trees.

		Args:
			 trees (numpy.ndarray): Population to test.
			 candidates (numpy.ndarray): Candidate population array to be updated.
			 age (numpy.ndarray[int32]): Age of trees.

		Returns:
			 Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray[int32]]:
				  1. Alive trees.
				  2. New candidate population.
				  3. Age of trees.
		"""
		lifeTimeExceeded = where(age > self.lt)
		candidates = trees[lifeTimeExceeded]
		trees = delete(trees, lifeTimeExceeded, axis=0)
		age = delete(age, lifeTimeExceeded, axis=0)
		return trees, candidates, age

	def survivalOfTheFittest(self, task, trees, candidates, age):
		r"""Evaluate and filter current population.

		Args:
			 task (Task): Optimization task.
			 trees (numpy.ndarray): Population to evaluate.
			 candidates (numpy.ndarray): Candidate population array to be updated.
			 age (numpy.ndarray[int32]): Age of trees.

		Returns:
			 Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray[float], numpy.ndarray[int32]]:
				  1. Trees sorted by fitness value.
				  2. Updated candidate population.
				  3. Population fitness values.
				  4. Age of trees
		"""
		evaluations = apply_along_axis(task.eval, 1, trees)
		ei = evaluations.argsort()
		candidates = append(candidates, trees[ei[self.al:]], axis=0)
		trees = trees[ei[:self.al]]
		age = age[ei[:self.al]]
		evaluations = evaluations[ei[:self.al]]
		return trees, candidates, evaluations, age

	def initPopulation(self, task):
		r"""Initialize the starting population.

		Args:
			 task (Task): Optimization task

		Returns:
			 Tuple[numpy.ndarray, numpy.ndarray[float], Dict[str, Any]]:
				  1. New population.
				  2. New population fitness/function values.
				  3. Additional arguments:
						* age (numpy.ndarray[int32]): Age of trees.

		See Also:
			 * :func:`NiaPy.algorithms.Algorithm.initPopulation`
		"""
		Trees, Evaluations, _ = Algorithm.initPopulation(self, task)
		age = zeros(self.NP, dtype=int32)
		self.dx = absolute(task.benchmark.Upper) / 5
		return Trees, Evaluations, {'age': age}

	def runIteration(self, task, Trees, Evaluations, xb, fxb, age, **dparams):
		r"""Core function of Forest Optimization Algorithm.

		Args:
			 task (Task): Optimization task.
			 Trees (numpy.ndarray): Current population.
			 Evaluations (numpy.ndarray[float]): Current population function/fitness values.
			 xb (numpy.ndarray): Global best individual.
			 fxb (float): Global best individual fitness/function value.
			 age (numpy.ndarray[int32]): Age of trees.
			 **dparams (Dict[str, Any]): Additional arguments.

		Returns:
			 Tuple[numpy.ndarray, numpy.ndarray[float], Dict[str, Any]]:
				  1. New population.
				  2. New population fitness/function values.
				  3. Additional arguments:
						* age (numpy.ndarray[int32]): Age of trees.
		"""
		candidatePopulation = ndarray((0, task.D + 1))
		zeroAgeTrees = Trees[age == 0]
		localSeeds = self.localSeeding(task, zeroAgeTrees)
		age += 1
		Trees, candidatePopulation, age = self.removeLifeTimeExceeded(Trees, candidatePopulation, age)
		Trees = append(Trees, localSeeds, axis=0)
		age = append(age, zeros(len(localSeeds), dtype=int32))
		Trees, candidatePopulation, Evaluations, age = self.survivalOfTheFittest(task, Trees, candidatePopulation, age)
		gsn = int(self.tr * len(candidatePopulation))
		if gsn > 0:
			globalSeeds = self.globalSeeding(task, candidatePopulation, gsn)
			Trees = append(Trees, globalSeeds, axis=0)
			age = append(age, zeros(len(globalSeeds), dtype=int32))
			gste = apply_along_axis(task.eval, 1, globalSeeds)
			Evaluations = append(Evaluations, gste)
		ib = argmin(Evaluations)
		age[ib] = 0
		if Evaluations[ib] < fxb: xb, fxb = Trees[ib].copy(), Evaluations[ib]
		return Trees, Evaluations, xb, fxb, {'age': age}
