# encoding=utf8

import logging

from numpy import argsort

from NiaPy.algorithms.algorithm import Individual
from NiaPy.algorithms.basic.de import MultiStrategyDifferentialEvolution, DynNpDifferentialEvolution, DifferentialEvolution
from NiaPy.algorithms.other.mts import MTS_LS1, MTS_LS1v1, MTS_LS2, MTS_LS3, MTS_LS3v1, MultipleTrajectorySearch

logging.basicConfig()
logger = logging.getLogger('NiaPy.algorithms.modified')
logger.setLevel('INFO')

__all__ = ['DifferentialEvolutionMTS', 'DifferentialEvolutionMTSv1', 'DynNpDifferentialEvolutionMTS', 'DynNpDifferentialEvolutionMTSv1', 'MultiStrategyDifferentialEvolutionMTS', 'MultiStrategyDifferentialEvolutionMTSv1', 'DynNpMultiStrategyDifferentialEvolutionMTS', 'DynNpMultiStrategyDifferentialEvolutionMTSv1']

class MtsIndividual(Individual):
	r"""Individual for MTS local searches.

	Attributes:
		SR (numpy.ndarray): Search range.
		grade (int): Grade of individual.
		enable (bool): If enabled.
		improved (bool): If improved.

	See Also:
		:class:`NiaPy.algorithms.algorithm.Individual`
	"""
	def __init__(self, SR=None, grade=0, enable=True, improved=False, task=None, **kwargs):
		r"""Initialize the individual.

		Args:
			SR (numpy.ndarray): Search range.
			grade (Optional[int]): Grade of individual.
			enable (Optional[bool]): If enabled individual.
			improved (Optional[bool]): If individual improved.
			**kwargs (Dict[str, Any]): Additional arguments.

		See Also:
			:func:`NiaPy.algorithms.algorithm.Individual.__init__`
		"""
		Individual.__init__(self, task=task, **kwargs)
		self.grade, self.enable, self.improved = grade, enable, improved
		if SR is None and task is not None: self.SR = task.bRange / 4
		else: self.SR = SR

class DifferentialEvolutionMTS(DifferentialEvolution, MultipleTrajectorySearch):
	r"""Implementation of Differential Evolution with MTS local searches.

	Algorithm:
		Differential Evolution withm MTS local searches

	Date:
		2018

	Author:
		Klemen Berkovič

	License:
		MIT

	Attributes:
		name (List[str]): List of strings representing algorithm names.
		LSs (Iterable[Callable[[numpy.ndarray, float, numpy.ndarray, float, bool, numpy.ndarray, Task, Dict[str, Any]], Tuple[numpy.ndarray, float, numpy.ndarray, float, bool, int, numpy.ndarray]]]): Local searches to use.
		BONUS1 (int): Bonus for improving global best solution.
		BONUS2 (int): Bonus for improving solution.
		NoLsTests (int): Number of test runs on local search algorithms.
		NoLs (int): Number of local search algorithm runs.
		NoEnabled (int): Number of best solution for testing.

	See Also:
		* :class:`NiaPy.algorithms.basic.de.DifferentialEvolution`
		* :class:`NiaPy.algorithms.other.mts.MultipleTrajectorySearch`
	"""
	name = ['DifferentialEvolutionMTS', 'DEMTS']

	@staticmethod
	def parameter_types():
		r"""Get dictionary with functions for checking values of parameters.

		Returns:
			Dict[str, Callable]:
				* NoLsTests (Callable[[int], bool]): TODO
				* NoLs (Callable[[int], bool]): TODO
				* NoEnabled (Callable[[int], bool]): TODO

		See Also:
			:func:`NiaPy.algorithms.basic.de.DifferentialEvolution.parameter_types`
		"""
		d = DifferentialEvolution.parameter_types()
		d.update({
			'NoLsTests': lambda x: isinstance(x, int) and x >= 0,
			'NoLs': lambda x: isinstance(x, int) and x >= 0,
			'NoEnabled': lambda x: isinstance(x, int) and x > 0
		})
		return d

	def set_parameters(self, NoLsTests=1, NoLs=2, NoEnabled=2, BONUS1=10, BONUS2=2, LSs=(MTS_LS1, MTS_LS2, MTS_LS3), **ukwargs):
		r"""Set the algorithm parameters.

		Arguments:
			SR (numpy.ndarray): Search range.

		See Also:
			:func:`NiaPy.algorithms.basic.de.DifferentialEvolution.set_parameters`
		"""
		DifferentialEvolution.set_parameters(self, itype=ukwargs.pop('itype', MtsIndividual), **ukwargs)
		self.LSs, self.NoLsTests, self.NoLs, self.NoEnabled = LSs, NoLsTests, NoLs, NoEnabled
		self.BONUS1, self.BONUS2 = BONUS1, BONUS2
		if ukwargs: logger.info('Unused arguments: %s' % (ukwargs))

	def postSelection(self, X, task, xb, **kwargs):
		r"""Post selection operator.

		Args:
			X (numpy.ndarray[Individual]): Current populaiton.
			task (Task): Optimization task.
			xb (Individual): Global best individual.
			**kwargs (Dict[str, Any]): Additional arguments.

		Returns:
			numpy.ndarray[Individual]: New population.
		"""
		for x in X:
			if not x.enable: continue
			x.enable, x.grades = False, 0
			x.x, x.f, xb.x, xb.f, k = self.GradingRun(x.x, x.f, xb.x, xb.f, x.improved, x.SR, task)
			x.x, x.f, xb.x, xb.f, x.improved, x.SR, x.grades = self.LsRun(k, x.x, x.f, xb.x, xb.f, x.improved, x.SR, xb.grade, task)
		for i in X[argsort([x.grade for x in X])[:self.NoEnabled]]: i.enable = True
		return X

class DifferentialEvolutionMTSv1(DifferentialEvolutionMTS):
	r"""Implementation of Differential Evolution withm MTSv1 local searches.

	Algorithm:
		Differential Evolution withm MTSv1 local searches

	Date:
		2018

	Author:
		Klemen Berkovič

	License:
		MIT

	Attributes:
		name (List[str]): List of strings representing algorithm name.

	See Also:
		:class:`NiaPy.algorithms.modified.DifferentialEvolutionMTS`
	"""
	name = ['DifferentialEvolutionMTSv1', 'DEMTSv1']

	def set_parameters(self, **ukwargs):
		r"""Set core parameters of DifferentialEvolutionMTSv1 algorithm.

		Args:
			**ukwargs (Dict[str, Any]): Additional arguments.

		See Also:
			:func:`NiaPy.algorithms.modified.DifferentialEvolutionMTS.set_parameters`
		"""
		DifferentialEvolutionMTS.set_parameters(self, LSs=(MTS_LS1v1, MTS_LS2, MTS_LS3v1), **ukwargs)

class DynNpDifferentialEvolutionMTS(DifferentialEvolutionMTS, DynNpDifferentialEvolution):
	r"""Implementation of Differential Evolution withm MTS local searches dynamic and population size.

	Algorithm:
		Differential Evolution withm MTS local searches and dynamic population size

	Date:
		2018

	Author:
		Klemen Berkovič

	License:
		MIT

	Attributes:
		name (List[str]): List of strings representing algorithm name

	See Also:
		* :class:`NiaPy.algorithms.modified.DifferentialEvolutionMTS`
		* :class:`NiaPy.algorithms.basic.de.DynNpDifferentialEvolution`
	"""
	name = ['DynNpDifferentialEvolutionMTS', 'dynNpDEMTS']

	def set_parameters(self, pmax=10, rp=3, **ukwargs):
		r"""Set core parameters or DynNpDifferentialEvolutionMTS algorithm.

		Args:
			pmax (Optional[int]):
			rp (Optional[float]):
			**ukwargs (Dict[str, Any]): Additional arguments.

		See Also:
			* :func:`NiaPy.algorithms.modified.hde.DifferentialEvolutionMTS.set_parameters`
			* :func`NiaPy.algorithms.basic.de.DynNpDifferentialEvolution.set_parameters`
		"""
		DynNpDifferentialEvolution.set_parameters(self, pmax=pmax, rp=rp, **ukwargs)
		DifferentialEvolutionMTS.set_parameters(self, **ukwargs)

	def postSelection(self, X, task, xb, **kwargs):
		nX = DynNpDifferentialEvolution.postSelection(self, X, task)
		nX = DifferentialEvolutionMTS.postSelection(self, nX, task, xb)
		return nX

class DynNpDifferentialEvolutionMTSv1(DynNpDifferentialEvolutionMTS):
	r"""Implementation of Differential Evolution withm MTSv1 local searches and dynamic population size.

	Algorithm:
		Differential Evolution with MTSv1 local searches and dynamic population size

	Date:
		2018

	Author:
		Klemen Berkovič

	License:
		MIT

	Attributes:
		name (List[str]): List of strings representing algorithm name.

	See Also:
		:class:`NiaPy.algorithms.modified.hde.DifferentialEvolutionMTS`
	"""
	name = ['DynNpDifferentialEvolutionMTSv1', 'dynNpDEMTSv1']

	def set_parameters(self, **ukwargs):
		r"""Set core arguments of DynNpDifferentialEvolutionMTSv1 algorithm.

		Args:
			**ukwargs (Dict[str, Any]): Additional arguments.

		See Also:
			:func:`NiaPy.algorithms.modified.hde.DifferentialEvolutionMTS.set_parameters`
		"""
		DynNpDifferentialEvolutionMTS.set_parameters(self, LSs=(MTS_LS1v1, MTS_LS2, MTS_LS3v1), **ukwargs)

class MultiStrategyDifferentialEvolutionMTS(DifferentialEvolutionMTS, MultiStrategyDifferentialEvolution):
	r"""Implementation of Differential Evolution withm MTS local searches and multiple mutation strategys.

	Algorithm:
		Differential Evolution withm MTS local searches and multiple mutation strategys

	Date:
		2018

	Author:
		Klemen Berkovič

	License:
		MIT

	Attributes:
		name (List[str]): List of strings representing algorithm name.

	See Also:
		* :class:`NiaPy.algorithms.modified.hde.DifferentialEvolutionMTS`
		* :class:`NiaPy.algorithms.basic.de.MultiStrategyDifferentialEvolution`
	"""
	name = ['MultiStrategyDifferentialEvolutionMTS', 'MSDEMTS']

	def set_parameters(self, **ukwargs):
		r"""TODO.

		Args:
			**ukwargs (Dict[str, Any]): Additional arguments.

		See Also:
			* :func:`NiaPy.algorithms.modified.DifferentialEvolutionMTS.set_parameters`
			* :func:`NiaPy.algorithms.basic.MultiStrategyDifferentialEvolution.set_parameters`
		"""
		DifferentialEvolutionMTS.set_parameters(self, **ukwargs)
		MultiStrategyDifferentialEvolution.set_parameters(self, itype=ukwargs.pop('itype', MtsIndividual), **ukwargs)

	def evolve(self, pop, xb, task, **kwargs):
		r"""Evolve population.

		Args:
			pop (numpy.ndarray[Individual]): Current population of individuals.
			xb (Individual): Global best individual.
			task (Task): Optimization task.
			**kwargs (Dict[str, Any]): Additional arguments.

		Returns:
			numpy.ndarray[Individual]: Evolved population.
		"""
		return MultiStrategyDifferentialEvolution.evolve(self, pop, xb, task, **kwargs)

class MultiStrategyDifferentialEvolutionMTSv1(MultiStrategyDifferentialEvolutionMTS):
	r"""Implementation of Differential Evolution with MTSv1 local searches and multiple mutation strategys.

	Algorithm:
		Differential Evolution withm MTSv1 local searches and multiple mutation strategys

	Date:
		2018

	Author:
		Klemen Berkovič

	License:
		MIT

	Attributes:
		name (List[str]): List of stings representing algorithm name.

	See Also:
		* :class:`NiaPy.algorithms.modified.MultiStrategyDifferentialEvolutionMTS`
	"""
	name = ['MultiStrategyDifferentialEvolutionMTSv1', 'MSDEMTSv1']

	def set_parameters(self, **ukwargs):
		r"""Set core parameters of MultiStrategyDifferentialEvolutionMTSv1 algorithm.

		Args:
			**ukwargs (Dict[str, Any]): Additional arguments.

		See Also:
			* :func:`NiaPy.algorithms.modified.MultiStrategyDifferentialEvolutionMTS.set_parameters`
		"""
		MultiStrategyDifferentialEvolutionMTS.set_parameters(self, LSs=(MTS_LS1v1, MTS_LS2, MTS_LS3v1), **ukwargs)

class DynNpMultiStrategyDifferentialEvolutionMTS(MultiStrategyDifferentialEvolutionMTS, DynNpDifferentialEvolutionMTS):
	r"""Implementation of Differential Evolution withm MTS local searches, multiple mutation strategys and dynamic population size.

	Algorithm:
		Differential Evolution withm MTS local searches, multiple mutation strategys and dynamic population size

	Date:
		2018

	Author:
		Klemen Berkovič

	License:
		MIT

	Attributes:
		name (List[str]): List of strings representing algorithm name

	See Also:
		* :class:`NiaPy.algorithms.modified.MultiStrategyDifferentialEvolutionMTS`
		* :class:`NiaPy.algorithms.modified.DynNpDifferentialEvolutionMTS`
	"""
	name = ['DynNpMultiStrategyDifferentialEvolutionMTS', 'dynNpMSDEMTS']

	def set_parameters(self, **ukwargs):
		r"""Set core arguments of DynNpMultiStrategyDifferentialEvolutionMTS algorithm.

		Args:
			**ukwargs (Dict[str, Any]): Additional arguments.

		See Also:
			* :func:`NiaPy.algorithms.modified.MultiStrategyDifferentialEvolutionMTS.set_parameters`
			* :func:`NiaPy.algorithms.modified.DynNpDifferentialEvolutionMTS.set_parameters`
		"""
		DynNpDifferentialEvolutionMTS.set_parameters(self, **ukwargs)
		MultiStrategyDifferentialEvolutionMTS.set_parameters(self, **ukwargs)

class DynNpMultiStrategyDifferentialEvolutionMTSv1(DynNpMultiStrategyDifferentialEvolutionMTS):
	r"""Implementation of Differential Evolution withm MTSv1 local searches, multiple mutation strategys and dynamic population size.

	Algorithm:
		Differential Evolution withm MTSv1 local searches, multiple mutation strategys and dynamic population size

	Date:
		2018

	Author:
		Klemen Berkovič

	License:
		MIT

	Attributes:
		name (List[str]): List of strings representing algorithm name.

	See Also:
		* :class:`NiaPy.algorithm.modified.DynNpMultiStrategyDifferentialEvolutionMTS`
	"""
	name = ['DynNpMultiStrategyDifferentialEvolutionMTSv1', 'dynNpMSDEMTSv1']

	def set_parameters(self, **kwargs):
		r"""Set core parameters of DynNpMultiStrategyDifferentialEvolutionMTSv1 algorithm.

		Args:
			**kwargs (Dict[str, Any]): Additional arguments.

		See Also:
			* :func:`NiaPy.algorithm.modified.DynNpMultiStrategyDifferentialEvolutionMTS.set_parameters`
		"""
		DynNpMultiStrategyDifferentialEvolutionMTS.set_parameters(self, LSs=(MTS_LS1v1, MTS_LS2, MTS_LS3v1), **kwargs)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
