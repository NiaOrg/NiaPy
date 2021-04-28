# encoding=utf8
import logging

import numpy as np

from niapy.algorithms.basic.de import DifferentialEvolution

logging.basicConfig()
logger = logging.getLogger('niapy.algorithms.modified')
logger.setLevel('INFO')

__all__ = [
	'AdaptiveArchiveDifferentialEvolution',
	'CrossRandCurr2Pbest'
]

def CrossRandCurr2Pbest(pop, ic, fpop, f, cr, rng, p=0.2, arc=None, *args):
	r"""Mutation strategy with crossover.

	Mutation strategy uses two different random individuals from population to perform mutation.

	Mutation:
		Name: DE/curr2pbest/1

	Args:
		pop (numpy.ndarray): Current population.
		ic (int): Index of current individual.
		fpop (numpy.ndarray): Current population scores.
		f (float): Scale factor.
		cr (float): Crossover probability.
		p (float): Procentage of best individuals to use.
		arc (numpy.ndarray): Achived individuals.
		rng (numpy.random.Generator): Random generator.
		args (Dict[str, Any]): Additional argumets.

	Returns:
		numpy.ndarray: New position.
	"""
	# Get random index from current population
	pb = [1.0 / (len(pop) - 1) if i != ic else 0 for i in range(len(pop))] if len(pop) > 1 else None
	r = rng.choice(len(pop), 1, replace=not len(pop) >= 3, p=pb)
	# Get pbest index
	index, pi = np.argsort(fpop), int(len(fpop) * p)
	ppop = pop[index[:pi]]
	pb = [1.0 / len(ppop) for i in range(pi)] if len(ppop) > 1 else None
	rp = rng.choice(pi, 1, replace=not len(ppop) >= 1, p=pb)
	# Get union population and archive index
	apop = np.concatenate((pop, arc)) if arc is not None else pop
	pb = [1.0 / (len(apop) - 1) if i != ic else 0 for i in range(len(apop))] if len(apop) > 1 else None
	ra = rng.choice(len(apop), 1, replace=not len(apop) >= 1, p=pb)
	# Generate new position
	j = rng.integers(0, len(pop[ic]))
	x = [el + f * (ppop[rp[0]][elidx] - el) + f * (pop[r[0]][elidx] - apop[ra[0]][elidx]) if rng.random() < cr or elidx == j else el for elidx, el in enumerate(pop[ic])]
	return np.vstack(x)

class AdaptiveArchiveDifferentialEvolution(DifferentialEvolution):
	r"""Implementation of Adaptive Differential Evolution With Optional External Archive algorithm.

	Algorithm:
		Adaptive Differential Evolution With Optional External Archive

	Date:
		2019

	Author:
		Klemen Berkovič

	License:
		MIT

	Reference URL:
		https://ieeexplore.ieee.org/document/5208221

	Reference paper:
		Zhang, Jingqiao, and Arthur C. Sanderson. "JADE: adaptive differential evolution with optional external archive." IEEE Transactions on evolutionary computation 13.5 (2009): 945-958.

	Attributes:
		Name (List[str]): List of strings representing algorithm name.

	See Also:
		:class:`niapy.algorithms.basic.DifferentialEvolution`
	"""
	Name = ['AdaptiveArchiveDifferentialEvolution', 'JADE']

	@staticmethod
	def algorithmInfo():
		r"""Get algorithm information.

		Returns:
			str: Alogrithm information.

		See Also:
			:func:`niapy.algorithms.algorithm.Algorithm.algorithmInfo`
		"""
		return r"""Zhang, Jingqiao, and Arthur C. Sanderson. "JADE: adaptive differential evolution with optional external archive." IEEE Transactions on evolutionary computation 13.5 (2009): 945-958."""

	def setParameters(self, **kwargs):
		DifferentialEvolution.setParameters(self, **kwargs)
		# TODO add parameters of the algorithm

	def getParameters(self):
		d = DifferentialEvolution.getParameters(self)
		# TODO add paramters values
		return d

	def runIteration(self, task, pop, fpop, xb, fxb, **dparams):
		# TODO Implement algorithm
		return pop, fpop, xb, fxb, dparams

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3