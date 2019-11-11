# encoding=utf8
import logging

from numpy import random as rand, concatenate, asarray, argsort

from NiaPy.algorithms.basic.de import DifferentialEvolution

logging.basicConfig()
logger = logging.getLogger('NiaPy.algorithms.modified')
logger.setLevel('INFO')

__all__ = [
	'AdaptiveArchiveDifferentialEvolution',
	'CrossRandCurr2Pbest'
]

def CrossRandCurr2Pbest(pop, ic, x_b, f, cr, p=0.2, arc=None, rnd=rand, *args):
	r"""Mutation strategy with crossover.

	Mutation strategy uses two different random individuals from population to perform mutation.

	Mutation:
		Name: DE/curr2pbest/1

	Args:
		pop (numpy.ndarray): Current population.
		ic (int): Index of current individual.
		x_b (numpy.ndarray): Global best individual.
		f (float): Scale factor.
		cr (float): Crossover probability.
		p (float): Procentage of best individuals to use.
		arc (numpy.ndarray): Achived individuals.
		rnd (mtrand.RandomState): Random generator.
		*args (Dict[str, Any]): Additional argumets.

	Returns:
		numpy.ndarray: New position.
	"""
	# Get random index from current population
	pb = [1.0 / (len(pop) - 1) if i != ic else 0 for i in range(len(pop))] if len(pop) > 1 else None
	r = rnd.choice(len(pop), 1, replace=not len(pop) >= 3, p=pb)
	# Get pbest index
	index, pi = argsort(pop), int(len(pop) * p)
	ppop = pop[index[:pi]]
	pb = [1.0 / len(ppop) for i in range(pi)] if len(ppop) > 1 else None
	rp = rnd.choice(pi, 1, replace=not len(ppop) >= 1, p=pb)
	# Get union population and archive index
	apop = concatenate((pop, arc)) if arc is not None else pop
	pb = [1.0 / (len(apop) - 1) if i != ic else 0 for i in range(len(apop))] if len(apop) > 1 else None
	ra = rnd.choice(len(apop), 1, replace=not len(apop) >= 1, p=pb)
	# Generate new positoin
	j = rnd.randint(len(pop[ic]))
	x = [pop[ic][i] + f * (ppop[rp[0]][i] - pop[ic][i]) + f * (pop[r[0]][i] - apop[ra[0]][i]) if rnd.rand() < cr or i == j else pop[ic][i] for i in range(len(pop[ic]))]
	return asarray(x)

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
		:class:`NiaPy.algorithms.basic.DifferentialEvolution`
	"""
	Name = ['AdaptiveArchiveDifferentialEvolution', 'JADE']

	@staticmethod
	def algorithmInfo():
		r"""Get algorithm information.

		Returns:
			str: Alogrithm information.

		See Also:
			:func:`NiaPy.algorithms.algorithm.Algorithm.algorithmInfo`
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
