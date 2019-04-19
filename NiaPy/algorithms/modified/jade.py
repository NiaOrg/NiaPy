# encoding=utf8
# pylint: disable=mixed-indentation, multiple-statements, logging-not-lazy, attribute-defined-outside-init, line-too-long, arguments-differ, singleton-comparison, bad-continuation, dangerous-default-value, consider-using-enumerate, unused-argument, keyword-arg-before-vararg
import logging

from numpy import random as rand, concatenate, asarray, argsort  # , argmin, argmax, mean, cos

# from NiaPy.algorithms.algorithm import Individual
from NiaPy.algorithms.basic.de import DifferentialEvolution  # , CrossBest1, CrossRand1, CrossCurr2Best1, CrossBest2, CrossCurr2Rand1, proportional

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
		name: DE/curr2pbest/1

	Args:
		pop (numpy.ndarray[Individual]): Current population.
		ic (int): Index of current individual.
		x_b (Individual): Global best individual.
		f (float): Scale factor.
		cr (float): Crossover probability.
		p (float): Procentage of best individuals to use.
		arc (numpy.ndarray[Individual]): Achived individuals.
		rnd (mtrand.RandomState): Random generator.
		*args (Dict[str, Any]): Additional argumets.

	Returns:
		numpy.ndarray: New position.
	"""
	# Get random index from current population
	pb = [1 / (len(pop) - 1) if i != ic else 0 for i in range(len(pop))] if len(pop) > 1 else None
	r = rnd.choice(len(pop), 1, replace=not len(pop) >= 3, p=pb)
	# Get pbest index
	index, pi = argsort(pop), int(len(pop) * p)
	ppop = pop[index[:pi]]
	pb = [1 / (len(ppop) - 1) if i != ic else 0 for i in range(len(ppop))] if len(ppop) > 1 else None
	rp = rnd.choice(len(ppop), 1, replace=not len(ppop) >= 1, p=pb)
	# Get union population and archive index
	apop = concatenate((pop, arc)) if arc is not None else pop
	pb = [1 / (len(apop) - 1) if i != ic else 0 for i in range(len(apop))] if len(apop) > 1 else None
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
		name (List[str]): List of strings representing algorithm name.

	See Also:
		:class:`NiaPy.algorithms.basic.DifferentialEvolution`
	"""
	name = ['AdaptiveArchiveDifferentialEvolution', 'JADE']

	def set_parameters(self, **kwargs):
		pass

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
