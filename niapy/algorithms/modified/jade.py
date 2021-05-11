# encoding=utf8
# TODO Not implemented, commenting out.
# import logging
#
# import numpy as np
#
# from niapy.algorithms.basic.de import DifferentialEvolution
#
# logging.basicConfig()
# logger = logging.getLogger('niapy.algorithms.modified')
# logger.setLevel('INFO')
#
# __all__ = [
#     'AdaptiveArchiveDifferentialEvolution',
#     'cross_curr2p_best'
# ]
#
#
# def cross_curr2p_best(pop, ic, fpop, f, cr, rng, p=0.2, arc=None, **_kwargs):
#     r"""Mutation strategy with crossover.
#
#     Mutation strategy uses two different random individuals from population to perform mutation.
#
#     Mutation:
#         Name: DE/curr2pbest/1
#
#     Args:
#         pop (numpy.ndarray): Current population.
#         ic (int): Index of current individual.
#         fpop (numpy.ndarray): Current population scores.
#         f (float): Scale factor.
#         cr (float): Crossover probability.
#         p (float): Percentage of best individuals to use.
#         arc (numpy.ndarray): Archived individuals.
#         rng (numpy.random.Generator): Random generator.
#
#     Returns:
#         numpy.ndarray: New position.
#     """
#     # Get random index from current population
#     pb = [1.0 / (len(pop) - 1) if i != ic else 0 for i in range(len(pop))] if len(pop) > 1 else None
#     r = rng.choice(len(pop), 1, replace=not len(pop) >= 3, p=pb)
#     # Get pbest index
#     index, pi = np.argsort(fpop), int(len(fpop) * p)
#     p_pop = pop[index[:pi]]
#     pb = [1.0 / len(p_pop) for _ in range(pi)] if len(p_pop) > 1 else None
#     rp = rng.choice(pi, 1, replace=not len(p_pop) >= 1, p=pb)
#     # Get union population and archive index
#     a_pop = np.concatenate((pop, arc)) if arc is not None else pop
#     pb = [1.0 / (len(a_pop) - 1) if i != ic else 0 for i in range(len(a_pop))] if len(a_pop) > 1 else None
#     ra = rng.choice(len(a_pop), 1, replace=not len(a_pop) >= 1, p=pb)
#     # Generate new position
#     j = rng.integers(0, len(pop[ic]))
#     x = [el + f * (p_pop[rp[0]][el_idx] - el) + f * (
#                 pop[r[0]][el_idx] - a_pop[ra[0]][el_idx]) if rng.random() < cr or el_idx == j else el for el_idx, el in
#          enumerate(pop[ic])]
#     return np.vstack(x)
#
#
# class AdaptiveArchiveDifferentialEvolution(DifferentialEvolution):
#     r"""Implementation of Adaptive Differential Evolution With Optional External Archive algorithm.
#
#     Algorithm:
#         Adaptive Differential Evolution With Optional External Archive
#
#     Date:
#         2019
#
#     Author:
#         Klemen Berkovič
#
#     License:
#         MIT
#
#     Reference URL:
#         https://ieeexplore.ieee.org/document/5208221
#
#     Reference paper:
#         Zhang, Jingqiao, and Arthur C. Sanderson. "JADE: adaptive differential evolution with optional external archive." IEEE Transactions on evolutionary computation 13.5 (2009): 945-958.
#
#     Attributes:
#         Name (List[str]): List of strings representing algorithm name.
#
#     See Also:
#         :class:`niapy.algorithms.basic.DifferentialEvolution`
#     """
#     Name = ['AdaptiveArchiveDifferentialEvolution', 'JADE']
#
#     @staticmethod
#     def info():
#         r"""Get algorithm information.
#
#         Returns:
#             str: Alogrithm information.
#
#         See Also:
#             :func:`niapy.algorithms.algorithm.Algorithm.info`
#         """
#         return r"""Zhang, Jingqiao, and Arthur C. Sanderson. "JADE: adaptive differential evolution with optional external archive." IEEE Transactions on evolutionary computation 13.5 (2009): 945-958."""
#
#     def set_parameters(self, **kwargs):
#         DifferentialEvolution.set_parameters(self, **kwargs)
#
#
#     def get_parameters(self):
#         d = DifferentialEvolution.get_parameters(self)
#         return d
#
#     def run_iteration(self, task, population, population_fitness, best_x, best_fitness, **params):
#         return population, population_fitness, best_x, best_fitness, params

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
