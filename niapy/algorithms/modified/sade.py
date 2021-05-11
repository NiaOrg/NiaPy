# encoding=utf8
# TODO Not implemented, commenting out
# import logging
#
# from niapy.algorithms.algorithm import Individual
# from niapy.algorithms.basic.de import \
#     DifferentialEvolution  # , cross_best1, CrossRand1, cross_curr2best1, cross_best2, cross_curr2rand1, proportional
#
# logging.basicConfig()
# logger = logging.getLogger('niapy.algorithms.modified')
# logger.setLevel('INFO')
#
# __all__ = [
#     'StrategyAdaptationDifferentialEvolution',
#     'StrategyAdaptationDifferentialEvolutionV1'
# ]
#
#
# class StrategyAdaptationDifferentialEvolution(DifferentialEvolution):
#     r"""Implementation of Differential Evolution Algorithm With Strategy Adaptation algorithm.
#
#     Algorithm:
#         Differential Evolution Algorithm With StrategyAdaptation
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
#         https://ieeexplore.ieee.org/document/1554904
#
#     Reference paper:
#         Qin, A. Kai, and Ponnuthurai N. Suganthan. "Self-adaptive differential evolution algorithm for numerical optimization." 2005 IEEE congress on evolutionary computation. Vol. 2. IEEE, 2005.
#
#     Attributes:
#         Name (List[str]): List of strings representing algorithm name.
#
#     See Also:
#         :class:`niapy.algorithms.basic.DifferentialEvolution`
#     """
#     Name = ['StrategyAdaptationDifferentialEvolution', 'SADE', 'SaDE']
#
#     @staticmethod
#     def info():
#         r"""Get basic information about the algorithm.
#
#         Returns:
#             str: Basic information.
#
#         See Also:
#             :func:`niapy.algorithms.algorithm.Algorithm.info`
#         """
#         return r"""Qin, A. Kai, and Ponnuthurai N. Suganthan. "Self-adaptive differential evolution algorithm for numerical optimization." 2005 IEEE congress on evolutionary computation. Vol. 2. IEEE, 2005."""
#
#     def set_parameters(self, **kwargs):
#         DifferentialEvolution.set_parameters(self, **kwargs)
#
#
#
#     def get_parameters(self):
#         d = DifferentialEvolution.get_parameters(self)
#
#         return d
#
#     def run_iteration(self, task, population, population_fitness, best_x, best_fitness, **params):
#
#         return population, population_fitness, best_x, best_fitness, params
#
#
# class StrategyAdaptationDifferentialEvolutionV1(DifferentialEvolution):
#     r"""Implementation of Differential Evolution Algorithm With Strategy Adaptation algorithm.
#
#     Algorithm:
#         Differential Evolution Algorithm With StrategyAdaptation
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
#         https://ieeexplore.ieee.org/document/4632146
#
#     Reference paper:
#         Qin, A. Kai, Vicky Ling Huang, and Ponnuthurai N. Suganthan. "Differential evolution algorithm with strategy adaptation for global numerical optimization." IEEE transactions on Evolutionary Computation 13.2 (2009): 398-417.
#
#     Attributes:
#         Name (List[str]): List of strings representing algorithm name.
#
#     See Also:
#         :class:`niapy.algorithms.basic.DifferentialEvolution`
#     """
#     Name = ['StrategyAdaptationDifferentialEvolutionV1', 'SADEV1', 'SaDEV1']
#
#     @staticmethod
#     def info():
#         r"""Get basic information about the algorithm.
#
#         Returns:
#             str: Basic information.
#
#         See Also:
#             :func:`niapy.algorithms.algorithm.Algorithm.info`
#         """
#         return r"""Qin, A. Kai, Vicky Ling Huang, and Ponnuthurai N. Suganthan. "Differential evolution algorithm with strategy adaptation for global numerical optimization." IEEE transactions on Evolutionary Computation 13.2 (2009): 398-417."""
#
#     def set_parameters(self, **kwargs):
#         DifferentialEvolution.set_parameters(self, **kwargs)
#
#
#
#     def get_parameters(self):
#         d = DifferentialEvolution.get_parameters(self)
#         return d
#
#     def run_iteration(self, task, population, population_fitness, best_x, best_fitness, **params):
#         return population, population_fitness, best_x, best_fitness, params

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
