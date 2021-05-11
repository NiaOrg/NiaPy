# encoding=utf8
import logging

from niapy.algorithms.algorithm import Individual
from niapy.algorithms.basic.de import DifferentialEvolution, cross_best1, cross_rand1, cross_curr2best1, cross_best2, \
    cross_curr2rand1, multi_mutations  # , DynNpDifferentialEvolution, proportional
from niapy.util import objects_to_array

logging.basicConfig()
logger = logging.getLogger('niapy.algorithms.modified')
logger.setLevel('INFO')

__all__ = [
    'SolutionJDE',
    'SelfAdaptiveDifferentialEvolution',
    'MultiStrategySelfAdaptiveDifferentialEvolution',
    # 'DynNpSelfAdaptiveDifferentialEvolutionAlgorithm',
    # 'AgingSelfAdaptiveDifferentialEvolution',
    # 'DynNpMultiStrategySelfAdaptiveDifferentialEvolution'
]


class SolutionJDE(Individual):
    r"""Individual for jDE algorithm.

    Attributes:
        differential_weight (float): Scale factor.
        crossover_probability (float): Crossover probability.

    See Also:
        :class:`niapy.algorithms.Individual`

    """

    def __init__(self, differential_weight=2, crossover_probability=0.5, **kwargs):
        r"""Initialize SolutionJDE.

        Attributes:
            differential_weight (float): Scale factor.
            crossover_probability (float): Crossover probability.

        See Also:
            :func:`niapy.algorithm.Individual.__init__`

        """
        super().__init__(**kwargs)
        self.differential_weight = differential_weight
        self.crossover_probability = crossover_probability


class SelfAdaptiveDifferentialEvolution(DifferentialEvolution):
    r"""Implementation of Self-adaptive differential evolution algorithm.

    Algorithm:
        Self-adaptive differential evolution algorithm

    Date:
        2018

    Author:
        Uros Mlakar and Klemen Berkovič

    License:
        MIT

    Reference paper:
        Brest, J., Greiner, S., Boskovic, B., Mernik, M., Zumer, V. Self-adapting control parameters in differential evolution: A comparative study on numerical benchmark problems. IEEE transactions on evolutionary computation, 10(6), 646-657, 2006.

    Attributes:
        Name (List[str]): List of strings representing algorithm name
        f_lower (float): Scaling factor lower limit.
        f_upper (float): Scaling factor upper limit.
        tao1 (float): Change rate for differential_weight parameter update.
        tao2 (float): Change rate for crossover_probability parameter update.

    See Also:
        * :class:`niapy.algorithms.basic.DifferentialEvolution`

    """

    Name = ['SelfAdaptiveDifferentialEvolution', 'jDE']

    @staticmethod
    def info():
        r"""Get algorithm information.

        Returns:
            str: Algorithm information.

        See Also:
            * :func:`niapy.algorithms.Algorithm.info`

        """
        return r"""Brest, J., Greiner, S., Boskovic, B., Mernik, M., Zumer, V. Self-adapting control parameters in differential evolution: A comparative study on numerical benchmark problems. IEEE transactions on evolutionary computation, 10(6), 646-657, 2006."""

    def __init__(self, f_lower=0.0, f_upper=1.0, tao1=0.4, tao2=0.2, *args, **kwargs):
        """Initialize SelfAdaptiveDifferentialEvolution.

        Args:
            f_lower (Optional[float]): Scaling factor lower limit.
            f_upper (Optional[float]): Scaling factor upper limit.
            tao1 (Optional[float]): Change rate for differential_weight parameter update.
            tao2 (Optional[float]): Change rate for crossover_probability parameter update.

        See Also:
            * :func:`niapy.algorithms.basic.DifferentialEvolution.__init__`

        """
        super().__init__(individual_type=kwargs.pop('individual_type', SolutionJDE), *args, **kwargs)
        self.f_lower = f_lower
        self.f_upper = f_upper
        self.tao1 = tao1
        self.tao2 = tao2

    def set_parameters(self, f_lower=0.0, f_upper=1.0, tao1=0.4, tao2=0.2, **kwargs):
        r"""Set the parameters of an algorithm.

        Args:
            f_lower (Optional[float]): Scaling factor lower limit.
            f_upper (Optional[float]): Scaling factor upper limit.
            tao1 (Optional[float]): Change rate for differential_weight parameter update.
            tao2 (Optional[float]): Change rate for crossover_probability parameter update.

        See Also:
            * :func:`niapy.algorithms.basic.DifferentialEvolution.set_parameters`

        """
        super().set_parameters(individual_type=kwargs.pop('individual_type', SolutionJDE), **kwargs)
        self.f_lower = f_lower
        self.f_upper = f_upper
        self.tao1 = tao1
        self.tao2 = tao2

    def get_parameters(self):
        r"""Get algorithm parameters.

        Returns:
            Dict[str, Any]: Algorithm parameters.

        """
        d = DifferentialEvolution.get_parameters(self)
        d.update({
            'f_lower': self.f_lower,
            'f_upper': self.f_upper,
            'tao1': self.tao1,
            'tao2': self.tao2
        })
        return d

    def adaptive_gen(self, x):
        r"""Adaptive update scale factor in crossover probability.

        Args:
            x (IndividualJDE): Individual to apply function on.

        Returns:
            Individual: New individual with new parameters

        """
        f = self.f_lower + self.random() * (self.f_upper - self.f_lower) if self.random() < self.tao1 else x.differential_weight
        cr = self.random() if self.random() < self.tao2 else x.crossover_probability
        return self.individual_type(x=x.x, differential_weight=f, crossover_probability=cr, e=False)

    def evolve(self, pop, xb, task, **_kwargs):
        r"""Evolve current population.

        Args:
            pop (numpy.ndarray[Individual]): Current population.
            xb (Individual): Global best individual.
            task (Task): Optimization task.

        Returns:
            numpy.ndarray: New population.

        """
        new_pop = objects_to_array([self.adaptive_gen(e) for e in pop])
        for i, e in enumerate(new_pop):
            new_pop[i].x = self.strategy(new_pop, i, e.differential_weight, e.crossover_probability, rng=self.rng, x_b=xb)
        for e in new_pop:
            e.evaluate(task, rng=self.random)
        return new_pop


class MultiStrategySelfAdaptiveDifferentialEvolution(SelfAdaptiveDifferentialEvolution):
    r"""Implementation of self-adaptive differential evolution algorithm with multiple mutation strategies.

    Algorithm:
        Self-adaptive differential evolution algorithm with multiple mutation strategies

    Date:
        2018

    Author:
        Klemen Berkovič

    License:
        MIT

    Attributes:
        Name (List[str]): List of strings representing algorithm name

    See Also:
        * :class:`niapy.algorithms.modified.SelfAdaptiveDifferentialEvolution`

    """

    Name = ['MultiStrategySelfAdaptiveDifferentialEvolution', 'MsjDE']

    def __init__(self, strategies=(cross_curr2rand1, cross_curr2best1, cross_rand1, cross_best1, cross_best2), *args,
                 **kwargs):
        """Initialize MultiStrategySelfAdaptiveDifferentialEvolution.

        Args:
            strategies (Optional[Iterable[Callable]]): Mutations strategies to use in algorithm.

        See Also:
            * :func:`niapy.algorithms.modified.SelfAdaptiveDifferentialEvolution.__init__`

        """
        super().__init__(strategy=kwargs.pop('strategy', multi_mutations), *args, **kwargs)
        self.strategies = strategies

    def set_parameters(self, strategies=(cross_curr2rand1, cross_curr2best1, cross_rand1, cross_best1, cross_best2),
                       **kwargs):
        r"""Set core parameters of MultiStrategySelfAdaptiveDifferentialEvolution algorithm.

        Args:
            strategies (Optional[Iterable[Callable]]): Mutations strategies to use in algorithm.

        See Also:
            * :func:`niapy.algorithms.modified.SelfAdaptiveDifferentialEvolution.set_parameters`

        """
        super().set_parameters(strategy=kwargs.pop('strategy', multi_mutations), **kwargs)
        self.strategies = strategies

    def evolve(self, pop, xb, task, **kwargs):
        r"""Evolve population with the help multiple mutation strategies.

        Args:
            pop (numpy.ndarray[Individual]): Current population.
            xb (Individual): Current best individual.
            task (Task): Optimization task.
            **kwargs (Dict[str, Any]): Additional arguments.

        Returns:
            numpy.ndarray[Individual]: New population of individuals.

        """
        return objects_to_array(
            [self.strategy(pop, i, xb, self.differential_weight, self.crossover_probability, self.rng, task, self.individual_type, self.strategies) for i in
             range(len(pop))])


# # TODO: Incomplete or not working implementations, will comment out for now.
# class AgingIndividualJDE(SolutionJDE):
#     r"""Individual with age.
#
#     Attributes:
#         age (int): Age of individual.
#
#     See Also:
#         * :func:`niapy.algorithms.modified.SolutionJDE`
#
#     """
#
#     def __init__(self, **kwargs):
#         r"""Initialize aging individual for jDE algorithm.
#
#         Args:
#             **kwargs (Dict[str, Any]): Additional arguments.
#
#         See Also:
#             * :func:`niapy.algorithms.modified.SolutionJDE.__init__`
#
#         """
#         super().__init__(**kwargs)
#         self.age = 0


# class AgingSelfAdaptiveDifferentialEvolution(SelfAdaptiveDifferentialEvolution):
#     r"""Implementation of Dynamic population size with aging self-adaptive differential evolution algorithm.
#
#     Algorithm:
#         Dynamic population size with aging self-adaptive self adaptive differential evolution algorithm
#
#     Date:
#         2018
#
#     Author:
#         Jan Popič and Klemen Berkovič
#
#     License:
#         MIT
#
#     Reference URL:
#         https://link.springer.com/article/10.1007/s10489-007-0091-x
#
#     Reference paper:
#         Brest, Janez, and Mirjam Sepesy Maučec. Population size reduction for the differential evolution algorithm. Applied Intelligence 29.3 (2008): 228-247.
#
#     Attributes:
#         Name (List[str]): List of strings representing algorithm name.
#
#     """
#
#     Name = ['AgingSelfAdaptiveDifferentialEvolution', 'ANpjDE']
#
#     @staticmethod
#     def info():
#         r"""Get basic information about the algorithm.
#
#         Returns:
#             str: Basic information.
#
#         See Also:
#             * :func:`niapy.algorithms.Algorithm.info`
#
#         """
#         return r"""Brest, Janez, and Mirjam Sepesy Maučec. Population size reduction for the differential evolution algorithm. Applied Intelligence 29.3 (2008): 228-247."""
#
#     def set_parameters(self, LT_min=1, LT_max=7, age=proportional, **kwargs):
#         r"""Set core parameters of AgingSelfAdaptiveDifferentialEvolution algorithm.
#
#         Args:
#             LT_min (Optional[int]): Minimum age.
#             LT_max (Optional[int]): Maximum age.
#             age (Optional[Callable[[], int]]): Function for calculating age of individual.
#             **kwargs (Dict[str, Any]): Additional arguments.
#
#         See Also:
#             * :func:`SelfAdaptiveDifferentialEvolution.set_parameters`
#
#         """
#         SelfAdaptiveDifferentialEvolution.set_parameters(self, **kwargs)
#         self.LT_min, self.LT_max, self.age = LT_min, LT_max, age
#         self.mu = abs(self.LT_max - self.LT_min) / 2


# class DynNpSelfAdaptiveDifferentialEvolutionAlgorithm(SelfAdaptiveDifferentialEvolution, DynNpDifferentialEvolution):
#     r"""Implementation of Dynamic population size self-adaptive differential evolution algorithm.
#
#     Algorithm:
#         Dynamic population size self-adaptive differential evolution algorithm
#
#     Date:
#         2018
#
#     Author:
#         Jan Popič and Klemen Berkovič
#
#     License:
#         MIT
#
#     Reference URL:
#         https://link.springer.com/article/10.1007/s10489-007-0091-x
#
#     Reference paper:
#         Brest, Janez, and Mirjam Sepesy Maučec. Population size reduction for the differential evolution algorithm. Applied Intelligence 29.3 (2008): 228-247.
#
#     Attributes:
#         Name (List[str]): List of strings representing algorithm name.
#
#     See Also:
#         * :class:`niapy.algorithms.modified.SelfAdaptiveDifferentialEvolution`
#
#     """
#     Name = ['DynNpSelfAdaptiveDifferentialEvolutionAlgorithm', 'dynNPjDE']
#
#     @staticmethod
#     def info():
#         r"""Get basic information about the algorithm.
#
#         Returns:
#             str: Basic information.
#
#         See Also:
#             * :func:`niapy.algorithms.Algorithm.info`
#         """
#         return r"""Brest, Janez, and Mirjam Sepesy Maučec. Population size reduction for the differential evolution algorithm. Applied Intelligence 29.3 (2008): 228-247."""
#
#     def set_parameters(self, rp=0, p_max=10, **kwargs):
#         r"""Set the parameters of an algorithm.
#
#         Arguments:
#             rp (Optional[int]): Small non-negative number which is added to value of gen (if it's not divisible).
#             p_max (Optional[int]): Number of population reductions.
#
#         See Also:
#             * :func:`niapy.algorithms.modified.SelfAdaptiveDifferentialEvolution.set_parameters`
#
#         """
#         DynNpDifferentialEvolution.set_parameters(self, rp=rp, p_max=p_max, **kwargs)
#         SelfAdaptiveDifferentialEvolution.set_parameters(self, **kwargs)
#
#     def post_selection(self, pop, task, **kwargs):
#         r"""Post selection operator.
#
#         Args:
#             pop (numpy.ndarray[Individual]): Current population.
#             task (Task): Optimization task.
#
#         Returns:
#             numpy.ndarray[Individual]: New population.
#
#         """
#         return DynNpDifferentialEvolution.post_selection(self, pop, task, **kwargs)


# class DynNpMultiStrategySelfAdaptiveDifferentialEvolution(MultiStrategySelfAdaptiveDifferentialEvolution,
#                                                           DynNpSelfAdaptiveDifferentialEvolutionAlgorithm):
#     r"""Implementation of Dynamic population size self-adaptive differential evolution algorithm with multiple mutation strategies.
#
#     Algorithm:
#         Dynamic population size self-adaptive differential evolution algorithm with multiple mutation strategies
#
#     Date:
#         2018
#
#     Author:
#         Klemen Berkovič
#
#     License:
#         MIT
#
#     Attributes:
#         Name (List[str]): List of strings representing algorithm name.
#
#     See Also:
#         * :class:`niapy.algorithms.modified.MultiStrategySelfAdaptiveDifferentialEvolution`
#         * :class:`niapy.algorithms.modified.DynNpSelfAdaptiveDifferentialEvolutionAlgorithm`
#
#     """
#
#     Name = ['DynNpMultiStrategySelfAdaptiveDifferentialEvolution', 'dynNpMsjDE']
#
#     def set_parameters(self, p_max=10, rp=5, **kwargs):
#         r"""Set core parameters for algorithm instance.
#
#         Args:
#             p_max (Optional[int]):
#             rp (Optional[int]):
#             **kwargs (Dict[str, Any]):
#
#         See Also:
#             * :func:`niapy.algorithms.modified.MultiStrategySelfAdaptiveDifferentialEvolution.set_parameters`
#
#         """
#         MultiStrategySelfAdaptiveDifferentialEvolution.set_parameters(self, **kwargs)
#         self.p_max, self.rp = p_max, rp
#
#     def post_selection(self, pop, task, **kwargs):
#         return DynNpSelfAdaptiveDifferentialEvolutionAlgorithm.post_selection(self, pop, task)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
