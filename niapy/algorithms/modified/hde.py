# encoding=utf8
import logging

import numpy as np

from niapy.algorithms.algorithm import Individual
from niapy.algorithms.basic.de import MultiStrategyDifferentialEvolution, DynNpDifferentialEvolution, DifferentialEvolution
from niapy.algorithms.other.mts import mts_ls1v1, mts_ls2, mts_ls3v1, MultipleTrajectorySearch

logging.basicConfig()
logger = logging.getLogger('niapy.algorithms.modified')
logger.setLevel('INFO')

__all__ = ['DifferentialEvolutionMTS', 'DifferentialEvolutionMTSv1', 'DynNpDifferentialEvolutionMTS',
           'DynNpDifferentialEvolutionMTSv1', 'MultiStrategyDifferentialEvolutionMTS',
           'MultiStrategyDifferentialEvolutionMTSv1', 'DynNpMultiStrategyDifferentialEvolutionMTS',
           'DynNpMultiStrategyDifferentialEvolutionMTSv1']


class MtsIndividual(Individual):
    r"""Individual for MTS local searches.

    Attributes:
        search_range (numpy.ndarray): Search range.
        grade (int): Grade of individual.
        enable (bool): If enabled.
        improved (bool): If improved.

    See Also:
        :class:`niapy.algorithms.algorithm.Individual`

    """

    def __init__(self, search_range=None, grade=0, enable=True, improved=False, task=None, **kwargs):
        r"""Initialize the individual.

        Args:
            search_range (numpy.ndarray): Search range.
            grade (Optional[int]): Grade of individual.
            enable (Optional[bool]): If enabled individual.
            improved (Optional[bool]): If individual improved.

        See Also:
            :func:`niapy.algorithms.algorithm.Individual.__init__`

        """
        super().__init__(task=task, **kwargs)
        self.grade, self.enable, self.improved = grade, enable, improved
        if search_range is None and task is not None:
            self.search_range = task.range / 4
        else:
            self.search_range = search_range


class DifferentialEvolutionMTS(DifferentialEvolution, MultipleTrajectorySearch):
    r"""Implementation of Differential Evolution with MTS local searches.

    Algorithm:
        Differential Evolution with MTS local searches

    Date:
        2018

    Author:
        Klemen Berkovič

    License:
        MIT

    Attributes:
        Name (List[str]): List of strings representing algorithm names.

    See Also:
        * :class:`niapy.algorithms.basic.de.DifferentialEvolution`
        * :class:`niapy.algorithms.other.mts.MultipleTrajectorySearch`

    """

    Name = ['DifferentialEvolutionMTS', 'DEMTS']

    @staticmethod
    def info():
        r"""Get basic information about the algorithm.

        Returns:
            str: Basic information.

        See Also:
            * :func:`niapy.algorithms.Algorithm.info`

        """
        return r"""TODO"""

    def __init__(self, population_size=40, *args, **kwargs):
        """Initialize DifferentialEvolutionMTS."""
        super().__init__(population_size, individual_type=kwargs.pop('individual_type', MtsIndividual), *args, **kwargs)

    def set_parameters(self, **kwargs):
        r"""Set the algorithm parameters.

        See Also:
            :func:`niapy.algorithms.basic.de.DifferentialEvolution.set_parameters`

        """
        MultipleTrajectorySearch.set_parameters(self, **kwargs)
        DifferentialEvolution.set_parameters(self, individual_type=kwargs.pop('individual_type', MtsIndividual),
                                             **kwargs)

    def get_parameters(self):
        """Get algorithm parameters."""
        d = DifferentialEvolution.get_parameters(self)
        d.update(MultipleTrajectorySearch.get_parameters(self))
        return d

    def post_selection(self, population, task, xb, fxb, **kwargs):
        r"""Post selection operator.

        Args:
            population (numpy.ndarray): Current population.
            task (Task): Optimization task.
            xb (numpy.ndarray): Global best individual.
            fxb (float): Global best fitness.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray, float]: New population.

        """
        for x in population:
            if not x.enable:
                continue
            x.enable, x.grades = False, 0
            x.x, x.f, xb, fxb, k = self.grading_run(x.x, x.f, xb, fxb, x.improved, x.search_range, task)
            x.x, x.f, xb, fxb, x.improved, x.search_range, x.grades = self.run_local_search(k, x.x, x.f, xb, fxb, x.improved, x.search_range, 0, task)
        for i in population[np.argsort([x.grade for x in population])[:self.num_enabled:]]:
            i.enable = True
        return population, xb, fxb


class DifferentialEvolutionMTSv1(DifferentialEvolutionMTS):
    r"""Implementation of Differential Evolution with MTSv1 local searches.

    Algorithm:
        Differential Evolution with MTSv1 local searches

    Date:
        2018

    Author:
        Klemen Berkovič

    License:
        MIT

    Attributes:
        Name (List[str]): List of strings representing algorithm name.

    See Also:
        :class:`niapy.algorithms.modified.DifferentialEvolutionMTS`

    """

    Name = ['DifferentialEvolutionMTSv1', 'DEMTSv1']

    @staticmethod
    def info():
        r"""Get basic information about the algorithm.

        Returns:
            str: Basic information.

        See Also:
            * :func:`niapy.algorithms.Algorithm.info`

        """
        return r"""TODO"""

    def __init__(self, *args, **kwargs):
        """Initialize DifferentialEvolutionMTSv1."""
        super().__init__(local_searches=(mts_ls1v1, mts_ls2, mts_ls3v1), *args, **kwargs)

    def set_parameters(self, **kwargs):
        r"""Set core parameters of DifferentialEvolutionMTSv1 algorithm.

        See Also:
            :func:`niapy.algorithms.modified.DifferentialEvolutionMTS.set_parameters`

        """
        super().set_parameters(local_searches=(mts_ls1v1, mts_ls2, mts_ls3v1), **kwargs)


class DynNpDifferentialEvolutionMTS(DifferentialEvolutionMTS, DynNpDifferentialEvolution):
    r"""Implementation of Differential Evolution with MTS local searches dynamic and population size.

    Algorithm:
        Differential Evolution with MTS local searches and dynamic population size

    Date:
        2018

    Author:
        Klemen Berkovič

    License:
        MIT

    Attributes:
        Name (List[str]): List of strings representing algorithm name

    See Also:
        * :class:`niapy.algorithms.modified.DifferentialEvolutionMTS`
        * :class:`niapy.algorithms.basic.de.DynNpDifferentialEvolution`

    """

    Name = ['DynNpDifferentialEvolutionMTS', 'dynNpDEMTS']

    @staticmethod
    def info():
        r"""Get basic information about the algorithm.

        Returns:
            str: Basic information.

        See Also:
            * :func:`niapy.algorithms.Algorithm.info`

        """
        return r"""TODO"""

    def __init__(self, *args, **kwargs):
        """Initialize DynNpDifferentialEvolutionMTS."""
        super().__init__(*args, **kwargs)

    def set_parameters(self, p_max=10, rp=3, **kwargs):
        r"""Set core parameters or DynNpDifferentialEvolutionMTS algorithm.

        Args:
            p_max (Optional[int]):
            rp (Optional[float]):

        See Also:
            * :func:`niapy.algorithms.modified.hde.DifferentialEvolutionMTS.set_parameters`
            * :func`niapy.algorithms.basic.de.DynNpDifferentialEvolution.set_parameters`

        """
        DynNpDifferentialEvolution.set_parameters(self, p_max=p_max, rp=rp, **kwargs)
        DifferentialEvolutionMTS.set_parameters(self, **kwargs)

    def post_selection(self, population, task, xb, fxb, **kwargs):
        new_x, xb, fxb = DynNpDifferentialEvolution.post_selection(self, population, task, xb, fxb)
        new_x, xb, fxb = DifferentialEvolutionMTS.post_selection(self, new_x, task, xb, fxb)
        return new_x, xb, fxb


class DynNpDifferentialEvolutionMTSv1(DynNpDifferentialEvolutionMTS):
    r"""Implementation of Differential Evolution with MTSv1 local searches and dynamic population size.

    Algorithm:
        Differential Evolution with MTSv1 local searches and dynamic population size

    Date:
        2018

    Author:
        Klemen Berkovič

    License:
        MIT

    Attributes:
        Name (List[str]): List of strings representing algorithm name.

    See Also:
        :class:`niapy.algorithms.modified.hde.DifferentialEvolutionMTS`

    """

    Name = ['DynNpDifferentialEvolutionMTSv1', 'dynNpDEMTSv1']

    @staticmethod
    def info():
        r"""Get basic information about the algorithm.

        Returns:
            str: Basic information.

        See Also:
            * :func:`niapy.algorithms.Algorithm.info`

        """
        return r"""TODO"""

    def __init__(self, *args, **kwargs):
        """Initialize DynNpDifferentialEvolutionMTSv1."""
        super().__init__(local_searches=(mts_ls1v1, mts_ls2, mts_ls3v1), *args, **kwargs)

    def set_parameters(self, **kwargs):
        r"""Set core arguments of DynNpDifferentialEvolutionMTSv1 algorithm.

        See Also:
            :func:`niapy.algorithms.modified.hde.DifferentialEvolutionMTS.set_parameters`

        """
        DynNpDifferentialEvolutionMTS.set_parameters(self, local_searches=(mts_ls1v1, mts_ls2, mts_ls3v1), **kwargs)


class MultiStrategyDifferentialEvolutionMTS(DifferentialEvolutionMTS, MultiStrategyDifferentialEvolution):
    r"""Implementation of Differential Evolution with MTS local searches and multiple mutation strategies.

    Algorithm:
        Differential Evolution with MTS local searches and multiple mutation strategies

    Date:
        2018

    Author:
        Klemen Berkovič

    License:
        MIT

    Attributes:
        Name (List[str]): List of strings representing algorithm name.

    See Also:
        * :class:`niapy.algorithms.modified.hde.DifferentialEvolutionMTS`
        * :class:`niapy.algorithms.basic.de.MultiStrategyDifferentialEvolution`

    """

    Name = ['MultiStrategyDifferentialEvolutionMTS', 'MSDEMTS']

    @staticmethod
    def info():
        r"""Get basic information about the algorithm.

        Returns:
            str: Basic information.

        See Also:
            * :func:`niapy.algorithms.Algorithm.info`

        """
        return r"""TODO"""

    def __init__(self, *args, **kwargs):
        """Initialize MultiStrategyDifferentialEvolutionMTS."""
        super().__init__(individual_type=kwargs.pop('individual_type', MtsIndividual), *args, **kwargs)

    def set_parameters(self, **kwargs):
        r"""Set algorithm parameters.

        See Also:
            * :func:`niapy.algorithms.modified.DifferentialEvolutionMTS.set_parameters`
            * :func:`niapy.algorithms.basic.MultiStrategyDifferentialEvolution.set_parameters`

        """
        DifferentialEvolutionMTS.set_parameters(self, **kwargs)
        MultiStrategyDifferentialEvolution.set_parameters(self, individual_type=kwargs.pop('individual_type', MtsIndividual), **kwargs)

    def evolve(self, pop, xb, task, **kwargs):
        r"""Evolve population.

        Args:
            pop (numpy.ndarray[Individual]): Current population of individuals.
            xb (numpy.ndarray): Global best individual.
            task (Task): Optimization task.

        Returns:
            numpy.ndarray[Individual]: Evolved population.

        """
        return MultiStrategyDifferentialEvolution.evolve(self, pop, xb, task, **kwargs)


class MultiStrategyDifferentialEvolutionMTSv1(MultiStrategyDifferentialEvolutionMTS):
    r"""Implementation of Differential Evolution with MTSv1 local searches and multiple mutation strategies.

    Algorithm:
        Differential Evolution with MTSv1 local searches and multiple mutation strategies

    Date:
        2018

    Author:
        Klemen Berkovič

    License:
        MIT

    Attributes:
        Name (List[str]): List of stings representing algorithm name.

    See Also:
        * :class:`niapy.algorithms.modified.MultiStrategyDifferentialEvolutionMTS`

    """

    Name = ['MultiStrategyDifferentialEvolutionMTSv1', 'MSDEMTSv1']

    @staticmethod
    def info():
        r"""Get basic information about the algorithm.

        Returns:
            str: Basic information.

        See Also:
            * :func:`niapy.algorithms.Algorithm.info`

        """
        return r"""TODO"""

    def __init__(self, *args, **kwargs):
        """Initialize MultiStrategyDifferentialEvolutionMTSv1."""
        super().__init__(local_searches=(mts_ls1v1, mts_ls2, mts_ls3v1), *args, **kwargs)

    def set_parameters(self, **kwargs):
        r"""Set core parameters of MultiStrategyDifferentialEvolutionMTSv1 algorithm.

        See Also:
            * :func:`niapy.algorithms.modified.MultiStrategyDifferentialEvolutionMTS.set_parameters`

        """
        MultiStrategyDifferentialEvolutionMTS.set_parameters(self, local_searches=(mts_ls1v1, mts_ls2, mts_ls3v1), **kwargs)


class DynNpMultiStrategyDifferentialEvolutionMTS(MultiStrategyDifferentialEvolutionMTS, DynNpDifferentialEvolutionMTS):
    r"""Implementation of Differential Evolution with MTS local searches, multiple mutation strategies and dynamic population size.

    Algorithm:
        Differential Evolution with MTS local searches, multiple mutation strategies and dynamic population size

    Date:
        2018

    Author:
        Klemen Berkovič

    License:
        MIT

    Attributes:
        Name (List[str]): List of strings representing algorithm name

    See Also:
        * :class:`niapy.algorithms.modified.MultiStrategyDifferentialEvolutionMTS`
        * :class:`niapy.algorithms.modified.DynNpDifferentialEvolutionMTS`

    """

    Name = ['DynNpMultiStrategyDifferentialEvolutionMTS', 'dynNpMSDEMTS']

    @staticmethod
    def info():
        r"""Get basic information about the algorithm.

        Returns:
            str: Basic information.

        See Also:
            * :func:`niapy.algorithms.Algorithm.info`

        """
        return r"""TODO"""

    def __init__(self, *args, **kwargs):
        """Initialize DynNpMultiStrategyDifferentialEvolutionMTS."""
        super().__init__(*args, **kwargs)

    def set_parameters(self, **kwargs):
        r"""Set core arguments of DynNpMultiStrategyDifferentialEvolutionMTS algorithm.

        See Also:
            * :func:`niapy.algorithms.modified.MultiStrategyDifferentialEvolutionMTS.set_parameters`
            * :func:`niapy.algorithms.modified.DynNpDifferentialEvolutionMTS.set_parameters`

        """
        DynNpDifferentialEvolutionMTS.set_parameters(self, **kwargs)
        MultiStrategyDifferentialEvolutionMTS.set_parameters(self, **kwargs)


class DynNpMultiStrategyDifferentialEvolutionMTSv1(DynNpMultiStrategyDifferentialEvolutionMTS):
    r"""Implementation of Differential Evolution with MTSv1 local searches, multiple mutation strategies and dynamic population size.

    Algorithm:
        Differential Evolution with MTSv1 local searches, multiple mutation strategies and dynamic population size

    Date:
        2018

    Author:
        Klemen Berkovič

    License:
        MIT

    Attributes:
        Name (List[str]): List of strings representing algorithm name.

    See Also:
        * :class:`niapy.algorithm.modified.DynNpMultiStrategyDifferentialEvolutionMTS`

    """

    Name = ['DynNpMultiStrategyDifferentialEvolutionMTSv1', 'dynNpMSDEMTSv1']

    @staticmethod
    def info():
        r"""Get basic information about the algorithm.

        Returns:
            str: Basic information.

        See Also:
            * :func:`niapy.algorithms.Algorithm.info`

        """
        return r"""TODO"""

    def __init__(self, *args, **kwargs):
        """Initialize DynNpMultiStrategyDifferentialEvolutionMTSv1."""
        super().__init__(local_searches=(mts_ls1v1, mts_ls2, mts_ls3v1), *args, **kwargs)

    def set_parameters(self, **kwargs):
        r"""Set core parameters of DynNpMultiStrategyDifferentialEvolutionMTSv1 algorithm.

        See Also:
            * :func:`niapy.algorithm.modified.DynNpMultiStrategyDifferentialEvolutionMTS.set_parameters`

        """
        DynNpMultiStrategyDifferentialEvolutionMTS.set_parameters(self, local_searches=(mts_ls1v1, mts_ls2, mts_ls3v1), **kwargs)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
