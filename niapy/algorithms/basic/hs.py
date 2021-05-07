# encoding=utf8
import logging

import numpy as np

from niapy.algorithms.algorithm import Algorithm

logging.basicConfig()
logger = logging.getLogger("niapy.algorithms.basic")
logger.setLevel("INFO")

__all__ = ["HarmonySearch", "HarmonySearchV1"]


class HarmonySearch(Algorithm):
    r"""Implementation of Harmony Search algorithm.

    Algorithm:
        Harmony Search Algorithm

    Date:
        2018

    Authors:
        Klemen Berkovič

    License:
        MIT

    Reference URL:
        https://journals.sagepub.com/doi/10.1177/003754970107600201

    Reference paper:
        Geem, Z. W., Kim, J. H., & Loganathan, G. V. (2001). A new heuristic optimization algorithm: harmony search. Simulation, 76(2), 60-68.

    Attributes:
        Name (List[str]): List of strings representing algorithm names
        r_accept (float): Probability of accepting new bandwidth into harmony.
        r_pa (float): Probability of accepting random bandwidth into harmony.
        b_range (float): Range of bandwidth.

    See Also:
        * :class:`niapy.algorithms.algorithm.Algorithm`

    """

    Name = ["HarmonySearch", "HS"]

    @staticmethod
    def info():
        r"""Get basic information about the algorithm.

        Returns:
            str: Basic information.

        """
        return r"""Geem, Z. W., Kim, J. H., & Loganathan, G. V. (2001). A new heuristic optimization algorithm: harmony search. Simulation, 76(2), 60-68."""

    def __init__(self, population_size=30, r_accept=0.7, r_pa=0.35, b_range=1.42, *args, **kwargs):
        """Initialize HarmonySearch.

        Args:
            population_size (Optional[int]): Number of harmony in the memory.
            r_accept (Optional[float]): Probability of accepting new bandwidth to harmony.
            r_pa (Optional[float]): Probability of accepting random bandwidth into harmony.
            b_range (Optional[float]): Bandwidth range.

        """
        super().__init__(population_size, *args, **kwargs)
        self.r_accept = r_accept
        self.r_pa = r_pa
        self.b_range = b_range

    def set_parameters(self, population_size=30, r_accept=0.7, r_pa=0.35, b_range=1.42, **kwargs):
        r"""Set the arguments of the algorithm.

        Args:
            population_size (Optional[int]): Number of harmony in the memory.
            r_accept (Optional[float]): Probability of accepting new bandwidth to harmony.
            r_pa (Optional[float]): Probability of accepting random bandwidth into harmony.
            b_range (Optional[float]): Bandwidth range.

        See Also:
            * :func:`niapy.algorithms.algorithm.Algorithm.set_parameters`

        """
        super().set_parameters(population_size=population_size, **kwargs)
        self.r_accept = r_accept
        self.r_pa = r_pa
        self.b_range = b_range

    def get_parameters(self):
        """Get algorithm parameters."""
        d = super().get_parameters()
        d.update({
            'r_accept': self.r_accept,
            'r_pa': self.r_pa,
            'b_range': self.b_range
        })
        return d

    def bw(self, task):
        r"""Get bandwidth.

        Args:
            task (Task): Optimization task.

        Returns:
            float: Bandwidth.

        """
        return self.uniform(-1, 1) * self.b_range

    def adjustment(self, x, task):
        r"""Adjust value based on bandwidth.

        Args:
            x (Union[int, float]): Current position.
            task (Task): Optimization task.

        Returns:
            float: New position.

        """
        return x + self.bw(task)

    def improvise(self, harmonies, task):
        r"""Create new individual.

        Args:
            harmonies (numpy.ndarray): Current population.
            task (Task): Optimization task.

        Returns:
            numpy.ndarray: New individual.

        """
        harmony = np.zeros(task.dimension)
        for i in range(task.dimension):
            r, j = self.random(), self.integers(self.population_size)
            harmony[i] = harmonies[j, i] if r > self.r_accept else self.adjustment(harmonies[j, i], task) if r > self.r_pa else self.uniform(
                task.lower[i], task.upper[i])
        return harmony

    def run_iteration(self, task, population, population_fitness, best_x, best_fitness, **params):
        r"""Core function of HarmonySearch algorithm.

        Args:
            task (Task): Optimization task.
            population (numpy.ndarray): Current population.
            population_fitness (numpy.ndarray): Current populations function/fitness values.
            best_x (numpy.ndarray): Global best individual.
            best_fitness (float): Global best fitness/function value.
            **params (Dict[str, Any]): Additional arguments.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, float, Dict[str, Any]]:
                1. New harmony/population.
                2. New populations function/fitness values.
                3. New global best solution
                4. New global best solution fitness/objective value
                5. Additional arguments.

        """
        harmony = self.improvise(population, task)
        harmony_fitness = task.eval(task.repair(harmony, self.rng))
        iw = np.argmax(population_fitness)
        if harmony_fitness <= population_fitness[iw]:
            population[iw], population_fitness[iw] = harmony, harmony_fitness
        best_x, best_fitness = self.get_best(harmony, harmony_fitness, best_x, best_fitness)
        return population, population_fitness, best_x, best_fitness, {}


class HarmonySearchV1(HarmonySearch):
    r"""Implementation of harmony search algorithm.

    Algorithm:
        Harmony Search Algorithm

    Date:
        2018

    Authors:
        Klemen Berkovič

    License:
        MIT

    Reference URL:
        https://link.springer.com/chapter/10.1007/978-3-642-00185-7_1

    Reference paper:
        Yang, Xin-She. "Harmony search as a metaheuristic algorithm." Music-inspired harmony search algorithm. Springer, Berlin, Heidelberg, 2009. 1-14.

    Attributes:
        Name (List[str]): List of strings representing algorithm name.
        bw_min (float): Minimal bandwidth.
        bw_max (float): Maximal bandwidth.

    See Also:
        * :class:`niapy.algorithms.basic.hs.HarmonySearch`

    """

    Name = ["HarmonySearchV1", "HSv1"]

    @staticmethod
    def info():
        r"""Get basic information about algorithm.

        Returns:
            str: Basic information.

        """
        return r"""Yang, Xin-She. "Harmony search as a metaheuristic algorithm." Music-inspired harmony search algorithm. Springer, Berlin, Heidelberg, 2009. 1-14."""

    def __init__(self, bw_min=1, bw_max=2, *args, **kwargs):
        """Initialize HarmonySearchV1.

        Args:
            bw_min (Optional[float]): Minimal bandwidth.
            bw_max (Optional[float]): Maximal bandwidth.

        """
        super().__init__(*args, **kwargs)
        self.bw_min = bw_min
        self.bw_max = bw_max

    def set_parameters(self, bw_min=1, bw_max=2, **kwargs):
        r"""Set the parameters of the algorithm.

        Args:
            bw_min (Optional[float]): Minimal bandwidth
            bw_max (Optional[float]): Maximal bandwidth

        See Also:
            * :func:`niapy.algorithms.basic.hs.HarmonySearch.set_parameters`

        """
        super().set_parameters(**kwargs)
        self.bw_min, self.bw_max = bw_min, bw_max

    def get_parameters(self):
        """Get algorithm parameters."""
        d = super().get_parameters()
        d.update({
            'bw_min': self.bw_min,
            'bw_max': self.bw_max
        })
        return d

    def bw(self, task):
        r"""Get new bandwidth.

        Args:
            task (Task): Optimization task.

        Returns:
            float: New bandwidth.

        """
        return self.bw_min * np.exp(np.log(self.bw_min / self.bw_max) * (task.iters + 1) / task.max_iters)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
