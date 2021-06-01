# encoding=utf8
import logging

import numpy as np

from niapy.algorithms.algorithm import Algorithm
from niapy.util import euclidean

logging.basicConfig()
logger = logging.getLogger('niapy.algorithms.basic')
logger.setLevel('INFO')

__all__ = ['GlowwormSwarmOptimization', 'GlowwormSwarmOptimizationV1', 'GlowwormSwarmOptimizationV2',
           'GlowwormSwarmOptimizationV3']


class GlowwormSwarmOptimization(Algorithm):
    r"""Implementation of glowworm swarm optimization.

    Algorithm:
        Glowworm Swarm Optimization Algorithm

    Date:
        2018

    Authors:
        Klemen Berkovič

    License:
        MIT

    Reference URL:
        https://www.springer.com/gp/book/9783319515946

    Reference paper:
        Kaipa, Krishnanand N., and Debasish Ghose. Glowworm swarm optimization: theory, algorithms, and applications. Vol. 698. Springer, 2017.

    Attributes:
        Name (List[str]): List of strings representing algorithm name.
        l0 (float): Initial luciferin quantity for each glowworm.
        nt (float): Number of neighbors.
        rho (float): Luciferin decay constant.
        gamma (float): Luciferin enhancement constant.
        beta (float): Constant.
        s (float): Step size.
        distance (Callable[[numpy.ndarray, numpy.ndarray], float]]): Measure distance between two individuals.

    See Also:
        * :class:`NiaPy.algorithms.algorithm.Algorithm`

    """

    Name = ['GlowwormSwarmOptimization', 'GSO']

    @staticmethod
    def info():
        r"""Get basic information of algorithm.

        Returns:
            str: Basic information.

        """
        return r"""Kaipa, Krishnanand N., and Debasish Ghose. Glowworm swarm optimization: theory, algorithms, and applications. Vol. 698. Springer, 2017."""

    def __init__(self, population_size=25, l0=5, nt=5, rho=0.4, gamma=0.6, beta=0.08, s=0.03, distance=euclidean, *args,
                 **kwargs):
        """Initialize GlowwormSwarmOptimization.

        Args:
            population_size (Optional[int]): Number of glowworms in population.
            l0 (Optional[float]): Initial luciferin quantity for each glowworm.
            nt (Optional[int]): Number of neighbors.
            rho (Optional[float]): Luciferin decay constant.
            gamma (Optional[float]): Luciferin enhancement constant.
            beta (Optional[float]): Constant.
            s (Optional[float]): Step size.
            distance (Optional[Callable[[numpy.ndarray, numpy.ndarray], float]]]): Measure distance between two individuals.

        """
        super().__init__(population_size, *args, **kwargs)
        self.l0 = l0
        self.nt = nt
        self.rho = rho
        self.gamma = gamma
        self.beta = beta
        self.s = s
        self.distance = distance

    def set_parameters(self, population_size=25, l0=5, nt=5, rho=0.4, gamma=0.6, beta=0.08, s=0.03, distance=euclidean,
                       **kwargs):
        r"""Set the arguments of an algorithm.

        Args:
            population_size (Optional[int]): Number of glowworms in population.
            l0 (Optional[float]): Initial luciferin quantity for each glowworm.
            nt (Optional[int]): Number of neighbors.
            rho (Optional[float]): Luciferin decay constant.
            gamma (Optional[float]): Luciferin enhancement constant.
            beta (Optional[float]): Constant.
            s (Optional[float]): Step size.
            distance (Optional[Callable[[numpy.ndarray, numpy.ndarray], float]]]): Measure distance between two individuals.

        """
        super().set_parameters(population_size=population_size, **kwargs)
        self.l0 = l0
        self.nt = nt
        self.rho = rho
        self.gamma = gamma
        self.beta = beta
        self.s = s
        self.distance = distance

    def get_parameters(self):
        r"""Get algorithms parameters values.

        Returns:
            Dict[str, Any]: Algorithm parameters.

        """
        d = super().get_parameters()
        d.update({
            'l0': self.l0,
            'nt': self.nt,
            'rho': self.rho,
            'gamma': self.gamma,
            'beta': self.beta,
            's': self.s,
            'distance': self.distance
        })
        return d

    def get_neighbors(self, i, r, glowworms, luciferin):
        r"""Get neighbours of glowworm.

        Args:
            i (int): Index of glowworm.
            r (float): Neighborhood distance.
            glowworms (numpy.ndarray):
            luciferin (numpy.ndarray[float]): Luciferin value of glowworm.

        Returns:
            numpy.ndarray[int]: Indexes of neighborhood glowworms.

        """
        neighbors = np.zeros(self.population_size, dtype=np.int32)
        for j, gw in enumerate(glowworms):
            neighbors[j] = 1 if i != j and self.distance(glowworms[i], gw) <= r and luciferin[i] >= luciferin[j] else 0
        return neighbors

    def probabilities(self, i, neighbors, luciferin):
        r"""Calculate probabilities for glowworm to movement.

        Args:
            i (int): Index of glowworm to search for probable movement.
            neighbors (numpy.ndarray[float]):
            luciferin (numpy.ndarray[float]):

        Returns:
            numpy.ndarray[float]: Probabilities for each glowworm in swarm.

        """
        d = np.sum(luciferin[neighbors == 1] - luciferin[i]) + np.finfo(float).eps
        probabilities = np.zeros(self.population_size)
        probabilities[neighbors == 1] = (luciferin[neighbors == 1] - luciferin[i]) / d
        return probabilities

    def move_select(self, pb, i):
        r"""Get move index for the i-th glowworm.

        Args:
            pb (numpy.ndarray): Probabilities.
            i (int): Index of the glowworm.

        Returns:
            int: Index i-th glowworm will move towards.

        """
        r, b_l, b_u = self.random(), 0, 0
        for j in range(self.population_size):
            b_l, b_u = b_u, b_u + pb[i]
            if b_l < r < b_u:
                return j
        return self.integers(self.population_size)

    def calculate_luciferin(self, luciferin, fitness):
        return (1 - self.rho) * luciferin + self.gamma * fitness

    def range_update(self, range_, neighbors, sensing_range):
        """Update range."""
        return range_ + self.beta * (self.nt - np.sum(neighbors))

    def init_population(self, task):
        r"""Initialize population.

        Args:
            task (Task): Optimization task.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray[float], Dict[str, Any]]:
                1. Initialized population of glowworms.
                2. Initialized populations function/fitness values.
                3. Additional arguments:
                    * luciferin (numpy.ndarray): Luciferin values of glowworms.
                    * ranges (numpy.ndarray): Ranges.
                    * sensing_range (float): Sensing range.

        """
        population, fitness, d = super().init_population(task)
        sensing_range = euclidean(np.zeros(task.dimension), task.range)
        luciferin = np.full(self.population_size, self.l0)
        ranges = np.full(self.population_size, sensing_range)
        d.update({'luciferin': luciferin, 'ranges': ranges, 'sensing_range': sensing_range})
        return population, fitness, d

    def run_iteration(self, task, population, population_fitness, best_x, best_fitness, **params):
        r"""Core function of GlowwormSwarmOptimization algorithm.

        Args:
            task (Task): Optimization task.
            population (numpy.ndarray): Current population.
            population_fitness (numpy.ndarray): Current populations fitness/function values.
            best_x (numpy.ndarray): Global best individual.
            best_fitness (float): Global best individuals function/fitness value.
            **params Dict[str, Any]: Additional arguments.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, float, Dict[str, Any]]:
                1. Initialized population of glowworms.
                2. Initialized populations function/fitness values.
                3. New global best solution
                4. New global best solutions fitness/objective value.
                5. Additional arguments:
                    * luciferin (numpy.ndarray): Luciferin values of glowworms.
                    * ranges (numpy.ndarray): Ranges.
                    * sensing_range (float): Sensing range.

        """
        luciferin = params.pop('luciferin')
        ranges = params.pop('ranges')
        sensing_range = params.pop('sensing_range')

        old_population, old_ranges = np.copy(population), np.copy(ranges)
        luciferin = self.calculate_luciferin(luciferin, population_fitness)
        neighbors = [self.get_neighbors(i, old_ranges[i], old_population, luciferin) for i in range(self.population_size)]
        probabilities = [self.probabilities(i, neighbors[i], luciferin) for i in range(self.population_size)]
        j = [self.move_select(probabilities[i], i) for i in range(self.population_size)]
        for i in range(self.population_size):
            distance = self.distance(old_population[j[i]], old_population[i])
            new_glowworm = old_population[i] + self.s * ((old_population[j[i]] - old_population[i]) / (distance + 1e-31))
            population[i] = task.repair(new_glowworm, rng=self.rng)
        for i in range(self.population_size):
            ranges[i] = max(0.0, min(sensing_range, self.range_update(old_ranges[i], neighbors[i], sensing_range)))
        population_fitness = np.apply_along_axis(task.eval, 1, population)
        best_x, best_fitness = self.get_best(population, population_fitness, best_x, best_fitness)
        return population, population_fitness, best_x, best_fitness, {'luciferin': luciferin,
                                                                      'ranges': ranges,
                                                                      'sensing_range': sensing_range}


class GlowwormSwarmOptimizationV1(GlowwormSwarmOptimization):
    r"""Implementation of glowworm swarm optimization.

    Algorithm:
        Glowworm Swarm Optimization Algorithm

    Date:
        2018

    Authors:
        Klemen Berkovič

    License:
        MIT

    Reference URL:
        https://www.springer.com/gp/book/9783319515946

    Reference paper:
        Kaipa, Krishnanand N., and Debasish Ghose. Glowworm swarm optimization: theory, algorithms, and applications. Vol. 698. Springer, 2017.

    Attributes:
        Name (List[str]): List of strings representing algorithm names.

    See Also:
        * :class:`NiaPy.algorithms.basic.GlowwormSwarmOptimization`

    """

    Name = ['GlowwormSwarmOptimizationV1', 'GSOv1']

    @staticmethod
    def info():
        r"""Get basic information of algorithm.

        Returns:
            str: Basic information.

        """
        return r"""Kaipa, Krishnanand N., and Debasish Ghose. Glowworm swarm optimization: theory, algorithms, and applications. Vol. 698. Springer, 2017."""

    def calculate_luciferin(self, luciferin, fitness):
        r = super().calculate_luciferin(luciferin, fitness)
        return np.fmax(0.0, r)

    def range_update(self, range_, neighbors, sensing_range):
        """Update range."""
        return sensing_range / (1 + self.beta * (np.sum(neighbors) / (np.pi * sensing_range ** 2)))


class GlowwormSwarmOptimizationV2(GlowwormSwarmOptimization):
    r"""Implementation of glowworm swarm optimization.

    Algorithm:
        Glowworm Swarm Optimization Algorithm

    Date:
        2018

    Authors:
        Klemen Berkovič

    License:
        MIT

    Reference URL:
        https://www.springer.com/gp/book/9783319515946

    Reference paper:
        Kaipa, Krishnanand N., and Debasish Ghose. Glowworm swarm optimization: theory, algorithms, and applications. Vol. 698. Springer, 2017.

    Attributes:
        Name (List[str]): List of strings representing algorithm names.
        alpha (float): --

    See Also:
        * :class:`NiaPy.algorithms.basic.GlowwormSwarmOptimization`

    """

    Name = ['GlowwormSwarmOptimizationV2', 'GSOv2']

    @staticmethod
    def info():
        r"""Get basic information of algorithm.

        Returns:
            str: Basic information.

        """
        return r"""Kaipa, Krishnanand N., and Debasish Ghose. Glowworm swarm optimization: theory, algorithms, and applications. Vol. 698. Springer, 2017."""

    def __init__(self, alpha=0.2, *args, **kwargs):
        """Initialize GlowwormSwarmOptimizationV2.

        Args:
            alpha (Optional[float]): Alpha parameter.

        """
        super().__init__(*args, **kwargs)
        self.alpha = alpha

    def set_parameters(self, alpha=0.2, **kwargs):
        r"""Set core parameters for GlowwormSwarmOptimizationV2 algorithm.

        Args:
            alpha (Optional[float]): Alpha parameter.

        See Also:
            * :func:`NiaPy.algorithms.basic.GlowwormSwarmOptimization.set_parameters`

        """
        super().set_parameters(**kwargs)
        self.alpha = alpha

    def range_update(self, range_, neighbors, sensing_range):
        """Update range."""
        return self.alpha + (sensing_range - self.alpha) / (1 + self.beta * np.sum(neighbors))


class GlowwormSwarmOptimizationV3(GlowwormSwarmOptimization):
    r"""Implementation of glowworm swarm optimization.

    Algorithm:
        Glowworm Swarm Optimization Algorithm

    Date:
        2018

    Authors:
        Klemen Berkovič

    License:
        MIT

    Reference URL:
        https://www.springer.com/gp/book/9783319515946

    Reference paper:
        Kaipa, Krishnanand N., and Debasish Ghose. Glowworm swarm optimization: theory, algorithms, and applications. Vol. 698. Springer, 2017.

    Attributes:
        Name (List[str]): List of strings representing algorithm names.
        beta1 (float): --

    See Also:
        * :class:`NiaPy.algorithms.basic.GlowwormSwarmOptimization`

    """

    Name = ['GlowwormSwarmOptimizationV3', 'GSOv3']

    @staticmethod
    def info():
        r"""Get basic information of algorithm.

        Returns:
            str: Basic information.

        """
        return r"""Kaipa, Krishnanand N., and Debasish Ghose. Glowworm swarm optimization: theory, algorithms, and applications. Vol. 698. Springer, 2017."""

    def __init__(self, beta1=0.2, *args, **kwargs):
        """Initialize GlowwormSwarmOptimizationV3.

        Args:
            beta1 (Optional[float]): Beta1 parameter.

        """
        super().__init__(*args, **kwargs)
        self.beta1 = beta1

    def set_parameters(self, beta1=0.2, **kwargs):
        r"""Set core parameters for GlowwormSwarmOptimizationV3 algorithm.

        Args:
            beta1 (Optional[float]): Beta1 parameter.

        See Also:
            * :func:`NiaPy.algorithms.basic.GlowwormSwarmOptimization.set_parameters`

        """
        super().set_parameters(**kwargs)
        self.beta1 = beta1

    def range_update(self, range_, neighbors, sensing_range):
        """Update range."""
        return range_ + (self.beta * np.sum(neighbors)) if np.sum(neighbors) < self.nt else (-self.beta1 * np.sum(neighbors))

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
