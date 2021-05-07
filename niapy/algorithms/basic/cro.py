# encoding=utf8
import logging

import numpy as np

from niapy.algorithms.algorithm import Algorithm

logging.basicConfig()
logger = logging.getLogger('niapy.algorithms.basic')
logger.setLevel('INFO')

__all__ = ['CoralReefsOptimization']


def default_sexual_crossover(pop, p, task, rng, **_kwargs):
    r"""Sexual reproduction of corals.

    Args:
        pop (numpy.ndarray): Current population.
        p (float): Probability in range [0, 1].
        task (Task): Optimization task.
        rng (numpy.random.Generator): Random generator.

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray]:
            1. New population.
            2. New population function/fitness values.

    """
    for i in range(len(pop) // 2):
        pop[i] = np.asarray([pop[i, d] if rng.random() < p else pop[i * 2, d] for d in range(task.dimension)])
    return pop, np.apply_along_axis(task.eval, 1, pop)


def default_brooding(pop, p, task, rng, **_kwargs):
    r"""Brooding or internal sexual reproduction of corals.

    Args:
        pop (numpy.ndarray): Current population.
        p (float): Probability in range [0, 1].
        task (Task): Optimization task.
        rng (numpy.random.Generator): Random generator.

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray]:
            1. New population.
            2. New population function/fitness values.

    """
    for i in range(len(pop)):
        pop[i] = task.repair(np.asarray(
            [pop[i, d] if rng.random() < p else task.lower[d] + task.range[d] * rng.random() for d in
             range(task.dimension)]), rng=rng)
    return pop, np.apply_along_axis(task.eval, 1, pop)


def move_corals(pop, p, f, task, rng, **_kwargs):
    r"""Move corals.

    Args:
        pop (numpy.ndarray): Current population.
        p (float): Probability in range [0, 1].
        f (float): Factor.
        task (Task): Optimization task.
        rng (numpy.random.Generator): Random generator.

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray]:
            1. New population.
            2. New population function/fitness values.

    """
    for i in range(len(pop)):
        pop[i] = task.repair(
            np.asarray(
                [pop[i, d] if rng.random() < p else pop[i, d] + f * rng.random() for d in range(task.dimension)]),
            rng=rng)
    return pop, np.apply_along_axis(task.eval, 1, pop)


class CoralReefsOptimization(Algorithm):
    r"""Implementation of Coral Reefs Optimization Algorithm.

    Algorithm:
        Coral Reefs Optimization Algorithm

    Date:
        2018

    Authors:
        Klemen Berkovič

    License:
        MIT

    Reference Paper:
        S. Salcedo-Sanz, J. Del Ser, I. Landa-Torres, S. Gil-López, and J. A. Portilla-Figueras,
        “The Coral Reefs Optimization Algorithm: A Novel Metaheuristic for Efficiently Solving Optimization Problems,”
        The Scientific World Journal, vol. 2014, Article ID 739768, 15 pages, 2014.

    Reference URL:
        https://doi.org/10.1155/2014/739768.

    Attributes:
        Name (List[str]): List of strings representing algorithm name.
        phi (float): Range of neighborhood.
        num_asexual_reproduction (int): Number of corals used in asexual reproduction.
        num_broadcast (int): Number of corals used in brooding.
        num_depredation (int): Number of corals used in depredation.
        k (int): Number of tries for larva setting.
        mutation_rate (float): Mutation variable :math:`\in [0, \infty]`.
        crossover_rate(float): Crossover rate in [0, 1].
        sexual_crossover (Callable[[numpy.ndarray, float, Task, numpy.random.Generator, Dict[str, Any]], Tuple[numpy.ndarray, numpy.ndarray[float]]]):
            Crossover function.
        brooding (Callable[[numpy.ndarray, float, Task, numpy.random.Generator, Dict[str, Any]], Tuple[numpy.ndarray, numpy.ndarray]]):
            Brooding function.

    See Also:
        * :class:`niapy.algorithms.Algorithm`

    """

    Name = ['CoralReefsOptimization', 'CRO']

    @staticmethod
    def info():
        r"""Get algorithms information.

        Returns:
            str: Algorithm information.

        See Also:
            * :func:`niapy.algorithms.Algorithm.info`

        """
        return r"""S. Salcedo-Sanz, J. Del Ser, I. Landa-Torres, S. Gil-López, and J. A. Portilla-Figueras,
        “The Coral Reefs Optimization Algorithm: A Novel Metaheuristic for Efficiently Solving Optimization Problems,”
        The Scientific World Journal, vol. 2014, Article ID 739768, 15 pages, 2014."""

    def __init__(self, population_size=25, phi=0.4, asexual_reproduction_prob=0.5, broadcast_prob=0.5,
                 depredation_prob=0.3, k=25, crossover_rate=0.5, mutation_rate=0.36,
                 sexual_crossover=default_sexual_crossover, brooding=default_brooding, *args, **kwargs):
        r"""Initialize CoralReefsOptimization.

        Args:
            population_size (int): population size for population initialization.
            phi (int): distance.
            asexual_reproduction_prob (float): Value $\in [0, 1]$ for Asexual reproduction size.
            broadcast_prob (float): Value $\in [0, 1]$ for brooding size.
            depredation_prob (float): Value $\in [0, 1]$ for Depredation size.
            k (int): Tries for larvae setting.
            sexual_crossover (Callable[[numpy.ndarray, float, Task, numpy.random.Generator, Dict[str, Any]], Tuple[numpy.ndarray, numpy.ndarray]]):
                Crossover function.
            crossover_rate (float): Crossover rate $\in [0, 1]$.
            brooding (Callable[[numpy.ndarray, float, Task, numpy.random.Generator, Dict[str, Any]], Tuple[numpy.ndarray, numpy.ndarray]]):
                brooding function.
            mutation_rate (float): Crossover rate $\in [0, 1]$.

        See Also:
            * :func:`niapy.algorithms.Algorithm.__init__`

        """
        super().__init__(population_size, *args, **kwargs)
        self.phi = phi
        self.k = k
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.num_asexual_reproduction = int(self.population_size * asexual_reproduction_prob)
        self.num_broadcast = int(self.population_size * broadcast_prob)
        self.num_depredation = int(self.population_size * depredation_prob)
        self.sexual_crossover = sexual_crossover
        self.brooding = brooding

    def set_parameters(self, population_size=25, phi=0.4, asexual_reproduction_prob=0.5, broadcast_prob=0.5,
                       depredation_prob=0.3, k=25, crossover_rate=0.5, mutation_rate=0.36,
                       sexual_crossover=default_sexual_crossover, brooding=default_brooding, **kwargs):
        r"""Set the parameters of the algorithm.

        Args:
            population_size (int): population size for population initialization.
            phi (int): distance.
            asexual_reproduction_prob (float): Value $\in [0, 1]$ for Asexual reproduction size.
            broadcast_prob (float): Value $\in [0, 1]$ for brooding size.
            depredation_prob (float): Value $\in [0, 1]$ for Depredation size.
            k (int): Tries for larvae setting.
            sexual_crossover (Callable[[numpy.ndarray, float, Task, numpy.random.Generator, Dict[str, Any]], Tuple[numpy.ndarray, numpy.ndarray]]):
                Crossover function.
            crossover_rate (float): Crossover rate $\in [0, 1]$.
            brooding (Callable[[numpy.ndarray, float, Task, numpy.random.Generator, Dict[str, Any]], Tuple[numpy.ndarray, numpy.ndarray]]):
                brooding function.
            mutation_rate (float): Crossover rate $\in [0, 1]$.

        See Also:
            * :func:`niapy.algorithms.Algorithm.set_parameters`

        """
        super().set_parameters(self, population_size=population_size, **kwargs)
        self.phi = phi
        self.k = k
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.num_asexual_reproduction = int(self.population_size * asexual_reproduction_prob)
        self.num_broadcast = int(self.population_size * broadcast_prob)
        self.num_depredation = int(self.population_size * depredation_prob)
        self.sexual_crossover = sexual_crossover
        self.brooding = brooding

    def get_parameters(self):
        r"""Get parameters values of the algorithm.

        Returns:
            Dict[str, Any]: Algorithm parameters.

        """
        d = super().get_parameters()
        d.update({
            'phi': self.phi,
            'k': self.k,
            'crossover_rate': self.crossover_rate,
            'mutation_rate': self.mutation_rate,
            'num_asexual_reproduction': self.num_asexual_reproduction,
            'num_depredation': self.num_depredation,
            'num_broadcast': self.num_broadcast
        })
        return d

    def asexual_reproduction(self, reef, reef_fitness, best_x, best_fitness, task):
        r"""Asexual reproduction of corals.

        Args:
            reef (numpy.ndarray): Current population of reefs.
            reef_fitness (numpy.ndarray): Current populations function/fitness values.
            best_x (numpy.ndarray): Global best coordinates.
            best_fitness (float): Global best fitness.
            task (Task): Optimization task.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray]:
                1. New population.
                2. New population fitness/function values.

        See Also:
            * :func:`niapy.algorithms.basic.CoralReefsOptimization.setting`
            * :func:`niapy.algorithms.basic.default_brooding`

        """
        brooding_indices = np.argsort(reef_fitness)[:self.num_asexual_reproduction]
        new_reef, new_reef_fitness = self.brooding(reef[brooding_indices], self.mutation_rate, task, rng=self.rng)
        best_x, best_fitness = self.get_best(new_reef, new_reef_fitness, best_x, best_fitness)
        reef, reef_fitness, best_x, best_fitness = self.settling(reef, reef_fitness, new_reef, new_reef_fitness, best_x,
                                                                 best_fitness, task)
        return reef, reef_fitness, best_x, best_fitness

    def depredation(self, reef, reef_fitness):
        r"""Depredation operator for reefs.

        Args:
            reef (numpy.ndarray): Current reefs.
            reef_fitness (numpy.ndarray): Current reefs function/fitness values.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray]:
                1. Best individual
                2. Best individual fitness/function value

        """
        depredation_index = np.argsort(reef_fitness)[::-1][:self.num_depredation]
        return np.delete(reef, depredation_index), np.delete(reef_fitness, depredation_index)

    def settling(self, reef, reef_fitness, new_reef, new_reef_fitness, best_x, best_fitness, task):
        r"""Operator for setting reefs.

        New reefs try to settle to selected position in search space.
        New reefs are successful if their fitness values is better or if they have no reef occupying same search space.

        Args:
            reef (numpy.ndarray): Current population of reefs.
            reef_fitness (numpy.ndarray): Current populations function/fitness values.
            new_reef (numpy.ndarray): New population of reefs.
            new_reef_fitness (numpy.ndarray): New populations function/fitness values.
            best_x (numpy.ndarray): Global best solution.
            best_fitness (float): Global best solutions fitness/objective value.
            task (Task): Optimization task.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, float]:
                1. New settled population.
                2. New settled population fitness/function values.

        """
        def update(arr, phi, xb, fxb):
            dist = np.asarray([np.sqrt(np.sum((arr - e) ** 2, axis=1)) for e in new_reef])
            ind = np.unique(np.where(dist < phi)[0])
            if ind.any():
                new_reef[ind], new_reef_fitness[ind] = move_corals(new_reef[ind],
                                                                   self.mutation_rate,
                                                                   self.mutation_rate,
                                                                   task,
                                                                   rng=self.rng)
                xb, fxb = self.get_best(new_reef[ind], new_reef_fitness[ind], xb, fxb)
            return xb, fxb

        for i in range(self.k):
            best_x, best_fitness = update(reef, self.phi, best_x, best_fitness)
            best_x, best_fitness = update(new_reef, self.phi, best_x, best_fitness)
        distances = np.asarray([np.sqrt(np.sum((reef - e) ** 2, axis=1)) for e in new_reef])
        indices = np.unique(np.where(distances >= self.phi)[0])
        return np.append(reef, new_reef[indices], 0), np.append(reef_fitness, new_reef_fitness[indices], 0), best_x, best_fitness

    def run_iteration(self, task, population, population_fitness, best_x, best_fitness, **params):
        r"""Core function of Coral Reefs Optimization algorithm.

        Args:
            task (Task): Optimization task.
            population (numpy.ndarray): Current population.
            population_fitness (numpy.ndarray): Current population fitness/function value.
            best_x (numpy.ndarray): Global best solution.
            best_fitness (float): Global best solution fitness/function value.
            **params: Additional arguments

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, float, Dict[str, Any]]:
                1. New population.
                2. New population fitness/function values.
                3. New global best solution
                4. New global best solutions fitness/objective value
                5. Additional arguments:

        See Also:
            * :func:`niapy.algorithms.basic.CoralReefsOptimization.sexual_crossover`
            * :func:`niapy.algorithms.basic.CoralReefsOptimization.brooding`

        """
        broadcast_index = self.rng.choice(len(population), size=self.num_broadcast, replace=False)
        reef_broadcast, reef_broadcast_fitness = self.sexual_crossover(population[broadcast_index],
                                                                       self.crossover_rate,
                                                                       task,
                                                                       rng=self.rng)
        best_x, best_fitness = self.get_best(reef_broadcast, reef_broadcast_fitness, best_x, best_fitness)
        reef_brooding, reef_brooding_fitness = self.brooding(np.delete(population, broadcast_index, 0),
                                                             self.mutation_rate,
                                                             task,
                                                             rng=self.rng)
        best_x, best_fitness = self.get_best(reef_brooding, reef_brooding_fitness, best_x, best_fitness)
        new_reef, new_reef_fitness, best_x, best_fitness = self.settling(population,
                                                                         population_fitness,
                                                                         np.append(reef_broadcast, reef_brooding, 0),
                                                                         np.append(reef_broadcast_fitness,
                                                                                   reef_brooding_fitness, 0), best_x,
                                                                         best_fitness,
                                                                         task)
        population, population_fitness, best_x, best_fitness = self.asexual_reproduction(new_reef,
                                                                                         new_reef_fitness,
                                                                                         best_x,
                                                                                         best_fitness,
                                                                                         task)
        if (task.iters + 1) % self.k == 0:
            population, population_fitness = self.depredation(population, population_fitness)
        return population, population_fitness, best_x, best_fitness, {}

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
