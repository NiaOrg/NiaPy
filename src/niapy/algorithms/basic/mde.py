# encoding=utf8

"""Multi-population based Differential Evolution module."""

import logging

import numpy as np

from niapy.algorithms.algorithm import Algorithm, default_numpy_init

logging.basicConfig()
logger = logging.getLogger('niapy.algorithms.basic')
logger.setLevel('INFO')

__all__ = ['MultiPopulationDifferentialEvolution']


class MultiPopulationDifferentialEvolution(Algorithm):
    r"""Implementation of Multi-population based Differential Evolution algorithm (MDE).

    Algorithm:
        Multi-population based Differential Evolution

    Date:
        2026

    Author:
        Ahmet Emin Karkinli

    License:
        MIT

    Reference paper:
        Karkinli, A. E. (2023). Detection of object boundary from point cloud by using multi-population based
        differential evolution algorithm. Neural Computing and Applications, 35, 5193-5206.
        https://doi.org/10.1007/s00521-022-07969-w

    Reference URL:
        https://github.com/Aekarkinli/Multi-Population-based-Differential-Evolution-Algorithm

    MDE is a parameter-free variant of Differential Evolution. It keeps a full population :math:`A`
    of ``population_size`` vectors (:math:`3N` in the paper). In every iteration :math:`N` distinct vectors
    are drawn at random from :math:`A` to form the sub-population :math:`B`, where
    :math:`N = \lfloor \mathit{population\_size} / 3 \rfloor`. For every :math:`\mathbf{b}_i \in B` a trial
    vector is built as

    :math:`\mathbf{t}_i = \mathbf{b}_{r_2} + s_i \cdot (\mathbf{d}^x_i - \mathbf{d}^y_i)`,

    where :math:`r_1 \neq r_2 \neq i`, each component of :math:`\mathbf{d}^x_i` is taken with equal probability
    from :math:`\mathbf{b}_{r_1}` or from the global best solution, each component of :math:`\mathbf{d}^y_i`
    from :math:`\mathbf{b}_i` or :math:`\mathbf{b}_{r_2}`, and the scale factor is
    :math:`s_i = |a - u^{b}| \cdot z^{d}` with :math:`a \sim U\{0, 1\}`, :math:`u \sim U(0, 1)`,
    :math:`b \sim U\{1, \dots, 10\}`, :math:`z \sim N(0, 1)` and :math:`d \sim U\{1, \dots, 5\}`.
    The trial vectors and a small noise term (computed from :math:`B` in the previous iteration) are mixed
    into :math:`B` through a randomly generated binary crossover map, components outside the search space
    are re-sampled uniformly inside the bounds (Eq. 15), and greedy selection decides which members of
    :math:`B` are replaced. :math:`B` is then written back into :math:`A`. One iteration costs
    :math:`N` function evaluations.

    Attributes:
        Name (List[str]): List of strings representing algorithm names.

    See Also:
        * :class:`niapy.algorithms.Algorithm`

    """

    Name = ['MultiPopulationDifferentialEvolution', 'MDE']

    @staticmethod
    def info():
        r"""Get basic information of algorithm.

        Returns:
            str: Basic information of algorithm.

        See Also:
            * :func:`niapy.algorithms.Algorithm.info`

        """
        return r"""Karkinli, A. E. (2023). Detection of object boundary from point cloud by using multi-population based differential evolution algorithm. Neural Computing and Applications, 35, 5193-5206. https://doi.org/10.1007/s00521-022-07969-w"""

    def __init__(self, population_size=90, *args, **kwargs):
        """Initialize MultiPopulationDifferentialEvolution.

        Args:
            population_size (Optional[int]): Size of the full population (:math:`3N` in the paper, at least 9).
                The sub-population evolved in each iteration has ``population_size // 3`` members.

        See Also:
            * :func:`niapy.algorithms.Algorithm.__init__`

        """
        self._check_population_size(population_size)
        super().__init__(population_size, *args, **kwargs)

    def set_parameters(self, population_size=90, initialization_function=default_numpy_init, individual_type=None,
                       *args, **kwargs):
        r"""Set the parameters of the algorithm.

        Args:
            population_size (Optional[int]): Size of the full population (:math:`3N` in the paper, at least 9).
                The sub-population evolved in each iteration has ``population_size // 3`` members.
            initialization_function (Optional[Callable[[int, Task, numpy.random.Generator, Dict[str, Any]], Tuple[numpy.ndarray, numpy.ndarray[float]]]]):
                Population initialization function.
            individual_type (Optional[Type[Individual]]): Individual type used in population, default is Numpy array.

        See Also:
            * :func:`niapy.algorithms.Algorithm.set_parameters`

        """
        self._check_population_size(population_size)
        super().set_parameters(population_size=population_size, initialization_function=initialization_function,
                               individual_type=individual_type, *args, **kwargs)

    @staticmethod
    def _check_population_size(population_size):
        r"""Check that the population is large enough for the variation operator.

        Args:
            population_size (int): Size of the full population.

        Raises:
            ValueError: If ``population_size`` is smaller than 9 (sub-population smaller than 3).

        """
        if population_size < 9:
            raise ValueError('population_size must be at least 9 (sub-population of at least 3 individuals).')

    def scale_factor(self):
        r"""Generate the parameter-free scale factor of one trial vector.

        Returns:
            float: :math:`|a - u^{b}| \cdot z^{d}`, :math:`a \sim U\{0, 1\}`, :math:`u \sim U(0, 1)`,
            :math:`b \sim U\{1, \dots, 10\}`, :math:`z \sim N(0, 1)`, :math:`d \sim U\{1, \dots, 5\}`.

        """
        return abs(self.integers(0, 2) - self.random() ** self.integers(1, 11)) * self.standard_normal() ** self.integers(1, 6)

    def trial_vectors(self, sub_population, best_x, noise):
        r"""Generate trial vectors from the sub-population.

        Args:
            sub_population (numpy.ndarray): Sub-population :math:`B` with shape ``(N, dimension)``.
            best_x (numpy.ndarray): Current global best solution.
            noise (numpy.ndarray): Noise matrix from the previous iteration with shape ``(N, dimension)``.

        Returns:
            numpy.ndarray: Trial vectors with shape ``(N, dimension)`` (not yet repaired).

        """
        n, dimension = sub_population.shape
        temp = np.empty_like(sub_population)
        for i in range(n):
            scale = self.scale_factor()
            while True:
                r1, r2 = self.rng.choice(n, 2, replace=False)
                if r1 != i and r2 != i:
                    break
            dx = np.where(self.random(dimension) < 0.5, sub_population[r1], best_x)
            dy = np.where(self.random(dimension) < 0.5, sub_population[i], sub_population[r2])
            temp[i] = sub_population[r2] + scale * (dx - dy)
        # parameter-free crossover map
        c = 1 if self.random() ** self.integers(1, 6) < 0.5 else dimension
        cross_map = np.abs(self.integers(0, 2, (n, c)) - self.random((n, dimension)) ** self.integers(1, 6, (n, c))) < 0.5
        return sub_population + cross_map * (temp + noise - sub_population)

    def noise(self, sub_population):
        r"""Generate the noise term used in the next iteration (Eq. 13).

        Args:
            sub_population (numpy.ndarray): Updated sub-population :math:`B` with shape ``(N, dimension)``.

        Returns:
            numpy.ndarray: :math:`B \cdot 10^{\phi} \cdot (u - 0.5)`, :math:`\phi \sim U\{-12, \dots, -9\}`
            per component and :math:`u \sim U(0, 1)` per individual.

        """
        n, dimension = sub_population.shape
        return sub_population * 10.0 ** self.integers(-12, -8, (n, dimension)) * (self.random((n, 1)) - 0.5)

    def repair(self, x, task):
        r"""Re-sample components outside the search space uniformly inside the bounds (Eq. 15).

        Args:
            x (numpy.ndarray): Solutions to repair.
            task (Task): Optimization task.

        Returns:
            numpy.ndarray: Repaired solutions.

        """
        out = (x < task.lower) | (x > task.upper)
        if out.any():
            x = np.where(out, task.lower + self.random(x.shape) * task.range, x)
        return x

    def init_population(self, task):
        r"""Initialize the starting population.

        Args:
            task (Task): Optimization task.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray, Dict[str, Any]]:
                1. New population.
                2. New population fitness/function values.
                3. Additional arguments:
                    * noise (numpy.ndarray): Noise matrix of the sub-population (zeros at the start).

        See Also:
            * :func:`niapy.algorithms.Algorithm.init_population`

        """
        population, fitness, d = super().init_population(task)
        d.update({'noise': np.zeros((len(population) // 3, task.dimension))})
        return population, fitness, d

    def run_iteration(self, task, population, population_fitness, best_x, best_fitness, **params):
        r"""Core function of Multi-population based Differential Evolution algorithm.

        Args:
            task (Task): Optimization task.
            population (numpy.ndarray): Current (full) population.
            population_fitness (numpy.ndarray): Current population fitness/function values.
            best_x (numpy.ndarray): Current best individual.
            best_fitness (float): Current best individual function/fitness value.
            **params (Dict[str, Any]): Additional arguments.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, float, Dict[str, Any]]:
                1. New population.
                2. New population fitness/function values.
                3. New global best solution.
                4. New global best solutions fitness/objective value.
                5. Additional arguments:
                    * noise (numpy.ndarray): Noise matrix for the next iteration.

        """
        noise = params.pop('noise')
        n = len(population) // 3
        indices = self.rng.permutation(len(population))[:n]
        sub_population, sub_fitness = population[indices], population_fitness[indices]

        trials = self.repair(self.trial_vectors(sub_population, best_x, noise), task)
        trials_fitness = np.asarray([task.eval(t) for t in trials])

        improved = trials_fitness < sub_fitness
        sub_population[improved], sub_fitness[improved] = trials[improved], trials_fitness[improved]
        population[indices], population_fitness[indices] = sub_population, sub_fitness

        best_index = np.argmin(population_fitness)
        best_x, best_fitness = population[best_index].copy(), population_fitness[best_index]
        return population, population_fitness, best_x, best_fitness, {'noise': self.noise(sub_population)}
