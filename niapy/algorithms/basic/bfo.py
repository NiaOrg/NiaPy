# encoding=utf8
import logging

import numpy as np

from niapy.algorithms.algorithm import Algorithm
from niapy.util.distances import euclidean

logging.basicConfig()
logger = logging.getLogger('niapy.algorithms.basic')
logger.setLevel('INFO')

__all__ = ['BacterialForagingOptimization']


class BacterialForagingOptimization(Algorithm):
    r"""Implementation of the Bacterial foraging optimization algorithm.

    Date:
        2021

    Author:
        Žiga Stupan

    License:
        MIT

    Reference paper:
        K. M. Passino, "Biomimicry of bacterial foraging for distributed optimization and control," in IEEE Control Systems Magazine, vol. 22, no. 3, pp. 52-67, June 2002, doi: 10.1109/MCS.2002.1004010.

    Attributes:
        Name (List[str]): list of strings representing algorithm names.
        population_size (Optional[int]): Number of individuals in population :math:`\in [1, \infty]`.
        n_chemotactic (Optional[int]): Number of chemotactic steps.
        n_swim (Optional[int]): Number of swim steps.
        n_reproduction (Optional[int]): Number of reproduction steps.
        n_elimination (Optional[int]): Number of elimination and dispersal steps.
        prob_elimination (Optional[float]): Probability of a bacterium being eliminated and a new one being created at a random location in the search space.
        step_size (Optional[float]): Size of a chemotactic step.
        d_attract (Optional[float]): Depth of the attractant released by the cell (a quantification of how much attractant is released).
        w_attract (Optional[float]): Width of the attractant signal (a quantification of the diffusion rate of the chemical).
        h_repel (Optional[float]): Height of the repellent effect (magnitude of its effect).
        w_repel (Optional[float]): Width of the repellent.

    See Also:
        * :class:`niapy.algorithms.Algorithm`

    """

    Name = ['BacterialForagingOptimization', 'BFO', 'BFOA']

    @staticmethod
    def info():
        r"""Get algorithm information.

        Returns:
            str: Bit item.

        See Also:
            * :func:`niapy.algorithms.Algorithm.info`

        """
        return r"""K. M. Passino, "Biomimicry of bacterial foraging for distributed optimization and control," in IEEE Control Systems Magazine, vol. 22, no. 3, pp. 52-67, June 2002, doi: 10.1109/MCS.2002.1004010."""

    def __init__(self, population_size=50, n_chemotactic=100, n_swim=4, n_reproduction=4, n_elimination=2,
                 prob_elimination=0.25, step_size=0.1, swarming=True, d_attract=0.1, w_attract=0.2, h_repel=0.1,
                 w_repel=10.0, *args, **kwargs):
        r"""Initialize algorithm.

        Args:
            population_size (Optional[int]): Number of individuals in population :math:`\in [1, \infty]`.
            n_chemotactic (Optional[int]): Number of chemotactic steps.
            n_swim (Optional[int]): Number of swim steps.
            n_reproduction (Optional[int]): Number of reproduction steps.
            n_elimination (Optional[int]): Number of elimination and dispersal steps.
            prob_elimination (Optional[float]): Probability of a bacterium being eliminated and a new one being created at a random location in the search space.
            step_size (Optional[float]): Size of a chemotactic step.
            swarming (Optional[bool]): If `True` use swarming.
            d_attract (Optional[float]): Depth of the attractant released by the cell (a quantification of how much attractant is released).
            w_attract (Optional[float]): Width of the attractant signal (a quantification of the diffusion rate of the chemical).
            h_repel (Optional[float]): Height of the repellent effect (magnitude of its effect).
            w_repel (Optional[float]): Width of the repellent.

        See Also:
            * :func:`niapy.algorithms.Algorithm.__init__`

        """
        super().__init__(population_size, *args, **kwargs)
        self.n_chemotactic = n_chemotactic
        self.n_swim = n_swim
        self.n_reproduction = n_reproduction
        self.n_elimination = n_elimination
        self.prob_elimination = prob_elimination
        self.step_size = step_size
        self.swarming = swarming
        self.d_attract = d_attract
        self.w_attract = w_attract
        self.h_repel = h_repel
        self.w_repel = w_repel

        self.i = 0  # elimination and dispersal step counter
        self.j = 0  # reproduction step counter
        self.k = 0  # chemotaxis step counter

    def set_parameters(self, population_size=50, n_chemotactic=100, n_swim=4, n_reproduction=4, n_elimination=2,
                       prob_elimination=0.25, step_size=0.1, swarming=True, d_attract=0.1, w_attract=0.2, h_repel=0.1,
                       w_repel=10.0, **kwargs):
        r"""Set the parameters/arguments of the algorithm.

        Args:
            population_size (Optional[int]): Number of individuals in population :math:`\in [1, \infty]`.
            n_chemotactic (Optional[int]): Number of chemotactic steps.
            n_swim (Optional[int]): Number of swim steps.
            n_reproduction (Optional[int]): Number of reproduction steps.
            n_elimination (Optional[int]): Number of elimination and dispersal steps.
            prob_elimination (Optional[float]): Probability of a bacterium being eliminated and a new one being created at a random location in the search space.
            step_size (Optional[float]): Size of a chemotactic step.
            swarming (Optional[bool]): If `True` use swarming.
            d_attract (Optional[float]): Depth of the attractant released by the cell (a quantification of how much attractant is released).
            w_attract (Optional[float]): Width of the attractant signal (a quantification of the diffusion rate of the chemical).
            h_repel (Optional[float]): Height of the repellent effect (magnitude of its effect).
            w_repel (Optional[float]): Width of the repellent.

        """
        super().set_parameters(population_size=population_size, **kwargs)
        self.n_chemotactic = n_chemotactic
        self.n_swim = n_swim
        self.n_reproduction = n_reproduction
        self.n_elimination = n_elimination
        self.prob_elimination = prob_elimination
        self.step_size = step_size
        self.d_attract = d_attract
        self.w_attract = w_attract
        self.h_repel = h_repel
        self.w_repel = w_repel

    def get_parameters(self):
        r"""Get parameters of the algorithm.

        Returns:
            Dict[str, Any]: Algorithm parameters.

        """
        params = super().get_parameters()
        params.update({
            'n_chemotactic': self.n_chemotactic,
            'n_swim': self.n_swim,
            'n_reproduction': self.n_reproduction,
            'n_elimination': self.n_elimination,
            'prob_elimination': self.prob_elimination,
            'step_size': self.step_size,
            'd_attract': self.d_attract,
            'w_attract': self.w_attract,
            'h_repel': self.h_repel,
            'w_repel': self.w_repel
        })
        return params

    def init_population(self, task):
        r"""Initialize the starting population.

        Args:
            task (Task): Optimization task

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray, Dict[str, Any]]:
                1. New population.
                2. New population fitness/function values.
                3. Additional arguments:
                    * cost (numpy.ndarray): Costs of cells i. e. Fitness + cell interaction
                    * health (numpy.ndarray): Cell health i. e. The accumulation of costs over all chemotactic steps.

        See Also:
            * :func:`niapy.algorithms.Algorithm.init_population`

        """
        pop, fpop, d = super().init_population(task)
        d.update({
            'cost': np.zeros(self.population_size, dtype=np.float64),
            'health': np.zeros(self.population_size, dtype=np.float64)
        })
        return pop, fpop, d

    def interaction(self, cell, population):
        r"""Compute cell to cell interaction J_cc.

        Args:
            cell (numpy.ndarray): Cell to compute interaction for.
            population (numpy.ndarray): Population

        Returns:
            float: Cell to cell interaction J_cc

        """
        if not self.swarming:
            return 0.0
        distances = euclidean(cell, population)
        attract = np.sum(-self.d_attract * np.exp(-self.w_attract * distances))
        repel = np.sum(self.h_repel * np.exp(-self.w_repel * distances))
        return attract + repel

    def random_direction(self, dimension):
        r"""Generate a random direction vector.

        Args:
            dimension (int): Problem dimension

        Returns:
            numpy.ndarray: Normalised random direction vector

        """
        delta = self.uniform(-1.0, 1.0, dimension)
        return delta / np.linalg.norm(delta)

    def run_iteration(self, task, population, population_fitness, best_x, best_fitness, **params):
        r"""Core function of Bacterial Foraging Optimization algorithm.

        Args:
            task (Task): Optimization task.
            population (numpy.ndarray): Current population.
            population_fitness (numpy.ndarray): Current populations fitness/function values.
            best_x (numpy.ndarray): Global best individual.
            best_fitness (float): Global best individuals function/fitness value.
            **params (Dict[str, Any]): Additional arguments.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, float, Dict[str, Any]]:
                1. New population.
                2. New populations function/fitness values.
                3. New global best solution,
                4. New global best solutions fitness/objective value.
                5. Additional arguments:
                    * cost (numpy.ndarray): Costs of cells i. e. Fitness + cell interaction
                    * health (numpy.ndarray): Cell health i. e. The accumulation of costs over all chemotactic steps.

        """
        cost = params.pop('cost')
        health = params.pop('health')

        # Chemotaxis
        for i in range(len(population)):
            cost[i] = population_fitness[i] + self.interaction(population[i], population)
            j_last = cost[i]
            step_direction = self.random_direction(task.dimension)

            m = 0
            while True:
                population[i] = task.repair(population[i] + self.step_size * step_direction)
                population_fitness[i] = task.eval(population[i])

                if population_fitness[i] < best_fitness:
                    best_x = population[i].copy()
                    best_fitness = population_fitness[i]

                cost[i] = population_fitness[i] + self.interaction(population[i], population)
                health[i] += cost[i]

                if m >= self.n_swim or cost[i] >= j_last:
                    break
                m += 1

        self.k += 1

        if self.k >= self.n_chemotactic:
            self.k = 0
            self.j += 1

            # Reproduction
            sorted_indices = np.argsort(health)
            population = population[sorted_indices]
            population_fitness = population_fitness[sorted_indices]
            cost = cost[sorted_indices]

            population = np.tile(population[:self.population_size // 2], (2, 1))
            population_fitness = np.tile(population_fitness[:self.population_size // 2], 2)
            cost = np.tile(cost[:self.population_size // 2], 2)

            health = np.zeros(len(population), dtype=np.float64)

        if self.j >= self.n_reproduction:
            self.j = 0
            self.i += 1

            # Elimination and dispersal
            for i in range(len(population)):
                if self.random() < self.prob_elimination:
                    population[i] = task.lower + self.random(task.dimension) * task.range
                    population_fitness[i] = task.eval(population[i])
                    if population_fitness[i] < best_fitness:
                        best_x = population[i].copy()
                        best_fitness = population_fitness[i]

        return population, population_fitness, best_x, best_fitness, {'cost': cost, 'health': health}

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
