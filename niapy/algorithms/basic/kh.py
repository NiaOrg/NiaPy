# encoding=utf8
import logging

import numpy as np

from niapy.algorithms.algorithm import Algorithm
from niapy.util import full_array, euclidean

logging.basicConfig()
logger = logging.getLogger('niapy.algorithms.basic')
logger.setLevel('INFO')

__all__ = ['KrillHerd']


class KrillHerd(Algorithm):
    r"""Implementation of krill herd algorithm.

    Algorithm:
        Krill Herd Algorithm

    Date:
        2018

    Authors:
        Klemen Berkovič

    License:
        MIT

    Reference URL:
        http://www.sciencedirect.com/science/article/pii/S1007570412002171

    Reference paper:
        Amir Hossein Gandomi, Amir Hossein Alavi, Krill herd: A new bio-inspired optimization algorithm, Communications in Nonlinear Science and Numerical Simulation, Volume 17, Issue 12, 2012, Pages 4831-4845, ISSN 1007-5704, https://doi.org/10.1016/j.cnsns.2012.05.010.

    Attributes:
        Name (List[str]): List of strings representing algorithm names.
        population_size (int): Number of krill herds in population.
        N_max (float): Maximum induced speed.
        V_f (float): Foraging speed.
        D_max (float): Maximum diffusion speed.
        C_t (float): Constant :math:`\in [0, 2]`
        W_n (Union[int, float, numpy.ndarray]): Inertia weights of the motion induced from neighbors :math:`\in [0, 1]`.
        W_f (Union[int, float, numpy.ndarray]): Inertia weights of the motion induced from foraging :math`\in [0, 1]`.
        d_s (float): Maximum euclidean distance for neighbors.
        nn (int): Maximum neighbors for neighbors effect.
        epsilon (float): Small numbers for division.

    See Also:
        * :class:`niapy.algorithms.algorithm.Algorithm`

    """

    Name = ['KrillHerd', 'KH']

    @staticmethod
    def info():
        r"""Get basic information of algorithm.

        Returns:
            str: Basic information of algorithm.

        See Also:
            * :func:`niapy.algorithms.Algorithm.info`

        """
        return r"""Amir Hossein Gandomi, Amir Hossein Alavi, Krill herd: A new bio-inspired optimization algorithm, Communications in Nonlinear Science and Numerical Simulation, Volume 17, Issue 12, 2012, Pages 4831-4845, ISSN 1007-5704, https://doi.org/10.1016/j.cnsns.2012.05.010."""

    def __init__(self, population_size=50, n_max=0.01, foraging_speed=0.02, diffusion_speed=0.002, c_t=0.93,
                 w_neighbor=0.42, w_foraging=0.38, d_s=2.63, max_neighbors=5, crossover_rate=0.2, mutation_rate=0.05,
                 *args, **kwargs):
        r"""Initialize KrillHerd.

        Args:
            population_size (Optional[int]): Number of krill herds in population.
            n_max (Optional[float]): Maximum induced speed.
            foraging_speed (Optional[float]): Foraging speed.
            diffusion_speed (Optional[float]): Maximum diffusion speed.
            c_t (Optional[float]): Constant $\in [0, 2]$.
            w_neighbor (Optional[Union[int, float, numpy.ndarray]]): Inertia weights of the motion induced from neighbors :math:`\in [0, 1]`.
            w_foraging (Optional[Union[int, float, numpy.ndarray]]): Inertia weights of the motion induced from foraging :math:`\in [0, 1]`.
            d_s (Optional[float]): Maximum euclidean distance for neighbors.
            max_neighbors (Optional[int]): Maximum neighbors for neighbors effect.
            crossover_rate (Optional[float]): Crossover probability.
            mutation_rate (Optional[float]): Mutation probability.

        See Also:
            * :func:`niapy.algorithms.algorithm.Algorithm.__init__`

        """
        super().__init__(population_size, *args, **kwargs)
        self.N_max = n_max
        self.V_f = foraging_speed
        self.D_max = diffusion_speed
        self.C_t = c_t
        self.W_n = w_neighbor
        self.W_f = w_foraging
        self.d_s = d_s
        self.nn = max_neighbors
        self._Cr = crossover_rate
        self._Mu = mutation_rate
        self.epsilon = np.finfo(float).eps

    def set_parameters(self, population_size=50, n_max=0.01, foraging_speed=0.02, diffusion_speed=0.002, c_t=0.93,
                       w_neighbor=0.42, w_foraging=0.38, d_s=2.63, max_neighbors=5, crossover_rate=0.2,
                       mutation_rate=0.05, **kwargs):
        r"""Set the arguments of an algorithm.

        Args:
            population_size (Optional[int]): Number of krill herds in population.
            n_max (Optional[float]): Maximum induced speed.
            foraging_speed (Optional[float]): Foraging speed.
            diffusion_speed (Optional[float]): Maximum diffusion speed.
            c_t (Optional[float]): Constant $\in [0, 2]$.
            w_neighbor (Optional[Union[int, float, numpy.ndarray]]): Inertia weights of the motion induced from neighbors :math:`\in [0, 1]`.
            w_foraging (Optional[Union[int, float, numpy.ndarray]]): Inertia weights of the motion induced from foraging :math:`\in [0, 1]`.
            d_s (Optional[float]): Maximum euclidean distance for neighbors.
            max_neighbors (Optional[int]): Maximum neighbors for neighbors effect.
            crossover_rate (Optional[float]): Crossover probability.
            mutation_rate (Optional[float]): Mutation probability.

        See Also:
            * :func:`niapy.algorithms.algorithm.Algorithm.set_parameters`

        """
        super().set_parameters(population_size=population_size, **kwargs)
        self.N_max = n_max
        self.V_f = foraging_speed
        self.D_max = diffusion_speed
        self.C_t = c_t
        self.W_n = w_neighbor
        self.W_f = w_foraging
        self.d_s = d_s
        self.nn = max_neighbors
        self._Cr = crossover_rate
        self._Mu = mutation_rate
        self.epsilon = np.finfo(float).eps

    def get_parameters(self):
        r"""Get parameter values for the algorithm.

        Returns:
            Dict[str, Any]: Algorithm parameters.

        """
        d = Algorithm.get_parameters(self)
        d.update({
            'N_max': self.N_max,
            'V_f': self.V_f,
            'D_max': self.D_max,
            'C_t': self.C_t,
            'W_n': self.W_n,
            'W_f': self.W_f,
            'd_s': self.d_s,
            'nn': self.nn,
        })
        return d

    def init_weights(self, task):
        r"""Initialize weights.

        Args:
            task (Task): Optimization task.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray]:
                1. Weights for neighborhood.
                2. Weights for foraging.

        """
        return full_array(self.W_n, task.dimension), full_array(self.W_f, task.dimension)

    def sense_range(self, ki, population):
        r"""Calculate sense range for selected individual.

        Args:
            ki (int): Selected individual.
            population (numpy.ndarray): Krill heard population.

        Returns:
            float: Sense range for krill.

        """
        return np.sum([euclidean(population[ki], population[i]) for i in range(self.population_size)]) / (self.nn * self.population_size)

    def get_neighbours(self, i, ids, population):
        r"""Get neighbours.

        Args:
            i (int): Individual looking for neighbours.
            ids (float): Maximal distance for being a neighbour.
            population (numpy.ndarray): Current population.

        Returns:
            numpy.ndarray: Neighbours of krill heard.

        """
        neighbors = list()
        for j in range(self.population_size):
            if j != i and ids > euclidean(population[i], population[j]):
                neighbors.append(j)
        if not neighbors:
            neighbors.append(self.integers(self.population_size))
        return np.asarray(neighbors)

    def get_x(self, x, y):
        r"""Get x values.

        Args:
            x (numpy.ndarray): First krill/individual.
            y (numpy.ndarray): Second krill/individual.

        Returns:
            numpy.ndarray: --

        """
        return ((y - x) + self.epsilon) / (euclidean(y, x) + self.epsilon)

    def get_k(self, x, y, b, w):
        r"""Get k values.

        Args:
            x (float): First krill/individual.
            y (float): Second krill/individual.
            b (float): Best krill/individual.
            w (float): Worst krill/individual.

        Returns:
            numpy.ndarray: K.

        """
        return ((x - y) + self.epsilon) / ((w - b) + self.epsilon)

    def induce_neighbors_motion(self, i, n, weights, population, population_fitness, best_index, worst_index, task):
        r"""Induced neighbours motion operator.

        Args:
            i (int): Index of individual being applied with operator.
            n:
            weights (numpy.ndarray[float]): Weights for this operator.
            population (numpy.ndarray): Current heard/population.
            population_fitness (numpy.ndarray[float]): Current populations/heard function/fitness values.
            best_index (numpy.ndarray): Current best krill in heard/population.
            worst_index (numpy.ndarray): Current worst krill in heard/population.
            task (Task): Optimization task.

        Returns:
            numpy.ndarray: Moved krill.

        """
        neighbor_i = self.get_neighbours(i, self.sense_range(i, population), population)
        neighbor_x, neighbor_f, f_b, f_w = population[neighbor_i], population_fitness[neighbor_i], population_fitness[best_index], population_fitness[worst_index]
        alpha_l = np.sum(
            np.asarray([self.get_k(population_fitness[i], j, f_b, f_w) for j in neighbor_f]) * np.asarray([self.get_x(population[i], j) for j in neighbor_x]).T)
        alpha_t = 2 * (1 + self.random() * (task.iters + 1) / task.max_iters)
        return self.N_max * (alpha_l + alpha_t) + weights * n

    def induce_foraging_motion(self, i, x, x_f, f, weights, population, population_fitness, best_index, worst_index, task):
        r"""Induced foraging motion operator.

        Args:
            i (int): Index of current krill being operated.
            x (numpy.ndarray): Position of food.
            x_f (float): Fitness/function values of food.
            f:
            weights (numpy.ndarray[float]): Weights for this operator.
            population (numpy.ndarray):  Current population/heard.
            population_fitness (numpy.ndarray[float]): Current heard/populations function/fitness values.
            best_index (numpy.ndarray): Index of current best krill in heard.
            worst_index (numpy.ndarray): Index of current worst krill in heard.
            task (Task): Optimization task.

        Returns:
            numpy.ndarray: Moved krill.

        """
        beta_f = 2 * (1 - (task.iters + 1) / task.max_iters) * self.get_k(population_fitness[i], x_f, population_fitness[best_index], population_fitness[worst_index]) * self.get_x(
            population[i], x) if population_fitness[best_index] < population_fitness[i] else 0
        beta_b = self.get_k(population_fitness[i], population_fitness[best_index], population_fitness[best_index], population_fitness[worst_index]) * self.get_x(population[i], population[best_index])
        return self.V_f * (beta_f + beta_b) + weights * f

    def induce_physical_diffusion(self, task):
        r"""Induced physical diffusion operator.

        Args:
            task (Task): Optimization task.

        Returns:
            numpy.ndarray:

        """
        return self.D_max * (1 - (task.iters + 1) / task.max_iters) * self.uniform(-1, 1, task.dimension)

    def delta_t(self, task):
        r"""Get new delta for all dimensions.

        Args:
            task (Task): Optimization task.

        Returns:
            numpy.ndarray: --

        """
        return self.C_t * np.sum(task.range)

    def crossover(self, x, xo, crossover_rate):
        r"""Crossover operator.

        Args:
            x (numpy.ndarray): Krill/individual being applied with operator.
            xo (numpy.ndarray): Krill/individual being used in conjunction within operator.
            crossover_rate (float): Crossover probability.

        Returns:
            numpy.ndarray: New krill/individual.

        """
        return [xo[i] if self.random() < crossover_rate else x[i] for i in range(len(x))]

    def mutate(self, x, x_b, mutation_rate):
        r"""Mutate operator.

        Args:
            x (numpy.ndarray): Individual being mutated.
            x_b (numpy.ndarray): Global best individual.
            mutation_rate (float): Probability of mutations.

        Returns:
            numpy.ndarray: Mutated krill.

        """
        return [x[i] if self.random() < mutation_rate else (x_b[i] + self.random()) for i in range(len(x))]

    def get_food_location(self, population, population_fitness, task):
        r"""Get food location for krill heard.

        Args:
            population (numpy.ndarray): Current heard/population.
            population_fitness (numpy.ndarray[float]): Current heard/populations function/fitness values.
            task (Task): Optimization task.

        Returns:
            Tuple[numpy.ndarray, float]:
                1. Location of food.
                2. Foods function/fitness value.

        """
        x_food = task.repair(np.asarray([np.sum(population[:, i] / population_fitness) for i in range(task.dimension)]) / np.sum(1 / population_fitness),
                             rng=self.rng)
        x_food_f = task.eval(x_food)
        return x_food, x_food_f

    def mutation_rate(self, xf, yf, xf_best, xf_worst):
        r"""Get mutation probability.

        Args:
            xf (float):
            yf (float):
            xf_best (float):
            xf_worst (float):

        Returns:
            float: New mutation probability.

        """
        return self._Mu / (self.get_k(xf, yf, xf_best, xf_worst) + 1e-31)

    def crossover_rate(self, xf, yf, xf_best, xf_worst):
        r"""Get crossover probability.

        Args:
            xf (float):
            yf (float):
            xf_best (float):
            xf_worst (float):

        Returns:
            float: New crossover probability.

        """
        return self._Cr * self.get_k(xf, yf, xf_best, xf_worst)

    def init_population(self, task):
        r"""Initialize stating population.

        Args:
            task (Task): Optimization task.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray, Dict[str, Any]]:
                1. Initialized population.
                2. Initialized populations function/fitness values.
                3. Additional arguments:
                    * w_neighbor (numpy.ndarray): Weights neighborhood.
                    * w_foraging (numpy.ndarray): Weights foraging.
                    * induced_speed (numpy.ndarray): Induced speed.
                    * foraging_speed (numpy.ndarray): Foraging speed.

        See Also:
            * :func:`niapy.algorithms.algorithm.Algorithm.init_population`

        """
        krill_herd, krill_herd_fitness, d = Algorithm.init_population(self, task)
        w_neighbor, w_foraging = self.init_weights(task)
        induced_speed, foraging_speed = np.zeros(self.population_size), np.zeros(self.population_size)
        d.update({'w_neighbor': w_neighbor, 'w_foraging': w_foraging, 'induced_speed': induced_speed, 'foraging_speed': foraging_speed})
        return krill_herd, krill_herd_fitness, d

    def run_iteration(self, task, population, population_fitness, best_x, best_fitness, **params):
        r"""Core function of KrillHerd algorithm.

        Args:
            task (Task): Optimization task.
            population (numpy.ndarray): Current heard/population.
            population_fitness (numpy.ndarray[float]): Current heard/populations function/fitness values.
            best_x (numpy.ndarray): Global best individual.
            best_fitness (float): Global best individuals function fitness values.
            **params (Dict[str, Any]): Additional arguments.

        Returns:
            Tuple [numpy.ndarray, numpy.ndarray, numpy.ndarray, float Dict[str, Any]]:
                1. New herd/population
                2. New herd/populations function/fitness values.
                3. New global best solution.
                4. New global best solutions fitness/objective value.
                5. Additional arguments:
                    * w_neighbor (numpy.ndarray): --
                    * w_foraging (numpy.ndarray): --
                    * induced_speed (numpy.ndarray): --
                    * foraging_speed (numpy.ndarray): --

        """
        w_neighbor = params.pop('w_neighbor')
        w_foraging = params.pop('w_foraging')
        induced_speed = params.pop('induced_speed')
        foraging_speed = params.pop('foraging_speed')

        ikh_b, ikh_w = np.argmin(population_fitness), np.argmax(population_fitness)
        x_food, x_food_f = self.get_food_location(population, population_fitness, task)
        if x_food_f < best_fitness:
            best_x, best_fitness = x_food, x_food_f  # noqa: F841
        induced_speed = np.asarray([self.induce_neighbors_motion(i, induced_speed[i], w_neighbor, population, population_fitness, ikh_b, ikh_w, task) for i in range(self.population_size)])
        foraging_speed = np.asarray([self.induce_foraging_motion(i, x_food, x_food_f, foraging_speed[i], w_foraging, population, population_fitness, ikh_b, ikh_w, task) for i in range(self.population_size)])
        diffusion = np.asarray([self.induce_physical_diffusion(task) for _ in range(self.population_size)])
        new_herd = population + (self.delta_t(task) * (induced_speed + foraging_speed + diffusion))
        crossover_rates = np.asarray([self.crossover_rate(population_fitness[i], population_fitness[ikh_b], population_fitness[ikh_b], population_fitness[ikh_w]) for i in range(self.population_size)])
        new_herd = np.asarray([self.crossover(new_herd[i], population[i], crossover_rates[i]) for i in range(self.population_size)])
        mutation_rates = np.asarray([self.mutation_rate(population_fitness[i], population_fitness[ikh_b], population_fitness[ikh_b], population_fitness[ikh_w]) for i in range(self.population_size)])
        new_herd = np.asarray([self.mutate(new_herd[i], population[ikh_b], mutation_rates[i]) for i in range(self.population_size)])
        population = np.apply_along_axis(task.repair, 1, new_herd, rng=self.rng)
        population_fitness = np.apply_along_axis(task.eval, 1, population)
        best_x, best_fitness = self.get_best(population, population_fitness, best_x, best_fitness)
        return population, population_fitness, best_x, best_fitness, {'w_neighbor': w_neighbor, 'w_foraging': w_foraging, 'induced_speed': induced_speed, 'foraging_speed': foraging_speed}

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
