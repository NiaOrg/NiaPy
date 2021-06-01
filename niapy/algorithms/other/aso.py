# encoding=utf8
import logging

import numpy as np

from niapy.algorithms.algorithm import Algorithm
from niapy.util import full_array, euclidean

logging.basicConfig()
logger = logging.getLogger('niapy.algorithms.other')
logger.setLevel('INFO')

__all__ = ['AnarchicSocietyOptimization', 'elitism', 'sequential', 'crossover']


def elitism(x, xpb, xb, xr, mp_c, mp_s, mp_p, mutation_rate, crossover_probability, task, rng):
    r"""Select the best of all three strategies.

    Args:
        x (numpy.ndarray): individual position.
        xpb (numpy.ndarray): individuals best position.
        xb (numpy.ndarray): current best position.
        xr (numpy.ndarray): random individual.
        mp_c (float): Fickleness index value.
        mp_s (float): External irregularity index value.
        mp_p (float): Internal irregularity index value.
        mutation_rate (float): scale factor.
        crossover_probability (float): crossover factor.
        task (Task): optimization task.
        rng (numpy.random.Generator): random number generator.

    Returns:
        Tuple[numpy.ndarray, float]:
            1. New position of individual
            2. New positions fitness/function value

    """
    xn = [task.repair(mp_current(x, mutation_rate, crossover_probability, mp_c, rng), rng=rng),
          task.repair(mp_society(x, xr, xb, crossover_probability, mp_s, rng), rng=rng),
          task.repair(mp_past(x, xpb, crossover_probability, mp_p, rng), rng=rng)]
    xn_f = np.apply_along_axis(task.eval, 1, xn)
    ib = np.argmin(xn_f)
    return xn[ib], xn_f[ib]


def sequential(x, xpb, xb, xr, mp_c, mp_s, mp_p, mutation_rate, crossover_probability, task, rng):
    r"""Sequentially combines all three strategies.

    Args:
        x (numpy.ndarray): individual position.
        xpb (numpy.ndarray): individuals best position.
        xb (numpy.ndarray): current best position.
        xr (numpy.ndarray): random individual.
        mp_c (float): Fickleness index value.
        mp_s (float): External irregularity index value.
        mp_p (float): Internal irregularity index value.
        mutation_rate (float): scale factor.
        crossover_probability (float): crossover factor.
        task (Task): optimization task.
        rng (numpy.random.Generator): random number generator.

    Returns:
        tuple[numpy.ndarray, float]:
            1. new position
            2. new positions function/fitness value

    """
    xn = task.repair(mp_society(
        mp_past(mp_current(x, mutation_rate, crossover_probability, mp_c, rng), xpb, crossover_probability, mp_p, rng),
        xr,
        xb, crossover_probability, mp_s, rng), rng=rng)
    return xn, task.eval(xn)


def crossover(x, xpb, xb, xr, mp_c, mp_s, mp_p, mutation_rate, crossover_probability, task, rng):
    r"""Create a crossover over all three strategies.

    Args:
        x (numpy.ndarray): individual position.
        xpb (numpy.ndarray): individuals best position.
        xb (numpy.ndarray): current best position.
        xr (numpy.ndarray): random individual.
        mp_c (float): Fickleness index value.
        mp_s (float): External irregularity index value.
        mp_p (float): Internal irregularity index value.
        mutation_rate (float): scale factor.
        crossover_probability (float): crossover factor.
        task (Task): optimization task.
        rng (numpy.random.Generator): random number generator.

    Returns:
        Tuple[numpy.ndarray, float]:
            1. new position
            2. new positions function/fitness value.

    """
    xns = [task.repair(mp_current(x, mutation_rate, crossover_probability, mp_c, rng), rng=rng),
           task.repair(mp_society(x, xr, xb, crossover_probability, mp_s, rng), rng=rng),
           task.repair(mp_past(x, xpb, crossover_probability, mp_p, rng), rng=rng)]
    index = rng.integers(len(xns))
    x = np.asarray([xns[index][i] if rng.random() < crossover_probability else x[i] for i in range(len(x))])
    return x, task.eval(x)


def mp_current(x, mutation_rate, crossover_rate, mp, rng):
    r"""Get bew position based on fickleness.

    Args:
        x (numpy.ndarray): Current individuals position.
        mutation_rate (float): Scale factor.
        crossover_rate (float): Crossover probability.
        mp (float): Fickleness index value
        rng (numpy.random.Generator): Random number generator

    Returns:
        numpy.ndarray: New position

    """
    if mp < 0.5:
        b = np.sort(rng.choice(len(x), 2, replace=False))
        x[b[0]:b[1]] = x[b[0]:b[1]] + mutation_rate * rng.normal(0, 1, b[1] - b[0])
        return x
    return np.asarray(
        [x[i] + mutation_rate * rng.normal(0, 1) if rng.random() < crossover_rate else x[i] for i in range(len(x))])


def mp_society(x, xr, xb, crossover_rate, mp, rng):
    r"""Get new position based on external irregularity.

    Args:
        x (numpy.ndarray): Current individuals position.
        xr (numpy.ndarray): Random individuals position.
        xb (numpy.ndarray): Global best individuals position.
        crossover_rate (float): Crossover probability.
        mp (float): External irregularity index.
        rng (numpy.random.Generator): Random number generator.

    Returns:
        numpy.ndarray: New position.

    """
    if mp < 0.25:
        b = np.sort(rng.choice(len(x), 2, replace=False))
        x[b[0]:b[1]] = xb[b[0]:b[1]]
        return x
    elif mp < 0.5:
        return np.asarray([xb[i] if rng.random() < crossover_rate else x[i] for i in range(len(x))])
    elif mp < 0.75:
        b = np.sort(rng.choice(len(x), 2, replace=False))
        x[b[0]:b[1]] = xr[b[0]:b[1]]
        return x
    return np.asarray([xr[i] if rng.random() < crossover_rate else x[i] for i in range(len(x))])


def mp_past(x, xpb, crossover_rate, mp, rng):
    r"""Get new position based on internal irregularity.

    Args:
        x (numpy.ndarray): Current individuals position.
        xpb (numpy.ndarray): Current individuals personal best position.
        crossover_rate (float): Crossover probability.
        mp (float): Internal irregularity index value.
        rng (numpy.random.Generator): Random number generator.

    Returns:
        numpy.ndarray: Current individuals new position.

    """
    if mp < 0.5:
        b = np.sort(rng.choice(len(x), 2, replace=False))
        x[b[0]:b[1]] = xpb[b[0]:b[1]]
        return x
    return np.asarray([xpb[i] if rng.random() < crossover_rate else x[i] for i in range(len(x))])


class AnarchicSocietyOptimization(Algorithm):
    r"""Implementation of Anarchic Society Optimization algorithm.

    Algorithm:
        Anarchic Society Optimization algorithm

    Date:
        2018

    Authors:
        Klemen Berkovič

    License:
        MIT

    Reference paper:
        Ahmadi-Javid, Amir. "Anarchic Society Optimization: A human-inspired method." Evolutionary Computation (CEC), 2011 IEEE Congress on. IEEE, 2011.

    Attributes:
        Name (list of str): List of stings representing name of algorithm.
        alpha (List[float]): Factor for fickleness index function :math:`\in [0, 1]`.
        gamma (List[float]): Factor for external irregularity index function :math:`\in [0, \infty)`.
        theta (List[float]): Factor for internal irregularity index function :math:`\in [0, \infty)`.
        d (Callable[[float, float], float]): function that takes two arguments that are function values and calculates the distance between them.
        dn (Callable[[numpy.ndarray, numpy.ndarray], float]): function that takes two arguments that are points in function landscape and calculates the distance between them.
        nl (float): Normalized range for neighborhood search :math:`\in (0, 1]`.
        F (float): Mutation parameter.
        CR (float): Crossover parameter :math:`\in [0, 1]`.
        Combination (Callable[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, float, float, float, float, float, float, Task, numpy.random.Generator]): Function for combining individuals to get new position/individual.

    See Also:
        * :class:`niapy.algorithms.Algorithm`

    """

    Name = ['AnarchicSocietyOptimization', 'ASO']

    @staticmethod
    def info():
        r"""Get basic information about the algorithm.

        Returns:
            str: Basic information.

        See Also:
            :func:`niapy.algorithms.algorithm.Algorithm.info`

        """
        return r"""Ahmadi-Javid, Amir. "Anarchic Society Optimization: A human-inspired method." Evolutionary Computation (CEC), 2011 IEEE Congress on. IEEE, 2011."""

    def __init__(self, population_size=43, alpha=(1, 0.83), gamma=(1.17, 0.56), theta=(0.932, 0.832), d=euclidean,
                 dn=euclidean, nl=1, mutation_rate=1.2, crossover_rate=0.25, combination=elitism, *args, **kwargs):
        r"""Initialize AnarchicSocietyOptimization.

        Args:
            population_size (Optional[int]): Population size.
            alpha (Optional[Tuple[float, ...]]): Factor for fickleness index function :math:`\in [0, 1]`.
            gamma (Optional[Tuple[float, ...]]): Factor for external irregularity index function :math:`\in [0, \infty)`.
            theta (Optional[List[float]]): Factor for internal irregularity index function :math:`\in [0, \infty)`.
            d (Optional[Callable[[float, float], float]]): function that takes two arguments that are function values and calculates the distance between them.
            dn (Optional[Callable[[numpy.ndarray, numpy.ndarray], float]]): function that takes two arguments that are points in function landscape and calculates the distance between them.
            nl (Optional[float]): Normalized range for neighborhood search :math:`\in (0, 1]`.
            mutation_rate (Optional[float]): Mutation parameter.
            crossover_rate (Optional[float]): Crossover parameter :math:`\in [0, 1]`.
            combination (Optional[Callable[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, float, float, float, float, float, float, Task, numpy.random.Generator]]): Function for combining individuals to get new position/individual.

        See Also:
            * :func:`niapy.algorithms.Algorithm.set_parameters`

        """
        super().__init__(population_size, *args, **kwargs)
        self.alpha = alpha
        self.gamma = gamma
        self.theta = theta
        self.d = d
        self.dn = dn
        self.nl = nl
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.Combination = combination

    def set_parameters(self, population_size=43, alpha=(1, 0.83), gamma=(1.17, 0.56), theta=(0.932, 0.832), d=euclidean,
                       dn=euclidean, nl=1, mutation_rate=1.2, crossover_rate=0.25, combination=elitism, **kwargs):
        r"""Set the parameters for the algorithm.

        Args:
            population_size (Optional[int]): Population size.
            alpha (Optional[Tuple[float, ...]]): Factor for fickleness index function :math:`\in [0, 1]`.
            gamma (Optional[Tuple[float, ...]]): Factor for external irregularity index function :math:`\in [0, \infty)`.
            theta (Optional[List[float]]): Factor for internal irregularity index function :math:`\in [0, \infty)`.
            d (Optional[Callable[[float, float], float]]): function that takes two arguments that are function values and calculates the distance between them.
            dn (Optional[Callable[[numpy.ndarray, numpy.ndarray], float]]): function that takes two arguments that are points in function landscape and calculates the distance between them.
            nl (Optional[float]): Normalized range for neighborhood search :math:`\in (0, 1]`.
            mutation_rate (Optional[float]): Mutation parameter.
            crossover_rate (Optional[float]): Crossover parameter :math:`\in [0, 1]`.
            combination (Optional[Callable[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, float, float, float, float, float, float, Task, numpy.random.Generator]]): Function for combining individuals to get new position/individual.

        See Also:
            * :func:`niapy.algorithms.Algorithm.set_parameters`
            * Combination methods:
                * :func:`niapy.algorithms.other.elitism`
                * :func:`niapy.algorithms.other.crossover`
                * :func:`niapy.algorithms.other.sequential`

        """
        super().set_parameters(population_size=population_size, **kwargs)
        self.alpha = alpha
        self.gamma = gamma
        self.theta = theta
        self.d = d
        self.dn = dn
        self.nl = nl
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.Combination = combination

    def init(self, _task):
        r"""Initialize dynamic parameters of algorithm.

        Args:
            _task (Task): Optimization task.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
                1. Array of `self.alpha` propagated values
                2. Array of `self.gamma` propagated values
                3. Array of `self.theta` propagated values

        """
        return full_array(self.alpha, self.population_size), full_array(self.gamma, self.population_size), full_array(
            self.theta, self.population_size)

    @staticmethod
    def fickleness_index(x_f, xpb_f, xb_f, alpha):
        r"""Get fickleness index.

        Args:
            x_f (float): Individuals fitness/function value.
            xpb_f (float): Individuals personal best fitness/function value.
            xb_f (float): Current best found individuals fitness/function value.
            alpha (float): Fickleness factor.

        Returns:
            float: Fickleness index.

        """
        return 1 - alpha * xb_f / x_f - (1 - alpha) * xpb_f / x_f

    def external_irregularity(self, x_f, xnb_f, gamma):
        r"""Get external irregularity index.

        Args:
            x_f (float): Individuals fitness/function value.
            xnb_f (float): Individuals new fitness/function value.
            gamma (float): External irregularity factor.

        Returns:
            float: External irregularity index.

        """
        return 1 - np.exp(-gamma * self.d(x_f, xnb_f))

    def irregularity_index(self, x_f, xpb_f, theta):
        r"""Get internal irregularity index.

        Args:
            x_f (float): Individuals fitness/function value.
            xpb_f (float): Individuals personal best fitness/function value.
            theta (float): Internal irregularity factor.

        Returns:
            float: Internal irregularity index

        """
        return 1 - np.exp(-theta * self.d(x_f, xpb_f))

    def get_best_neighbors(self, i, population, population_fitness, rs):
        r"""Get neighbors of individual.

        Measurement of distance for neighborhood is defined with `self.nl`.
        Function for calculating distances is define with `self.dn`.

        Args:
            i (int): Index of individual for hum we are looking for neighbours.
            population (numpy.ndarray): Current population.
            population_fitness (numpy.ndarray[float]): Current population fitness/function values.
            rs (numpy.ndarray[float]): distance between individuals.

        Returns:
            numpy.ndarray[int]: Indexes that represent individuals closest to `i`-th individual.

        """
        nn = np.asarray([self.dn(population[i], population[j]) / rs for j in range(len(population))])
        return np.argmin(population_fitness[np.where(nn <= self.nl)])

    @staticmethod
    def update_personal_best(population, population_fitness, personal_best, personal_best_fitness):
        r"""Update personal best solution of all individuals in population.

        Args:
            population (numpy.ndarray): Current population.
            population_fitness (numpy.ndarray[float]): Current population fitness/function values.
            personal_best (numpy.ndarray): Current population best positions.
            personal_best_fitness (numpy.ndarray[float]): Current populations best positions fitness/function values.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray[float], numpy.ndarray, float]:
                1. New personal best positions for current population.
                2. New personal best positions function/fitness values for current population.
                3. New best individual.
                4. New best individual fitness/function value.

        """
        ix_pb = np.where(population_fitness < personal_best_fitness)
        personal_best[ix_pb], personal_best_fitness[ix_pb] = population[ix_pb], population_fitness[ix_pb]
        return personal_best, personal_best_fitness

    def init_population(self, task):
        r"""Initialize first population and additional arguments.

        Args:
            task (Task): Optimization task

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray, dict]:
                1. Initialized population
                2. Initialized population fitness/function values
                3. Dict[str, Any]:
                    * x_best (numpy.ndarray): Initialized populations best positions.
                    * x_best_fitness (numpy.ndarray): Initialized populations best positions function/fitness values.
                    * alpha (numpy.ndarray):
                    * gamma (numpy.ndarray):
                    * theta (numpy.ndarray):
                    * rs (float): distance of search space.

        See Also:
            * :func:`niapy.algorithms.algorithm.Algorithm.init_population`
            * :func:`niapy.algorithms.other.aso.AnarchicSocietyOptimization.init`

        """
        population, population_fitness, d = Algorithm.init_population(self, task)
        alpha, gamma, theta = self.init(task)
        x_best, x_best_fitness = self.update_personal_best(population, task.optimization_type.value * population_fitness,
                                                           np.zeros((self.population_size, task.dimension)),
                                                           np.full(self.population_size, np.inf))
        d.update({'x_best': x_best, 'x_best_fitness': x_best_fitness, 'alpha': alpha, 'gamma': gamma, 'theta': theta,
                  'rs': self.d(task.upper, task.lower)})
        return population, population_fitness, d

    def run_iteration(self, task, population, population_fitness, best_x, best_fitness, **params):
        r"""Core function of AnarchicSocietyOptimization algorithm.

        Args:
            task (Task): Optimization task.
            population (numpy.ndarray): Current populations positions.
            population_fitness (numpy.ndarray): Current populations function/fitness values.
            best_x (numpy.ndarray): Current global best individuals position.
            best_fitness (float): Current global best individual function/fitness value.
            **params: Additional arguments.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, float, dict]:
                1. Initialized population
                2. Initialized population fitness/function values
                3. New global best solution
                4. New global best solutions fitness/objective value
                5. Dict[str, Union[float, int, numpy.ndarray]:
                    * x_best (numpy.ndarray): Initialized populations best positions.
                    * x_best_fitness (numpy.ndarray): Initialized populations best positions function/fitness values.
                    * alpha (numpy.ndarray):
                    * gamma (numpy.ndarray):
                    * theta (numpy.ndarray):
                    * rs (float): distance of search space.

        """
        x_best = params.pop('x_best')
        x_best_fitness = params.pop('x_best_fitness')
        alpha = params.pop('alpha')
        gamma = params.pop('gamma')
        theta = params.pop('theta')
        rs = params.pop('rs')

        x_in = [self.get_best_neighbors(i, population, population_fitness, rs) for i in range(len(population))]
        mp_c, mp_s, mp_p = np.asarray(
            [self.fickleness_index(population_fitness[i], x_best_fitness[i], best_fitness, alpha[i]) for i in
             range(len(population))]), np.asarray(
            [self.external_irregularity(population_fitness[i], population_fitness[x_in[i]], gamma[i]) for i in
             range(len(population))]), np.asarray(
            [self.irregularity_index(population_fitness[i], x_best_fitness[i], theta[i]) for i in range(len(population))])
        x_tmp = np.asarray([self.Combination(population[i], x_best[i], best_x,
                                             population[self.integers(len(population), skip=[i])], mp_c[i], mp_s[i],
                                             mp_p[i], self.mutation_rate, self.crossover_rate, task, self.rng) for i in range(len(population))],
                           dtype=object)
        population, population_fitness = np.asarray([x_tmp[i][0] for i in range(len(population))]), np.asarray(
            [x_tmp[i][1] for i in range(len(population))])
        x_best, x_best_fitness = self.update_personal_best(population, population_fitness, x_best, x_best_fitness)
        best_x, best_fitness = self.get_best(population, population_fitness, best_x, best_fitness)
        return population, population_fitness, best_x, best_fitness, {'x_best': x_best,
                                                                      'x_best_fitness': x_best_fitness,
                                                                      'alpha': alpha,
                                                                      'gamma': gamma,
                                                                      'theta': theta,
                                                                      'rs': rs}

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
