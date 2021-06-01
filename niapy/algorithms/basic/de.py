# encoding=utf8
import logging
import math

import numpy as np

from niapy.algorithms.algorithm import Algorithm, Individual, default_individual_init
from niapy.util.array import objects_to_array

__all__ = ['DifferentialEvolution', 'DynNpDifferentialEvolution', 'AgingNpDifferentialEvolution',
           'MultiStrategyDifferentialEvolution', 'DynNpMultiStrategyDifferentialEvolution', 'AgingIndividual',
           'cross_rand1', 'cross_rand2', 'cross_best2', 'cross_best1', 'cross_best2', 'cross_curr2rand1',
           'cross_curr2best1', 'multi_mutations', 'proportional', 'linear', 'bilinear']

logging.basicConfig()
logger = logging.getLogger('niapy.algorithms.basic')
logger.setLevel('INFO')


def cross_rand1(pop, ic, f, cr, rng, **_kwargs):
    r"""Mutation strategy with crossover.

    Mutation strategy uses three different random individuals from population to perform mutation.

    Mutation:
        Name: DE/rand/1

        :math:`\mathbf{x}_{r_1, G} + differential_weight \cdot (\mathbf{x}_{r_2, G} - \mathbf{x}_{r_3, G}`
        where :math:`r_1, r_2, r_3` are random indexes representing current population individuals.

    Crossover:
        Name: Binomial crossover

        :math:`\mathbf{x}_{i, G+1} = \begin{cases} \mathbf{u}_{i, G+1}, & \text{if $f(\mathbf{u}_{i, G+1}) \leq f(\mathbf{x}_{i, G})$}, \\ \mathbf{x}_{i, G}, & \text{otherwise}. \end{cases}`

    Args:
        pop (numpy.ndarray[Individual]): Current population.
        ic (int): Index of individual being mutated.
        f (float): Scale factor.
        cr (float): Crossover probability.
        rng (numpy.random.Generator): Random generator.

    Returns:
        numpy.ndarray: Mutated and mixed individual.

    """
    j = rng.integers(len(pop[ic]))
    p = [1 / (len(pop) - 1.0) if i != ic else 0 for i in range(len(pop))] if len(pop) > 3 else None
    r = rng.choice(len(pop), 3, replace=not len(pop) >= 3, p=p)
    x = [pop[r[0]][i] + f * (pop[r[1]][i] - pop[r[2]][i]) if rng.random() < cr or i == j else pop[ic][i] for i in
         range(len(pop[ic]))]
    return np.asarray(x)


def cross_best1(pop, ic, f, cr, rng, x_b=None, **_kwargs):
    r"""Mutation strategy with crossover.

    Mutation strategy uses two different random individuals from population and global best individual.

    Mutation:
        Name: de/best/1

        :math:`\mathbf{v}_{i, G} = \mathbf{x}_{best, G} + differential_weight \cdot (\mathbf{x}_{r_1, G} - \mathbf{x}_{r_2, G})`
        where :math:`r_1, r_2` are random indexes representing current population individuals.

    Crossover:
        Name: Binomial crossover

        :math:`\mathbf{x}_{i, G+1} = \begin{cases} \mathbf{u}_{i, G+1}, & \text{if $f(\mathbf{u}_{i, G+1}) \leq f(\mathbf{x}_{i, G})$}, \\ \mathbf{x}_{i, G}, & \text{otherwise}. \end{cases}`

    args:
        pop (numpy.ndarray[Individual]): Current population.
        ic (int): Index of individual being mutated.
        x_b (Individual): Current global best individual.
        f (float): Scale factor.
        cr (float): Crossover probability.
        rng (numpy.random.Generator): Random generator.

    returns:
        numpy.ndarray: Mutated and mixed individual.

    """
    j = rng.integers(len(pop[ic]))
    p = [1 / (len(pop) - 1.0) if i != ic else 0 for i in range(len(pop))] if len(pop) > 2 else None
    r = rng.choice(len(pop), 2, replace=not len(pop) >= 2, p=p)
    x = [x_b[i] + f * (pop[r[0]][i] - pop[r[1]][i]) if rng.random() < cr or i == j else pop[ic][i] for i in
         range(len(pop[ic]))]
    return np.asarray(x)


def cross_rand2(pop, ic, f, cr, rng, **_kwargs):
    r"""Mutation strategy with crossover.

    Mutation strategy uses five different random individuals from population.

    Mutation:
        Name: de/best/1

        :math:`\mathbf{v}_{i, G} = \mathbf{x}_{r_1, G} + differential_weight \cdot (\mathbf{x}_{r_2, G} - \mathbf{x}_{r_3, G}) + differential_weight \cdot (\mathbf{x}_{r_4, G} - \mathbf{x}_{r_5, G})`
        where :math:`r_1, r_2, r_3, r_4, r_5` are random indexes representing current population individuals.

    Crossover:
        Name: Binomial crossover

        :math:`\mathbf{x}_{i, G+1} = \begin{cases} \mathbf{u}_{i, G+1}, & \text{if $f(\mathbf{u}_{i, G+1}) \leq f(\mathbf{x}_{i, G})$}, \\ \mathbf{x}_{i, G}, & \text{otherwise}. \end{cases}`

    Args:
        pop (numpy.ndarray[Individual]): Current population.
        ic (int): Index of individual being mutated.
        f (float): Scale factor.
        cr (float): Crossover probability.
        rng (numpy.random.Generator): Random generator.

    Returns:
        numpy.ndarray: mutated and mixed individual.

    """
    j = rng.integers(len(pop[ic]))
    p = [1 / (len(pop) - 1.0) if i != ic else 0 for i in range(len(pop))] if len(pop) > 5 else None
    r = rng.choice(len(pop), 5, replace=not len(pop) >= 5, p=p)
    x = [pop[r[0]][i] + f * (pop[r[1]][i] - pop[r[2]][i]) + f * (
                pop[r[3]][i] - pop[r[4]][i]) if rng.random() < cr or i == j else pop[ic][i] for i in
         range(len(pop[ic]))]
    return np.asarray(x)


def cross_best2(pop, ic, f, cr, rng, x_b=None, **_kwargs):
    r"""Mutation strategy with crossover.

    Mutation:
        Name: de/best/2

        :math:`\mathbf{v}_{i, G} = \mathbf{x}_{best, G} + differential_weight \cdot (\mathbf{x}_{r_1, G} - \mathbf{x}_{r_2, G}) + differential_weight \cdot (\mathbf{x}_{r_3, G} - \mathbf{x}_{r_4, G})`
        where :math:`r_1, r_2, r_3, r_4` are random indexes representing current population individuals.

    Crossover:
        Name: Binomial crossover

        :math:`\mathbf{x}_{i, G+1} = \begin{cases} \mathbf{u}_{i, G+1}, & \text{if $f(\mathbf{u}_{i, G+1}) \leq f(\mathbf{x}_{i, G})$}, \\ \mathbf{x}_{i, G}, & \text{otherwise}. \end{cases}`

    Args:
        pop (numpy.ndarray[Individual]): Current population.
        ic (int): Index of individual being mutated.
        x_b (Individual): Current global best individual.
        f (float): Scale factor.
        cr (float): Crossover probability.
        rng (numpy.random.Generator): Random generator.

    Returns:
        numpy.ndarray: mutated and mixed individual.

    """
    j = rng.integers(len(pop[ic]))
    p = [1 / (len(pop) - 1.0) if i != ic else 0 for i in range(len(pop))] if len(pop) > 4 else None
    r = rng.choice(len(pop), 4, replace=not len(pop) >= 4, p=p)
    x = [x_b[i] + f * (pop[r[0]][i] - pop[r[1]][i]) + f * (
                pop[r[2]][i] - pop[r[3]][i]) if rng.random() < cr or i == j else pop[ic][i] for i in
         range(len(pop[ic]))]
    return np.asarray(x)


def cross_curr2rand1(pop, ic, f, cr, rng, **_kwargs):
    r"""Mutation strategy with crossover.

    Mutation:
        Name: de/curr2rand/1

        :math:`\mathbf{v}_{i, G} = \mathbf{x}_{i, G} + differential_weight \cdot (\mathbf{x}_{r_1, G} - \mathbf{x}_{r_2, G}) + differential_weight \cdot (\mathbf{x}_{r_3, G} - \mathbf{x}_{r_4, G})`
        where :math:`r_1, r_2, r_3, r_4` are random indexes representing current population individuals

    Crossover:
        Name: Binomial crossover

        :math:`\mathbf{x}_{i, G+1} = \begin{cases} \mathbf{u}_{i, G+1}, & \text{if $f(\mathbf{u}_{i, G+1}) \leq f(\mathbf{x}_{i, G})$}, \\ \mathbf{x}_{i, G}, & \text{otherwise}. \end{cases}`

    Args:
        pop (numpy.ndarray[Individual]): Current population.
        ic (int): Index of individual being mutated.
        f (float): Scale factor.
        cr (float): Crossover probability.
        rng (numpy.random.Generator): Random generator.

    Returns:
        numpy.ndarray: mutated and mixed individual.

    """
    j = rng.integers(len(pop[ic]))
    p = [1 / (len(pop) - 1.0) if i != ic else 0 for i in range(len(pop))] if len(pop) > 4 else None
    r = rng.choice(len(pop), 4, replace=not len(pop) >= 4, p=p)
    x = [pop[ic][i] + f * (pop[r[0]][i] - pop[r[1]][i]) + f * (
                pop[r[2]][i] - pop[r[3]][i]) if rng.random() < cr or i == j else pop[ic][i] for i in
         range(len(pop[ic]))]
    return np.asarray(x)


def cross_curr2best1(pop, ic, f, cr, rng, x_b=None, **_kwargs):
    r"""Mutation strategy with crossover.

    Mutation:
        Name: de/curr-to-best/1

        :math:`\mathbf{v}_{i, G} = \mathbf{x}_{i, G} + differential_weight \cdot (\mathbf{x}_{r_1, G} - \mathbf{x}_{r_2, G}) + differential_weight \cdot (\mathbf{x}_{r_3, G} - \mathbf{x}_{r_4, G})`
        where :math:`r_1, r_2, r_3, r_4` are random indexes representing current population individuals

    Crossover:
        Name: Binomial crossover

        :math:`\mathbf{x}_{i, G+1} = \begin{cases} \mathbf{u}_{i, G+1}, & \text{if $f(\mathbf{u}_{i, G+1}) \leq f(\mathbf{x}_{i, G})$}, \\ \mathbf{x}_{i, G}, & \text{otherwise}. \end{cases}`

    Args:
        pop (numpy.ndarray[Individual]): Current population.
        ic (int): Index of individual being mutated.
        x_b (Individual): Current global best individual.
        f (float): Scale factor.
        cr (float): Crossover probability.
        rng (numpy.random.Generator): Random generator.

    Returns:
        numpy.ndarray: mutated and mixed individual.

    """
    j = rng.integers(len(pop[ic]))
    p = [1 / (len(pop) - 1.0) if i != ic else 0 for i in range(len(pop))] if len(pop) > 3 else None
    r = rng.choice(len(pop), 3, replace=not len(pop) >= 3, p=p)
    x = [
        pop[ic][i] + f * (x_b[i] - pop[r[0]][i]) + f * (pop[r[1]][i] - pop[r[2]][i]) if rng.random() < cr or i == j else
        pop[ic][i] for i in range(len(pop[ic]))]
    return np.asarray(x)


class DifferentialEvolution(Algorithm):
    r"""Implementation of Differential evolution algorithm.

    Algorithm:
         Differential evolution algorithm

    Date:
        2018

    Author:
        Uros Mlakar and Klemen Berkovič

    License:
        MIT

    Reference paper:
        Storn, Rainer, and Kenneth Price. "Differential evolution - a simple and efficient heuristic for global optimization over continuous spaces." Journal of global optimization 11.4 (1997): 341-359.

    Attributes:
        Name (List[str]): List of string of names for algorithm.
        differential_weight (float): Scale factor.
        crossover_probability (float): Crossover probability.
        strategy (Callable[numpy.ndarray, int, numpy.ndarray, float, float, numpy.random.Generator, Dict[str, Any]]): crossover and mutation strategy.

    See Also:
        * :class:`niapy.algorithms.Algorithm`

    """

    Name = ['DifferentialEvolution', 'DE']

    @staticmethod
    def info():
        r"""Get basic information of algorithm.

        Returns:
            str: Basic information of algorithm.

        See Also:
            * :func:`niapy.algorithms.Algorithm.info`

        """
        return r"""Storn, Rainer, and Kenneth Price. "Differential evolution - a simple and efficient heuristic for global optimization over continuous spaces." Journal of global optimization 11.4 (1997): 341-359."""

    def __init__(self, population_size=50, differential_weight=1, crossover_probability=0.8, strategy=cross_rand1,
                 *args, **kwargs):
        """Initialize DifferentialEvolution.

        Args:
            population_size (Optional[int]): Population size.
            differential_weight (Optional[float]): Differential weight (differential_weight).
            crossover_probability (Optional[float]): Crossover rate.
            strategy (Optional[Callable[[numpy.ndarray, int, numpy.ndarray, float, float, numpy.random.Generator, list], numpy.ndarray]]):
                Crossover and mutation strategy.

        See Also:
            * :func:`niapy.algorithms.Algorithm.__init__`

        """
        super().__init__(population_size,
                         initialization_function=kwargs.pop('initialization_function', default_individual_init),
                         individual_type=kwargs.pop('individual_type', Individual), *args, **kwargs)
        self.differential_weight = differential_weight
        self.crossover_probability = crossover_probability
        self.strategy = strategy

    def set_parameters(self, population_size=50, differential_weight=1, crossover_probability=0.8, strategy=cross_rand1,
                       **kwargs):
        r"""Set the algorithm parameters.

        Args:
            population_size (Optional[int]): Population size.
            differential_weight (Optional[float]): Differential weight (differential_weight).
            crossover_probability (Optional[float]): Crossover rate.
            strategy (Optional[Callable[[numpy.ndarray, int, numpy.ndarray, float, float, numpy.random.Generator, list], numpy.ndarray]]):
                Crossover and mutation strategy.

        See Also:
            * :func:`niapy.algorithms.Algorithm.set_parameters`

        """
        super().set_parameters(population_size=population_size,
                               initialization_function=kwargs.pop('initialization_function', default_individual_init),
                               individual_type=kwargs.pop('individual_type', Individual), **kwargs)
        self.differential_weight = differential_weight
        self.crossover_probability = crossover_probability
        self.strategy = strategy

    def get_parameters(self):
        r"""Get parameters values of the algorithm.

        Returns:
            Dict[str, Any]: Algorithm parameters.

        See Also:
            * :func:`niapy.algorithms.Algorithm.get_parameters`

        """
        d = Algorithm.get_parameters(self)
        d.update({
            'differential_weight': self.differential_weight,
            'crossover_probability': self.crossover_probability,
            'strategy': self.strategy
        })
        return d

    def evolve(self, pop, xb, task, **kwargs):
        r"""Evolve population.

        Args:
            pop (numpy.ndarray): Current population.
            xb (numpy.ndarray): Current best individual.
            task (Task): Optimization task.

        Returns:
            numpy.ndarray: New evolved populations.

        """
        return objects_to_array(
            [self.individual_type(x=self.strategy(pop, i, self.differential_weight, self.crossover_probability, self.rng, x_b=xb), task=task, rng=self.rng, e=True) for i
             in range(len(pop))])

    def selection(self, population, new_population, best_x, best_fitness, task, **kwargs):
        r"""Operator for selection.

        Args:
            population (numpy.ndarray): Current population.
            new_population (numpy.ndarray): New Population.
            best_x (numpy.ndarray): Current global best solution.
            best_fitness (float): Current global best solutions fitness/objective value.
            task (Task): Optimization task.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray, float]:
                1. New selected individuals.
                2. New global best solution.
                3. New global best solutions fitness/objective value.

        """
        arr = objects_to_array([e if e.f < population[i].f else population[i] for i, e in enumerate(new_population)])
        best_x, best_fitness = self.get_best(arr, np.asarray([e.f for e in arr]), best_x, best_fitness)
        return arr, best_x, best_fitness

    def post_selection(self, pop, task, xb, fxb, **kwargs):
        r"""Apply additional operation after selection.

        Args:
            pop (numpy.ndarray): Current population.
            task (Task): Optimization task.
            xb (numpy.ndarray): Global best solution.
            fxb (float): Global best fitness.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray, float]:
                1. New population.
                2. New global best solution.
                3. New global best solutions fitness/objective value.

        """
        return pop, xb, fxb

    def run_iteration(self, task, population, population_fitness, best_x, best_fitness, **params):
        r"""Core function of Differential Evolution algorithm.

        Args:
            task (Task): Optimization task.
            population (numpy.ndarray): Current population.
            population_fitness (numpy.ndarray): Current populations fitness/function values.
            best_x (numpy.ndarray): Current best individual.
            best_fitness (float): Current best individual function/fitness value.
            **params (dict): Additional arguments.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, float, Dict[str, Any]]:
                1. New population.
                2. New population fitness/function values.
                3. New global best solution.
                4. New global best solutions fitness/objective value.
                5. Additional arguments.

        See Also:
            * :func:`niapy.algorithms.basic.DifferentialEvolution.evolve`
            * :func:`niapy.algorithms.basic.DifferentialEvolution.selection`
            * :func:`niapy.algorithms.basic.DifferentialEvolution.post_selection`

        """
        new_population = self.evolve(population, best_x, task)
        population, best_x, best_fitness = self.selection(population, new_population, best_x, best_fitness, task=task)
        population, best_x, best_fitness = self.post_selection(population, task, best_x, best_fitness)
        population_fitness = np.asarray([x.f for x in population])
        best_x, best_fitness = self.get_best(population, population_fitness, best_x, best_fitness)
        return population, population_fitness, best_x, best_fitness, {}


class DynNpDifferentialEvolution(DifferentialEvolution):
    r"""Implementation of Dynamic population size Differential evolution algorithm.

    Algorithm:
        Dynamic population size Differential evolution algorithm

    Date:
        2018

    Author:
        Klemen Berkovič

    License:
        MIT

    Attributes:
        Name (List[str]): List of strings representing algorithm names.
        p_max (int): Number of population reductions.
        rp (int): Small non-negative number which is added to value of generations.

    See Also:
        * :class:`niapy.algorithms.basic.DifferentialEvolution`

    """

    Name = ['DynNpDifferentialEvolution', 'dynNpDE']

    @staticmethod
    def info():
        r"""Get basic information of algorithm.

        Returns:
            str: Basic information of algorithm.

        See Also:
            * :func:`niapy.algorithms.Algorithm.info`

        """
        return r"""No info"""

    def __init__(self, population_size=10, p_max=50, rp=3, *args, **kwargs):
        """Initialize DynNpDifferentialEvolution.

        Args:
            p_max (Optional[int]): Number of population reductions.
            rp (Optional[int]): Small non-negative number which is added to value of generations.

        See Also:
            * :func:`niapy.algorithms.basic.DifferentialEvolution.__init__`

        """
        super().__init__(population_size, *args, **kwargs)
        self.p_max = p_max
        self.rp = rp

    def set_parameters(self, p_max=50, rp=3, **kwargs):
        r"""Set the algorithm parameters.

        Args:
            p_max (Optional[int]): Number of population reductions.
            rp (Optional[int]): Small non-negative number which is added to value of generations.

        See Also:
            * :func:`niapy.algorithms.basic.DifferentialEvolution.set_parameters`

        """
        super().set_parameters(**kwargs)
        self.p_max = p_max
        self.rp = rp

    def post_selection(self, pop, task, xb, fxb, **kwargs):
        r"""Post selection operator.

        In this algorithm the post selection operator decrements the population at specific iterations/generations.

        Args:
            pop (numpy.ndarray): Current population.
            task (Task): Optimization task.
            xb (numpy.ndarray): Global best individual coordinates.
            fxb (float): Global best fitness.
            kwargs (Dict[str, Any]): Additional arguments.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray, float]:
                1. Changed current population.
                2. New global best solution.
                3. New global best solutions fitness/objective value.

        """
        gr = task.max_evals // (self.p_max * len(pop)) + self.rp
        new_np = len(pop) // 2
        if (task.iters + 1) == gr and len(pop) > 3:
            pop = objects_to_array([pop[i] if pop[i].f < pop[i + new_np].f else pop[i + new_np] for i in range(new_np)])
        return pop, xb, fxb


def proportional(min_lifetime, max_lifetime, mu, x_f, avg, **_kwargs):
    r"""Proportional calculation of age of individual.

    Args:
        min_lifetime (int): Minimal life time.
        max_lifetime (int): Maximal life time.
        mu (float): Median of life time.
        x_f (float): Individuals function/fitness value.
        avg (float): Average fitness/function value of current population.

    Returns:
        int: Age of individual.

    """
    proportional_result = max_lifetime if math.isinf(avg) else min_lifetime + mu * avg / x_f
    return min(proportional_result, max_lifetime)


def linear(min_lifetime, mu, x_f, x_gw, x_gb, **_kwargs):
    r"""Linear calculation of age of individual.

    Args:
        min_lifetime (int): Minimal life time.
        mu (float): Median of life time.
        x_f (float): Individual function/fitness value.
        x_gw (float): Global worst fitness/function value.
        x_gb (float): Global best fitness/function value.

    Returns:
        int: Age of individual.

    """
    return min_lifetime + 2 * mu * (x_f - x_gw) / (x_gb - x_gw)


def bilinear(min_lifetime, max_lifetime, mu, x_f, avg, x_gw, x_gb, **_kwargs):
    r"""Bilinear calculation of age of individual.

    Args:
        min_lifetime (int): Minimal life time.
        max_lifetime (int): Maximal life time.
        mu (float): Median of life time.
        x_f (float): Individual function/fitness value.
        avg (float): Average fitness/function value.
        x_gw (float): Global worst fitness/function value.
        x_gb (float): Global best fitness/function value.

    Returns:
        int: Age of individual.

    """
    if avg < x_f:
        return min_lifetime + mu * (x_f - x_gw) / (x_gb - x_gw)
    return 0.5 * (min_lifetime + max_lifetime) + mu * (x_f - avg) / (x_gb - avg)


class AgingIndividual(Individual):
    r"""Individual with aging.

    Attributes:
        age (int): Age of individual.

    See Also:
        * :class:`niapy.algorithms.Individual`

    """

    def __init__(self, **kwargs):
        r"""Init Aging Individual.

        See Also:
            * :func:`niapy.algorithms.Individual.__init__`

        """
        super().__init__(**kwargs)
        self.age = 0


class AgingNpDifferentialEvolution(DifferentialEvolution):
    r"""Implementation of Differential evolution algorithm with aging individuals.

    Algorithm:
        Differential evolution algorithm with dynamic population size that is defined by the quality of population

    Date:
        2018

    Author:
        Klemen Berkovič

    License:
        MIT

    Attributes:
        Name (List[str]): list of strings representing algorithm names.
        Lt_min (int): Minimal age of individual.
        Lt_max (int): Maximal age of individual.
        delta_np (float): Proportion of how many individuals shall die.
        omega (float): Acceptance rate for individuals to die.
        mu (int): Mean of individual max and min age.
        age (Callable[[int, int, float, float, float, float, float], int]): Function for calculation of age for individual.

    See Also:
        * :class:`niapy.algorithms.basic.DifferentialEvolution`

    """

    Name = ['AgingNpDifferentialEvolution', 'ANpDE']

    @staticmethod
    def info():
        r"""Get basic information of algorithm.

        Returns:
            str: Basic information of algorithm.

        See Also:
            * :func:`niapy.algorithms.Algorithm.info`

        """
        return r"""No info"""

    def __init__(self, min_lifetime=0, max_lifetime=12, delta_np=0.3, omega=0.3, age=proportional, *args, **kwargs):
        """Initialize AgingNpDifferentialEvolution.

        Arguments:
            min_lifetime (Optional[int]): Minimum life time.
            max_lifetime (Optional[int]): Maximum life time.
            delta_np (Optional[float]): Proportion of how many individuals shall die.
            omega (Optional[float]): Acceptance rate for individuals to die.
            age (Optional[Callable[[int, int, float, float, float, float, float], int]]): Function for calculation of age for individual.

        See Also:
            * :func:`niapy.algorithms.basic.DifferentialEvolution.__init__`

        """
        super().__init__(individual_type=AgingIndividual, *args, **kwargs)
        self.min_lifetime = min_lifetime
        self.max_lifetime = max_lifetime
        self.age = age
        self.delta_np = delta_np
        self.omega = omega
        self.mu = abs(self.max_lifetime - self.min_lifetime) / 2

    def set_parameters(self, min_lifetime=0, max_lifetime=12, delta_np=0.3, omega=0.3, age=proportional, **kwargs):
        r"""Set the algorithm parameters.

        Arguments:
            min_lifetime (Optional[int]): Minimum life time.
            max_lifetime (Optional[int]): Maximum life time.
            delta_np (Optional[float]): Proportion of how many individuals shall die.
            omega (Optional[float]): Acceptance rate for individuals to die.
            age (Optional[Callable[[int, int, float, float, float, float, float], int]]): Function for calculation of age for individual.

        See Also:
            * :func:`niapy.algorithms.basic.DifferentialEvolution.set_parameters`

        """
        super().set_parameters(individual_type=AgingIndividual, **kwargs)
        self.min_lifetime = min_lifetime
        self.max_lifetime = max_lifetime
        self.age = age
        self.delta_np = delta_np
        self.omega = omega
        self.mu = abs(self.max_lifetime - self.min_lifetime) / 2

    def delta_pop_eliminated(self, t):
        r"""Calculate how many individuals are going to die.

        Args:
            t (int): Number of generations made by the algorithm.

        Returns:
            int: Number of individuals to dye.

        """
        return int(self.delta_np * abs(np.sin(t)))

    def delta_pop_created(self, t):
        r"""Calculate how many individuals are going to be created.

        Args:
            t (int): Number of generations made by the algorithm.

        Returns:
            int: Number of individuals to be born.

        """
        return int(self.delta_np * abs(np.cos(t)))

    def aging(self, task, pop):
        r"""Apply aging to individuals.

        Args:
            task (Task): Optimization task.
            pop (numpy.ndarray[Individual]): Current population.

        Returns:
            numpy.ndarray[Individual]: New population.

        """
        fpop = np.asarray([x.f for x in pop])
        x_b, x_w = pop[np.argmin(fpop)], pop[np.argmax(fpop)]
        avg = np.mean(fpop[np.isfinite(fpop)])
        new_population = []
        for x in pop:
            x.age += 1
            lifetime = round(
                self.age(min_lifetime=self.min_lifetime, max_lifetime=self.max_lifetime, mu=self.mu, x_f=x.f, avg=avg, x_gw=x_w.f, x_gb=x_b.f))
            if x.age <= lifetime:
                new_population.append(x)
        if len(new_population) == 0:
            new_population = objects_to_array([self.individual_type(task=task, rng=self.rng, e=True) for _ in range(self.population_size)])
        return new_population

    def increment_population(self, task):
        r"""Increment population.

        Args:
            task (Task): Optimization task.

        Returns:
            numpy.ndarray[Individual]: Increased population.

        """
        delta_pop = int(round(max(1, self.population_size * self.delta_pop_eliminated((task.iters + 1)))))
        return objects_to_array([self.individual_type(task=task, rng=self.rng, e=True) for _ in range(delta_pop)])

    def decrement_population(self, pop, task):
        r"""Decrement population.

        Args:
            pop (numpy.ndarray): Current population.
            task (Task): Optimization task.

        Returns:
            numpy.ndarray[Individual]: Decreased population.

        """
        delta_population = int(round(max(1, self.population_size * self.delta_pop_created((task.iters + 1)))))
        if len(pop) - delta_population <= 0:
            return pop
        ni = self.rng.choice(len(pop), delta_population, replace=False)
        new_population = []
        for i, e in enumerate(pop):
            if i not in ni:
                new_population.append(e)
            elif self.random() >= self.omega:
                new_population.append(e)
        return objects_to_array(new_population)

    def selection(self, population, new_population, best_x, best_fitness, task, **kwargs):
        r"""Select operator for individuals with aging.

        Args:
            population (numpy.ndarray): Current population.
            new_population (numpy.ndarray): New population.
            best_x (numpy.ndarray): Current global best solution.
            best_fitness (float): Current global best solutions fitness/objective value.
            task (Task): Optimization task.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray, float]:
                1. New population of individuals.
                2. New global best solution.
                3. New global best solutions fitness/objective value.

        """
        new_population, best_x, best_fitness = super().selection(population, new_population, best_x, best_fitness, task)
        new_population = np.append(new_population, self.increment_population(task))
        best_x, best_fitness = self.get_best(new_population, np.asarray([e.f for e in new_population]), best_x, best_fitness)
        population = self.aging(task, new_population)
        return population, best_x, best_fitness

    def post_selection(self, pop, task, xb, fxb, **kwargs):
        r"""Post selection operator.

        Args:
            pop (numpy.ndarray): Current population.
            task (Task): Optimization task.
            xb (Individual): Global best individual.
            fxb (float): Global best fitness.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray, float]:
            1. New population.
            2. New global best solution
            3. New global best solutions fitness/objective value

        """
        return self.decrement_population(pop, task) if len(pop) > self.population_size else pop, xb, fxb


def multi_mutations(pop, i, xb, differential_weight, crossover_probability, rng, task, individual_type, strategies,
                    **_kwargs):
    r"""Mutation strategy that takes more than one strategy and applies them to individual.

    Args:
        pop (numpy.ndarray[Individual]): Current population.
        i (int): Index of current individual.
        xb (Individual): Current best individual.
        differential_weight (float): Scale factor.
        crossover_probability (float): Crossover probability.
        rng (numpy.random.Generator): Random generator.
        task (Task): Optimization task.
        individual_type (Type[Individual]): Individual type used in algorithm.
        strategies (Iterable[Callable[[numpy.ndarray[Individual], int, Individual, float, float, numpy.random.Generator], numpy.ndarray[Individual]]]): List of mutation strategies.

    Returns:
        Individual: Best individual from applied mutations strategies.

    """
    population = [individual_type(x=strategy(pop, i, differential_weight, crossover_probability, x_b=xb, rng=rng), task=task, e=True, rng=rng) for strategy in strategies]
    return population[np.argmin([x.f for x in population])]


class MultiStrategyDifferentialEvolution(DifferentialEvolution):
    r"""Implementation of Differential evolution algorithm with multiple mutation strategies.

    Algorithm:
        Implementation of Differential evolution algorithm with multiple mutation strategies

    Date:
        2018

    Author:
        Klemen Berkovič

    License:
        MIT

    Attributes:
        Name (List[str]): List of strings representing algorithm names.
        strategies (Iterable[Callable[[numpy.ndarray[Individual], int, Individual, float, float, numpy.random.Generator], numpy.ndarray[Individual]]]): List of mutation strategies.

    See Also:
        * :class:`niapy.algorithms.basic.DifferentialEvolution`

    """

    Name = ['MultiStrategyDifferentialEvolution', 'MsDE']

    @staticmethod
    def info():
        r"""Get basic information of algorithm.

        Returns:
            str: Basic information of algorithm.

        See Also:
            * :func:`niapy.algorithms.Algorithm.info`

        """
        return r"""No info"""

    def __init__(self, population_size=40, strategies=(cross_rand1, cross_best1, cross_curr2best1, cross_rand2), *args, **kwargs):
        """Initialize MultiStrategyDifferentialEvolution.

        Args:
            strategies (Optional[Iterable[Callable[[numpy.ndarray[Individual], int, Individual, float, float, numpy.random.Generator], numpy.ndarray[Individual]]]]):
                List of mutation strategies.

        See Also:
            * :func:`niapy.algorithms.basic.DifferentialEvolution.__init__`

        """
        super().__init__(population_size, strategy=multi_mutations, *args, **kwargs)
        self.strategies = strategies

    def set_parameters(self, strategies=(cross_rand1, cross_best1, cross_curr2best1, cross_rand2), **kwargs):
        r"""Set the arguments of the algorithm.

        Args:
            strategies (Optional[Iterable[Callable[[numpy.ndarray[Individual], int, Individual, float, float, numpy.random.Generator], numpy.ndarray[Individual]]]]):
                List of mutation strategies.

        See Also:
            * :func:`niapy.algorithms.basic.DifferentialEvolution.set_parameters`

        """
        super().set_parameters(strategy=multi_mutations, **kwargs)
        self.strategies = strategies

    def get_parameters(self):
        r"""Get parameters values of the algorithm.

        Returns:
            Dict[str, Any]: Algorithm parameters.

        See Also:
            * :func:`niapy.algorithms.basic.DifferentialEvolution.get_parameters`

        """
        d = DifferentialEvolution.get_parameters(self)
        d.update({'strategies': self.strategies})
        return d

    def evolve(self, pop, xb, task, **kwargs):
        r"""Evolve population with the help multiple mutation strategies.

        Args:
            pop (numpy.ndarray): Current population.
            xb (numpy.ndarray): Current best individual.
            task (Task): Optimization task.

        Returns:
            numpy.ndarray: New population of individuals.

        """
        return objects_to_array(
            [self.strategy(pop, i, xb, self.differential_weight, self.crossover_probability, self.rng, task, self.individual_type, self.strategies) for i in
             range(len(pop))])


class DynNpMultiStrategyDifferentialEvolution(MultiStrategyDifferentialEvolution, DynNpDifferentialEvolution):
    r"""Implementation of Dynamic population size Differential evolution algorithm with dynamic population size that is defined by the quality of population.

    Algorithm:
        Dynamic population size Differential evolution algorithm with dynamic population size that is defined by the quality of population

    Date:
        2018

    Author:
        Klemen Berkovič

    License:
        MIT

    Attributes:
        Name (List[str]): List of strings representing algorithm name.

    See Also:
        * :class:`niapy.algorithms.basic.MultiStrategyDifferentialEvolution`
        * :class:`niapy.algorithms.basic.DynNpDifferentialEvolution`

    """

    Name = ['DynNpMultiStrategyDifferentialEvolution', 'dynNpMsDE']

    @staticmethod
    def info():
        r"""Get basic information of algorithm.

        Returns:
            str: Basic information of algorithm.

        See Also:
            * :func:`niapy.algorithms.Algorithm.info`

        """
        return r"""No info"""

    def set_parameters(self, **kwargs):
        r"""Set the arguments of the algorithm.

        See Also:
            * :func:`niapy.algorithms.basic.MultiStrategyDifferentialEvolution.set_parameters`
            * :func:`niapy.algorithms.basic.DynNpDifferentialEvolution.set_parameters`

        """
        DynNpDifferentialEvolution.set_parameters(self, **kwargs)
        MultiStrategyDifferentialEvolution.set_parameters(self, **kwargs)

    def evolve(self, pop, xb, task, **kwargs):
        r"""Evolve the current population.

        Args:
            pop (numpy.ndarray): Current population.
            xb (numpy.ndarray): Global best solution.
            task (Task): Optimization task.

        Returns:
            numpy.ndarray: Evolved new population.

        """
        return MultiStrategyDifferentialEvolution.evolve(self, pop, xb, task, **kwargs)

    def post_selection(self, pop, task, xb, fxb, **kwargs):
        r"""Post selection operator.

        Args:
            pop (numpy.ndarray): Current population.
            task (Task): Optimization task.
            xb (numpy.ndarray): Global best individual
            fxb (float): Global best fitness.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray, float]:
                1. New population.
                2. New global best solution.
                3. New global best solutions fitness/objective value.

        See Also:
            * :func:`niapy.algorithms.basic.DynNpDifferentialEvolution.post_selection`

        """
        return DynNpDifferentialEvolution.post_selection(self, pop, task, xb, fxb)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
