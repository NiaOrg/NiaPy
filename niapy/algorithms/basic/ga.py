# encoding=utf8
import logging

import numpy as np

from niapy.algorithms.algorithm import Algorithm, Individual, default_individual_init

logging.basicConfig()
logger = logging.getLogger('niapy.algorithms.basic')
logger.setLevel('INFO')

__all__ = ['GeneticAlgorithm', 'tournament_selection', 'roulette_selection', 'two_point_crossover',
           'multi_point_crossover',
           'uniform_crossover', 'uniform_mutation', 'creep_mutation', 'crossover_uros', 'mutation_uros']


def tournament_selection(pop, _ic, ts, _x_b, rng):
    r"""Tournament selection method.

    Args:
        pop (numpy.ndarray[Individual]): Current population.
        _ic (int): Index of current individual in population. (Unused)
        ts (int): Tournament size.
        _x_b (numpy.ndarray): Global best individual. (Unused)
        rng (numpy.random.Generator): Random generator.

    Returns:
        Individual: Winner of the tournament.

    """
    comps = [pop[i] for i in rng.choice(len(pop), ts, replace=False)]
    return comps[np.argmin([c.f for c in comps])]


def roulette_selection(pop, ic, _ts, x_b, rng):
    r"""Roulette selection method.

    Args:
        pop (numpy.ndarray[Individual]): Current population.
        ic (int): Index of current individual in population.
        _ts (int): Unused argument.
        x_b (numpy.ndarray): Global best individual.
        rng (numpy.random.Generator): Random generator.

    Returns:
        Individual: selected individual.

    """
    f = np.sum([x.f for x in pop])
    qi = np.sum([pop[i].f / f for i in range(ic + 1)])
    return pop[ic].x if rng.random() < qi else x_b


def two_point_crossover(pop, ic, _cr, rng):
    r"""Two point crossover method.

    Args:
        pop (numpy.ndarray[Individual]): Current population.
        ic (int): Index of current individual.
        _cr (float): Crossover probability. (Unused)
        rng (numpy.random.Generator): Random generator.

    Returns:
        numpy.ndarray: New genotype.

    """
    io = ic
    while io != ic:
        io = rng.integers(len(pop))
    r = np.sort(rng.choice(len(pop[ic]), 2))
    x = pop[ic].x
    x[r[0]:r[1]] = pop[io].x[r[0]:r[1]]
    return np.asarray(x)


def multi_point_crossover(pop, ic, n, rng):
    r"""Multi point crossover method.

    Args:
        pop (numpy.ndarray[Individual]): Current population.
        ic (int): Index of current individual.
        n (flat): Number of points.
        rng (numpy.random.Generator): Random generator.

    Returns:
        numpy.ndarray: New genotype.

    """
    io = ic
    while io != ic:
        io = rng.integers(len(pop))
    r, x = np.sort(rng.choice(len(pop[ic]), 2 * n)), pop[ic].x
    for i in range(n):
        x[r[2 * i]:r[2 * i + 1]] = pop[io].x[r[2 * i]:r[2 * i + 1]]
    return np.asarray(x)


def uniform_crossover(pop, ic, cr, rng):
    r"""Uniform crossover method.

    Args:
        pop (numpy.ndarray[Individual]): Current population.
        ic (int): Index of current individual.
        cr (float): Crossover probability.
        rng (numpy.random.Generator): Random generator.

    Returns:
        numpy.ndarray: New genotype.

    """
    io = ic
    while io != ic:
        io = rng.integers(len(pop))
    j = rng.integers(len(pop[ic]))
    x = [pop[io][i] if rng.random() < cr or i == j else pop[ic][i] for i in range(len(pop[ic]))]
    return np.asarray(x)


def crossover_uros(pop, ic, cr, rng):
    r"""Crossover made by Uros Mlakar.

    Args:
        pop (numpy.ndarray[Individual]): Current population.
        ic (int): Index of current individual.
        cr (float): Crossover probability.
        rng (numpy.random.Generator): Random generator.

    Returns:
        numpy.ndarray: New genotype.

    """
    io = ic
    while io != ic:
        io = rng.integers(len(pop))
    alpha = cr + (1 + 2 * cr) * rng.random(len(pop[ic]))
    x = alpha * pop[ic] + (1 - alpha) * pop[io]
    return x


def uniform_mutation(pop, ic, mr, task, rng):
    r"""Uniform mutation method.

    Args:
        pop (numpy.ndarray[Individual]): Current population.
        ic (int): Index of current individual.
        mr (float): Mutation probability.
        task (Task): Optimization task.
        rng (numpy.random.Generator): Random generator.

    Returns:
        numpy.ndarray: New genotype.

    """
    j = rng.integers(task.dimension)
    nx = [rng.uniform(task.lower[i], task.upper[i]) if rng.random() < mr or i == j else pop[ic][i] for i in
          range(task.dimension)]
    return np.asarray(nx)


def mutation_uros(pop, ic, mr, task, rng):
    r"""Mutation method made by Uros Mlakar.

    Args:
        pop (numpy.ndarray[Individual]): Current population.
        ic (int): Index of individual.
        mr (float): Mutation rate.
        task (Task): Optimization task.
        rng (numpy.random.Generator): Random generator.

    Returns:
        numpy.ndarray: New genotype.

    """
    return np.fmin(np.fmax(rng.normal(pop[ic], mr * task.range), task.lower), task.upper)


def creep_mutation(pop, _ic, mr, task, rng):
    r"""Creep mutation method.

    Args:
        pop (numpy.ndarray[Individual]): Current population.
        _ic (int): Index of current individual. (Unused)
        mr (float): Mutation probability.
        task (Task): Optimization task.
        rng (numpy.random.Generator): Random generator.

    Returns:
        numpy.ndarray: New genotype.

    """
    ic, j = rng.integers(len(pop)), rng.integers(task.dimension)
    nx = [rng.uniform(task.lower[i], task.upper[i]) if rng.random() < mr or i == j else pop[ic][i] for i in
          range(task.dimension)]
    return np.asarray(nx)


class GeneticAlgorithm(Algorithm):
    r"""Implementation of Genetic Algorithm.

    Algorithm:
        Genetic algorithm

    Date:
        2018

    Author:
        Klemen Berkovič

    Reference paper:
        Goldberg, David (1989). Genetic Algorithms in Search, Optimization and Machine Learning. Reading, MA: Addison-Wesley Professional.

    License:
        MIT

    Attributes:
        Name (List[str]): List of strings representing algorithm name.
        tournament_size (int): Tournament size.
        mutation_rate (float): Mutation rate.
        crossover_rate (float): Crossover rate.
        selection (Callable[[numpy.ndarray[Individual], int, int, Individual, numpy.random.Generator], Individual]): selection operator.
        crossover (Callable[[numpy.ndarray[Individual], int, float, numpy.random.Generator], Individual]): Crossover operator.
        mutation (Callable[[numpy.ndarray[Individual], int, float, Task, numpy.random.Generator], Individual]): Mutation operator.

    See Also:
        * :class:`niapy.algorithms.Algorithm`

    """

    Name = ['GeneticAlgorithm', 'GA']

    @staticmethod
    def info():
        r"""Get basic information of algorithm.

        Returns:
            str: Basic information of algorithm.

        See Also:
            * :func:`niapy.algorithms.Algorithm.info`

        """
        return r"""On info"""

    def __init__(self, population_size=25, tournament_size=5, mutation_rate=0.25, crossover_rate=0.25,
                 selection=tournament_selection, crossover=uniform_crossover, mutation=uniform_mutation, *args, **kwargs):
        """Initialize GeneticAlgorithm.

        Args:
            population_size (Optional[int]): Population size.
            tournament_size (Optional[int]): Tournament selection.
            mutation_rate (Optional[int]): Mutation rate.
            crossover_rate (Optional[float]): Crossover rate.
            selection (Optional[Callable[[numpy.ndarray[Individual], int, int, Individual, numpy.random.Generator], Individual]]): Selection operator.
            crossover (Optional[Callable[[numpy.ndarray[Individual], int, float, numpy.random.Generator], Individual]]): Crossover operator.
            mutation (Optional[Callable[[numpy.ndarray[Individual], int, float, Task, numpy.random.Generator], Individual]]): Mutation operator.

        See Also:
            * :func:`niapy.algorithms.Algorithm.set_parameters`
            * selection:
                * :func:`niapy.algorithms.basic.tournament_selection`
                * :func:`niapy.algorithms.basic.roulette_selection`
            * Crossover:
                * :func:`niapy.algorithms.basic.uniform_crossover`
                * :func:`niapy.algorithms.basic.two_point_crossover`
                * :func:`niapy.algorithms.basic.multi_point_crossover`
                * :func:`niapy.algorithms.basic.crossover_uros`
            * Mutations:
                * :func:`niapy.algorithms.basic.uniform_mutation`
                * :func:`niapy.algorithms.basic.creep_mutation`
                * :func:`niapy.algorithms.basic.mutation_uros`

        """
        super().__init__(population_size,
                         individual_type=kwargs.pop('individual_type', Individual),
                         initialization_function=kwargs.pop('initialization_function', default_individual_init),
                         *args, **kwargs)
        self.tournament_size = tournament_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.selection = selection
        self.crossover = crossover
        self.mutation = mutation

    def set_parameters(self, population_size=25, tournament_size=5, mutation_rate=0.25, crossover_rate=0.25,
                       selection=tournament_selection, crossover=uniform_crossover, mutation=uniform_mutation, **kwargs):
        r"""Set the parameters of the algorithm.

        Args:
            population_size (Optional[int]): Population size.
            tournament_size (Optional[int]): Tournament selection.
            mutation_rate (Optional[int]): Mutation rate.
            crossover_rate (Optional[float]): Crossover rate.
            selection (Optional[Callable[[numpy.ndarray[Individual], int, int, Individual, numpy.random.Generator], Individual]]): selection operator.
            crossover (Optional[Callable[[numpy.ndarray[Individual], int, float, numpy.random.Generator], Individual]]): Crossover operator.
            mutation (Optional[Callable[[numpy.ndarray[Individual], int, float, Task, numpy.random.Generator], Individual]]): Mutation operator.

        See Also:
            * :func:`niapy.algorithms.Algorithm.set_parameters`
            * selection:
                * :func:`niapy.algorithms.basic.tournament_selection`
                * :func:`niapy.algorithms.basic.roulette_selection`
            * Crossover:
                * :func:`niapy.algorithms.basic.uniform_crossover`
                * :func:`niapy.algorithms.basic.two_point_crossover`
                * :func:`niapy.algorithms.basic.multi_point_crossover`
                * :func:`niapy.algorithms.basic.crossover_uros`
            * Mutations:
                * :func:`niapy.algorithms.basic.uniform_mutation`
                * :func:`niapy.algorithms.basic.creep_mutation`
                * :func:`niapy.algorithms.basic.mutation_uros`

        """
        super().set_parameters(population_size=population_size,
                               individual_type=kwargs.pop('individual_type', Individual),
                               initialization_function=kwargs.pop('initialization_function', default_individual_init),
                               **kwargs)
        self.tournament_size = tournament_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.selection = selection
        self.crossover = crossover
        self.mutation = mutation

    def run_iteration(self, task, population, population_fitness, best_x, best_fitness, **params):
        r"""Core function of GeneticAlgorithm algorithm.

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
                3. New global best solution
                4. New global best solutions fitness/objective value
                5. Additional arguments.

        """
        new_pop = np.empty(self.population_size, dtype=object)
        for i in range(self.population_size):
            ind = self.individual_type(x=self.selection(population, i, self.tournament_size, best_x, self.rng), e=False)
            ind.x = self.crossover(population, i, self.crossover_rate, self.rng)
            ind.x = self.mutation(population, i, self.mutation_rate, task, self.rng)
            ind.evaluate(task, rng=self.rng)
            new_pop[i] = ind
            if new_pop[i].f < best_fitness:
                best_x, best_fitness = self.get_best(new_pop[i], new_pop[i].f, best_x, best_fitness)
        return new_pop, np.asarray([i.f for i in new_pop]), best_x, best_fitness, {}

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
