# encoding=utf8
import numpy as np
from numpy.random import uniform
from random import choice
import math
import copy
import logging

from niapy.algorithms.algorithm import Algorithm, Individual, default_individual_init

logging.basicConfig()
logger = logging.getLogger('niapy.algorithms.basic')
logger.setLevel('INFO')

__all__ = ['ClonalSelectionAlgorithm']


class Population:
    def __init__(self, bitstring='', vector=None, cost=0, affinity=0):
        if vector is None:
            vector = [0, 0]
        self.bitstring = bitstring
        self.vector = vector
        self.cost = cost
        self.affinity = affinity

    def set_bitstring(self, bitstring):
        self.bitstring = bitstring

    def set_vector(self, vector):
        self.vector = vector

    def set_cost(self, cost):
        self.cost = cost

    def set_affinity(self, affinity):
        self.affinity = affinity


class SolutionCLONALG(Individual):
    pass


class ClonalSelectionAlgorithm(Algorithm):
    r"""Implementation of Clonal Selection Algorithm.

        Algorithm:
            Clonal selection algorithm

        Date:
            2021

        Authors:
            Andraž Peršon

        License:
            MIT

        Reference paper:
            Brownlee, J. "Clever Algorithms: Nature-Inspired Programming Recipes" Revision 2. 2012. 280-286.

        Attributes:
            population_size (int): Size of population.
            search_space (List[Dict[int, int]]): Minimum and maximum values to be searched.
            max_gens (int): Number of iterations for algorithm.
            clone_factor (float): Factor based on which the clones are generated.
            num_rand (int): Random number for a seed.

        See Also:
            * :class:`niapy.algorithms.Algorithm`

        """

    Name = ['ClonalSelectionAlgorithm', 'CLONALG']

    @staticmethod
    def info():
        r"""Get algorithms information.

        Returns:
            str: Algorithm information.

        See Also:
            * :func:`niapy.algorithms.Algorithm.info`

        """
        return r"""Brownlee, J. "Clever Algorithms: Nature-Inspired Programming Recipes" Revision 2. 2012. 280-286."""

    def __init__(self, population_size=10, search_space=None, max_gens=100, clone_factor=0.1, num_rand=1, *args,
                 **kwargs):
        """Initialize ClonalSelectionAlgorithm.

                Args:
                    population_size (Optional[int]): Population size.
                    search_space (Optional[List[Dict[int, int]]]): Search space.
                    max_gens (Optional[int]): Maximum generations.
                    clone_factor (Optional[float]): Clone factor.
                    num_rand (Optional[int]): Random number.

                See Also:
                    :func:`niapy.algorithms.Algorithm.__init__`

                """
        super().__init__(
            population_size,
            initialization_function=default_individual_init,
            individual_type=SolutionCLONALG,
            *args, **kwargs
        )
        if search_space is None:
            search_space = [[-5, 5]]
        self.search_space = search_space
        self.max_gens = max_gens
        self.clone_factor = clone_factor
        self.num_rand = num_rand

    def set_parameters(self, population_size=10, search_space=None, max_gens=100, clone_factor=0.1, num_rand=0.1,
                       **kwargs):
        r"""Set the parameters of the algorithm.

                Args:
                    population_size (Optional[int]): Population size.
                    search_space (Optional[List[Dict[int, int]]]): Search space.
                    max_gens (Optional[int]): Maximum generations.
                    clone_factor (Optional[float]): Clone factor.
                    num_rand (Optional[int]): Random number.

                See Also:
                    * :func:`niapy.algorithms.Algorithm.set_parameters`

                """
        if search_space is None:
            search_space = [[-5, 5]]
        super().set_parameters(
            population_size=population_size,
            initialization_function=default_individual_init,
            individual_type=SolutionCLONALG,
            **kwargs
        )
        self.search_space = search_space
        self.max_gens = max_gens
        self.clone_factor = clone_factor
        self.num_rand = num_rand

    def get_parameters(self):
        r"""Get parameters of the algorithm.

        Returns:
            Dict[str, Any]: Algorithm parameters.

        """
        params = super().get_parameters()
        params.update({
            'search_space': self.search_space,
            'max_gens': self.max_gens,
            'clone_factor': self.clone_factor,
            'num_rand': self.num_rand
        })
        return params

    def objective_function(self, vector):
        return np.sum(np.power(vector, 2))

    def decode(self, bitstring, search_space, bits_per_param):
        vector = []

        for i, bounds in enumerate(search_space):
            off = i * bits_per_param
            sum = 0.0
            param_rev = bitstring[off:(off + bits_per_param)]
            param = param_rev[::-1]

            for x in range(len(param)):
                sum += (1.0 if (param[x] == '1') else 0.0) * (np.power(2.0, float(x)))

            min = bounds[0]
            max = bounds[1]

            vector.append(min + ((max - min) / (np.power(2.0, float(bits_per_param)) - 1.0)) * sum)
        return vector

    def evaluate(self, pop, search_space, bits_per_param):
        for p in pop:
            p.set_vector(self.decode(p.bitstring, search_space, bits_per_param))
            p.set_cost(self.objective_function(p.vector))

    def random_bitstring(self, num_bits):
        return ''.join(choice('01') for _ in range(num_bits))

    def point_mutation(self, bitstring, rate):
        child = ''

        for x in range(len(bitstring)):
            bit = bitstring[x]
            child += (('0' if (bit == '1') else '1') if (uniform(0, 1) < rate) else bit)

        return child

    def calculate_mutation_rate(self, antibody, mutate_factor=-2.5):
        return math.exp(mutate_factor * antibody.affinity)

    def calculate_num_clones(self, pop_size, clone_factor):
        return math.floor(pop_size * clone_factor)

    def calculate_affinity(self, pop):
        pop = sorted(pop, key=lambda _p: _p.cost)
        cost_range = pop[-1].cost - pop[0].cost

        if cost_range == 0.0:
            for p in pop:
                p.set_affinity(1.0)
        else:
            for p in pop:
                p.set_affinity(1.0 - (p.cost / cost_range))

    def clone_and_hypermutate(self, pop, clone_factor):
        clones = []
        num_clones = self.calculate_num_clones(len(pop), clone_factor)
        self.calculate_affinity(pop)
        for p in pop:
            m_rate = self.calculate_mutation_rate(p)
            for _ in range(num_clones):
                clone = Population(bitstring=self.point_mutation(p.bitstring, m_rate))
                clones.append(clone)

        return clones

    def random_insertion(self, search_space, pop, num_rand, bits_per_param):
        if num_rand == 0:
            return pop

        rands = []
        for _ in range(num_rand):
            rands.append(Population(bitstring=self.random_bitstring(len(search_space) * bits_per_param)))

        self.evaluate(rands, search_space, bits_per_param)

        sorted_pop = sorted(pop + rands, key=lambda p: p.cost)
        return sorted_pop[:len(pop)]

    def init_population(self, task):
        r"""Initialize the starting population.

        Parameters:
            task (Task): Optimization task

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray[float], Dict[str, Any]]:
                1. New population.
                2. New population fitness/function values.

        See Also:
            * :func:`niapy.algorithms.Algorithm.init_population`

        """
        population, fitness, d = super().init_population(task)
        return population, fitness, d

    def run_iteration(self, task, population, population_fitness, best_x, best_fitness, **params):
        r"""Core function of Clonal Selection Algorithm.

        Parameters:
            task (Task): Optimization task.
            population (numpy.ndarray): Current population
            population_fitness (numpy.ndarray[float]): Current population fitness/function values
            best_x (numpy.ndarray): Current best individual
            best_fitness (float): Current best individual function/fitness value
            params (Dict[str, Any]): Additional algorithm arguments

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, float, Dict[str, Any]]:
                1. New population
                2. New population fitness/function values
                3. New global best solution
                4. New global best fitness/objective value

        """
        pop = []
        bits_per_param = 16
        for _ in range(self.population_size):
            pop.append(Population(bitstring=self.random_bitstring(len(self.search_space) * bits_per_param)))

        self.evaluate(pop, self.search_space, bits_per_param)
        best = min(pop, key=lambda p: p.cost)
        for _ in range(self.max_gens):
            clones = self.clone_and_hypermutate(pop, self.clone_factor)
            self.evaluate(clones, self.search_space, bits_per_param)
            sorted_pop = sorted(pop + clones, key=lambda p: p.cost)
            pop = sorted_pop[:self.population_size]
            pop = self.random_insertion(self.search_space, pop, self.num_rand, bits_per_param)
            best = min((pop + [best]), key=lambda p: p.cost)

        return best

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
