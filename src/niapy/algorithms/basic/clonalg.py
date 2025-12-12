# encoding=utf8
import numpy as np
import logging

from niapy.algorithms.algorithm import Algorithm

logging.basicConfig()
logger = logging.getLogger('niapy.algorithms.basic')
logger.setLevel('INFO')

__all__ = ['ClonalSelectionAlgorithm']


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

    Reference papers:
        * \L\. \N\. de Castro and F. J. Von Zuben. Learning and optimization using the clonal selection principle. IEEE Transactions on Evolutionary Computation, 6:239–251, 2002.
        * Brownlee, J. "Clever Algorithms: Nature-Inspired Programming Recipes" Revision 2. 2012. 280-286.

    Attributes:
        population_size (int): Population size.
        clone_factor (float): Clone factor.
        mutation_factor (float): Mutation factor.
        num_rand (int): Number of random antibodies to be added to the population each generation.
        bits_per_param (int): Number of bits per parameter of solution vector.

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
        return r"""L. N. de Castro and F. J. Von Zuben. Learning and optimization using the clonal selection principle. IEEE Transactions on Evolutionary Computation, 6:239–251, 2002."""

    def __init__(self, population_size=10, clone_factor=0.1, mutation_factor=10.0, num_rand=1, bits_per_param=16, *args,
                 **kwargs):
        """Initialize ClonalSelectionAlgorithm.

        Args:
            population_size (Optional[int]): Population size.
            clone_factor (Optional[float]): Clone factor.
            mutation_factor (Optional[float]): Mutation factor.
            num_rand (Optional[int]): Number of random antibodies to be added to the population each generation.
            bits_per_param (Optional[int]): Number of bits per parameter of solution vector.

        See Also:
            :func:`niapy.algorithms.Algorithm.__init__`

        """
        super().__init__(population_size, *args, **kwargs)
        self.clone_factor = clone_factor
        self.num_clones = int(self.population_size * self.clone_factor)
        self.mutation_factor = mutation_factor
        self.num_rand = num_rand
        self.bits_per_param = bits_per_param

    def set_parameters(self, population_size=10, clone_factor=0.1, mutation_factor=10.0, num_rand=1, bits_per_param=16,
                       **kwargs):
        r"""Set the parameters of the algorithm.

        Args:
            population_size (Optional[int]): Population size.
            clone_factor (Optional[float]): Clone factor.
            mutation_factor (Optional[float]): Mutation factor.
            num_rand (Optional[int]): Random number.
            bits_per_param (Optional[int]): Number of bits per parameter of solution vector.

        See Also:
            * :func:`niapy.algorithms.Algorithm.set_parameters`

        """
        super().set_parameters(population_size=population_size, **kwargs)
        self.clone_factor = clone_factor
        self.num_clones = int(self.population_size * self.clone_factor)
        self.mutation_factor = mutation_factor
        self.num_rand = num_rand
        self.bits_per_param = bits_per_param

    def get_parameters(self):
        r"""Get parameters of the algorithm.

        Returns:
            Dict[str, Any]: Algorithm parameters.

        """
        params = super().get_parameters()
        params.update({
            'clone_factor': self.clone_factor,
            'mutation_factor': self.mutation_factor,
            'num_rand': self.num_rand,
            'bits_per_param': self.bits_per_param,
        })
        return params

    def decode(self, bitstrings, task):
        bits = np.flip(np.arange(self.bits_per_param))
        z = np.sum(bitstrings * 2 ** bits, axis=-1)
        return task.lower + task.range * z / (2 ** self.bits_per_param - 1)

    def evaluate(self, bitstrings, task):
        population = self.decode(bitstrings, task)
        fitness = np.apply_along_axis(task.eval, 1, population)
        return population, fitness

    def mutate(self, bitstring, mutation_rate):
        flip = self.random(bitstring.shape) > mutation_rate
        bitstring[flip] = np.logical_not(bitstring[flip])
        return bitstring

    def clone_and_hypermutate(self, bitstrings, population, population_fitness, task):
        clones = np.repeat(bitstrings, self.num_clones, axis=0)
        for i in range(clones.shape[0]):
            mutation_rate = np.exp(-self.mutation_factor * population_fitness[i // self.num_clones])
            clones[i] = self.mutate(clones[i], mutation_rate)

        clones_pop, clones_fitness = self.evaluate(clones, task)
        all_bitstrings = np.concatenate((bitstrings, clones), axis=0)
        all_population = np.concatenate((population, clones_pop), axis=0)
        all_fitness = np.concatenate((population_fitness, clones_fitness))
        sorted_ind = np.argsort(all_fitness)

        new_bitstrings = all_bitstrings[sorted_ind][:self.population_size]
        new_population = all_population[sorted_ind][:self.population_size]
        new_fitness = all_fitness[sorted_ind][:self.population_size]

        return new_bitstrings, new_population, new_fitness

    def random_insertion(self, bitstrings, population, population_fitness, task):
        if self.num_rand == 0:
            return bitstrings, population, population_fitness
        new_bitstrings = self.random((self.num_rand, task.dimension, self.bits_per_param)) > 0.5
        new_population, new_fitness = self.evaluate(new_bitstrings, task)

        all_bitstrings = np.concatenate((bitstrings, new_bitstrings), axis=0)
        all_population = np.concatenate((population, new_population), axis=0)
        all_fitness = np.concatenate((population_fitness, new_fitness))
        sorted_ind = np.argsort(all_fitness)

        next_bitstrings = all_bitstrings[sorted_ind][:self.population_size]
        next_population = all_population[sorted_ind][:self.population_size]
        next_fitness = all_fitness[sorted_ind][:self.population_size]

        return next_bitstrings, next_population, next_fitness

    def init_population(self, task):
        r"""Initialize the starting population.

        Parameters:
            task (Task): Optimization task

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray[float], Dict[str, Any]]:
                1. New population.
                2. New population fitness/function values.
                3. Additional arguments:
                    * bitstring (numpy.ndarray): Binary representation of the population.

        See Also:
            * :func:`niapy.algorithms.Algorithm.init_population`

        """
        bitstrings = self.random((self.population_size, task.dimension, self.bits_per_param)) > 0.5
        population, fitness = self.evaluate(bitstrings, task)
        return population, fitness, {'bitstrings': bitstrings}

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
                5. Additional arguments:
                    * bitstring (numpy.ndarray): Binary representation of the population.

        """
        bitstrings = params.pop('bitstrings')

        bitstrings, population, population_fitness = self.clone_and_hypermutate(bitstrings, population,
                                                                                population_fitness, task)
        best_x, best_fitness = self.get_best(population, population_fitness, best_x, best_fitness)
        bitstrings, population, population_fitness = self.random_insertion(bitstrings, population, population_fitness,
                                                                           task)
        best_x, best_fitness = self.get_best(population, population_fitness, best_x, best_fitness)

        return population, population_fitness, best_x, best_fitness, {'bitstrings': bitstrings}
