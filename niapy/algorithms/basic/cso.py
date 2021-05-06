# encoding=utf8
import logging
import math

import numpy as np

from niapy.algorithms.algorithm import Algorithm

logging.basicConfig()
logger = logging.getLogger('niapy.algorithms.basic')
logger.setLevel('INFO')

__all__ = ['CatSwarmOptimization']


class CatSwarmOptimization(Algorithm):
    r"""Implementation of Cat swarm optimization algorithm.

    **Algorithm:** Cat swarm optimization

    **Date:** 2019

    **Author:** Mihael Baketarić

    **License:** MIT

    **Reference paper:** Chu, S. C., Tsai, P. W., & Pan, J. S. (2006). Cat swarm optimization.
    In Pacific Rim international conference on artificial intelligence (pp. 854-858). Springer, Berlin, Heidelberg.

    """

    Name = ['CatSwarmOptimization', 'CSO']

    @staticmethod
    def info():
        r"""Get algorithm information.

        Returns:
            str: Algorithm information.

        See Also:
            * :func:`niapy.algorithms.Algorithm.info`

        """
        return r"""Chu, S. C., Tsai, P. W., & Pan, J. S. (2006). Cat swarm optimization.
        In Pacific Rim international conference on artificial intelligence (pp. 854-858).
        Springer, Berlin, Heidelberg."""

    def __init__(self, population_size=30, mixture_ratio=0.1, c1=2.05, smp=3, spc=True, cdc=0.85, srd=0.2,
                 max_velocity=1.9, *args, **kwargs):
        """Initialize CatSwarmOptimization.

        Args:
            population_size (int): Number of individuals in population.
            mixture_ratio (float): Mixture ratio.
            c1 (float): Constant in tracing mode.
            smp (int): Seeking memory pool.
            spc (bool): Self-position considering.
            cdc (float): Decides how many dimensions will be varied.
            srd (float): Seeking range of the selected dimension.
            max_velocity (float): Maximal velocity.

            See Also:
                * :func:`niapy.algorithms.Algorithm.__init__`

        """
        super().__init__(population_size, *args, **kwargs)
        self.mixture_ratio = mixture_ratio
        self.c1 = c1
        self.smp = smp
        self.spc = spc
        self.cdc = cdc
        self.srd = srd
        self.max_velocity = max_velocity

    def set_parameters(self, population_size=30, mixture_ratio=0.1, c1=2.05, smp=3, spc=True, cdc=0.85, srd=0.2,
                       max_velocity=1.9, **kwargs):
        r"""Set the algorithm parameters.

        Args:
            population_size (int): Number of individuals in population.
            mixture_ratio (float): Mixture ratio.
            c1 (float): Constant in tracing mode.
            smp (int): Seeking memory pool.
            spc (bool): Self-position considering.
            cdc (float): Decides how many dimensions will be varied.
            srd (float): Seeking range of the selected dimension.
            max_velocity (float): Maximal velocity.

            See Also:
                * :func:`niapy.algorithms.Algorithm.set_parameters`

        """
        super().set_parameters(population_size, **kwargs)
        self.mixture_ratio = mixture_ratio
        self.c1 = c1
        self.smp = smp
        self.spc = spc
        self.cdc = cdc
        self.srd = srd
        self.max_velocity = max_velocity

    def init_population(self, task):
        r"""Initialize population.

        Args:
            task (Task): Optimization task.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray[float], Dict[str, Any]]:
                1. Initialized population.
                2. Initialized populations fitness/function values.
                3. Additional arguments:
                    * Dictionary of modes (seek or trace) and velocities for each cat
        See Also:
            * :func:`niapy.algorithms.Algorithm.init_population`

        """
        pop, fpop, d = super().init_population(task)
        d['velocities'] = self.uniform(-self.max_velocity, self.max_velocity, (self.population_size, task.dimension))
        return pop, fpop, d

    def random_seek_trace(self):
        r"""Set cats into seeking/tracing mode randomly.

        Returns:
            numpy.ndarray: One or zero. One means tracing mode. Zero means seeking mode. Length of list is equal to population_size.

        """
        modes = np.zeros(self.population_size, dtype=np.int32)
        indices = self.rng.choice(self.population_size, int(self.population_size * self.mixture_ratio), replace=False)
        modes[indices] = 1
        return modes

    def weighted_selection(self, weights):
        r"""Random selection considering the weights.

        Args:
            weights (numpy.ndarray): weight for each potential position.

        Returns:
            int: index of selected next position.

        """
        cumulative_sum = np.cumsum(weights)
        return np.argmax(cumulative_sum >= (self.random() * cumulative_sum[-1]))

    def seeking_mode(self, task, cat, cat_fitness, pop, fpop, fxb):
        r"""Seeking mode.

        Args:
            task (Task): Optimization task.
            cat (numpy.ndarray): Individual from population.
            cat_fitness (float): Current individual's fitness/function value.
            pop (numpy.ndarray): Current population.
            fpop (numpy.ndarray): Current population fitness/function values.
            fxb (float): Current best cat fitness/function value.

        Returns:
            Tuple[numpy.ndarray, float, numpy.ndarray, float]:
                1. Updated individual's position
                2. Updated individual's fitness/function value
                3. Updated global best position
                4. Updated global best fitness/function value

        """
        cat_copies = []
        cat_copies_fs = []
        for j in range(self.smp - 1 if self.spc else self.smp):
            cat_copies.append(cat.copy())
            indexes = np.arange(task.dimension)
            self.rng.shuffle(indexes)
            to_vary_indexes = indexes[:int(task.dimension * self.cdc)]
            if self.integers(2) == 1:
                cat_copies[j][to_vary_indexes] += cat_copies[j][to_vary_indexes] * self.srd
            else:
                cat_copies[j][to_vary_indexes] -= cat_copies[j][to_vary_indexes] * self.srd
            cat_copies[j] = task.repair(cat_copies[j])
            cat_copies_fs.append(task.eval(cat_copies[j]))
        if self.spc:
            cat_copies.append(cat.copy())
            cat_copies_fs.append(cat_fitness)

        cat_copies_select_probs = np.ones(len(cat_copies))
        worst_fitness = np.max(cat_copies_fs)
        best_fitness = np.min(cat_copies_fs)
        if any(x != cat_copies_fs[0] for x in cat_copies_fs):
            fb = worst_fitness
            if math.isinf(fb):
                cat_copies_select_probs = np.full(len(cat_copies), fb)
            else:
                cat_copies_select_probs = np.abs(cat_copies_fs - fb) / (worst_fitness - best_fitness)
        if best_fitness < fxb:
            ind = self.integers(self.population_size)
            pop[ind] = cat_copies[np.where(cat_copies_fs == best_fitness)[0][0]]
            fpop[ind] = best_fitness
        sel_index = self.weighted_selection(cat_copies_select_probs)
        return cat_copies[sel_index], cat_copies_fs[sel_index], pop, fpop

    def tracing_mode(self, task, cat, velocity, xb):
        r"""Tracing mode.

        Args:
            task (Task): Optimization task.
            cat (numpy.ndarray): Individual from population.
            velocity (numpy.ndarray): Velocity of individual.
            xb (numpy.ndarray): Current best individual.
        Returns:
            Tuple[numpy.ndarray, float, numpy.ndarray]:
                1. Updated individual's position
                2. Updated individual's fitness/function value
                3. Updated individual's velocity vector

        """
        new_velocity = np.clip(velocity + (self.random(len(velocity)) * self.c1 * (xb - cat)),
                               -self.max_velocity, self.max_velocity)
        cat_new = task.repair(cat + new_velocity)
        return cat_new, task.eval(cat_new), new_velocity

    def run_iteration(self, task, population, population_fitness, best_x, best_fitness, **params):
        r"""Core function of Cat Swarm Optimization algorithm.

        Args:
            task (Task): Optimization task.
            population (numpy.ndarray): Current population.
            population_fitness (numpy.ndarray): Current population fitness/function values.
            best_x (numpy.ndarray): Current best individual.
            best_fitness (float): Current best cat fitness/function value.
            **params (Dict[str, Any]): Additional function arguments.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, float, Dict[str, Any]]:
                1. New population.
                2. New population fitness/function values.
                3. New global best solution.
                4. New global best solutions fitness/objective value.
                5. Additional arguments:
                    * velocities (numpy.ndarray): velocities of cats.

        """
        modes = self.random_seek_trace()
        velocities = params.pop('velocities')

        pop_copies = population.copy()
        for k in range(len(pop_copies)):
            if modes[k] == 0:
                pop_copies[k], population_fitness[k], pop_copies, population_fitness = self.seeking_mode(task,
                                                                                                         pop_copies[k],
                                                                                                         population_fitness[k],
                                                                                                         pop_copies,
                                                                                                         population_fitness,
                                                                                                         best_fitness)
            else:  # if cat in tracing mode
                pop_copies[k], population_fitness[k], velocities[k] = self.tracing_mode(task,
                                                                                        pop_copies[k],
                                                                                        velocities[k],
                                                                                        best_x)
        best_index = np.argmin(population_fitness)
        if population_fitness[best_index] < best_fitness:
            best_x, best_fitness = pop_copies[best_index].copy(), population_fitness[best_index]
        return pop_copies, population_fitness, best_x, best_fitness, {'velocities': velocities}

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
