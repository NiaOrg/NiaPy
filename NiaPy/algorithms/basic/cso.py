# encoding=utf8
import logging
import math

import numpy as np
from NiaPy.algorithms.algorithm import Algorithm
logging.basicConfig()
logger = logging.getLogger('NiaPy.algorithms.basic')
logger.setLevel('INFO')

__all__ = ['CatSwarmOptimization']

class CatSwarmOptimization(Algorithm):
    r"""Implementation of Cat swarm optimiization algorithm.

    **Algorithm:** Cat swarm optimization

    **Date:** 2019

    **Author:** Mihael Baketarić

    **License:** MIT

    **Reference paper:** Chu, Shu-Chuan & Tsai, Pei-Wei & Pan, Jeng-Shyang. (2006). Cat Swarm Optimization. 854-858. 10.1007/11801603_94.
    """
    Name = ['CatSwarmOptimization', 'CSO']

    @staticmethod
    def typeParameters(): return {
        'NP': lambda x: isinstance(x, int) and x > 0,
        'MR': lambda x: isinstance(x, (int, float)) and 0 <= x <= 1,
        'C1': lambda x: isinstance(x, (int, float)) and x >= 0,
        'SMP': lambda x: isinstance(x, int) and x > 0,
        'SPC': lambda x: isinstance(x, bool),
        'CDC': lambda x: isinstance(x, (int, float)) and 0 <= x <= 1,
        'SRD': lambda x: isinstance(x, (int, float)) and 0 <= x <= 1,
        'vMax': lambda x: isinstance(x, (int, float)) and x > 0
    }

    def setParameters(self, NP=30, MR=0.1, C1=2.05, SMP=3, SPC=True, CDC=0.85, SRD=0.2, vMax=1.9, **ukwargs):
        r"""Set the algorithm parameters.

        Arguments:
            NP (int): Number of individuals in population
            MR (float): Mixture ratio
            C1 (float): Constant in tracing mode
            SMP (int): Seeking memory pool
            SPC (bool): Self-position considering
            CDC (float): Decides how many dimensions will be varied
            SRD (float): Seeking range of the selected dimension
            vMax (float): Maximal velocity

            See Also:
                * :func:`NiaPy.algorithms.Algorithm.setParameters`
        """
        Algorithm.setParameters(self, NP=NP, **ukwargs)
        self.MR, self.C1, self.SMP, self.SPC, self.CDC, self.SRD, self.vMax = MR, C1, SMP, SPC, CDC, SRD, vMax

    def initPopulation(self, task):
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
            * :func:`NiaPy.algorithms.Algorithm.initPopulation`
        """
        pop, fpop, d = Algorithm.initPopulation(self, task)
        d['modes'] = self.randomSeekTrace()
        d['velocities'] = self.uniform(-self.vMax, self.vMax, [len(pop), task.D])
        return pop, fpop, d

    def repair(self, x, l, u):
        r"""Repair array to range.

        Args:
            x (numpy.ndarray): Array to repair.
            l (numpy.ndarray): Lower limit of allowed range.
            u (numpy.ndarray): Upper limit of allowed range.

        Returns:
            numpy.ndarray: Repaired array.
        """
        ir = np.where(x < l)
        x[ir] = l[ir]
        ir = np.where(x > u)
        x[ir] = u[ir]
        return x

    def randomSeekTrace(self):
        r"""Set cats into seeking/tracing mode.

        Returns:
            numpy.ndarray: One or zero. One means tracing mode. Zero means seeking mode. Length of list is equal to NP.
        """
        lista = np.zeros((self.NP,), dtype=int)
        indexes = np.arange(self.NP)
        self.Rand.shuffle(indexes)
        lista[indexes[:int(self.NP * self.MR)]] = 1
        return lista

    def weightedSelection(self, weights):
        r"""Random selection considering the weights.

        Args:
            weights (numpy.ndarray): weight for each potential position.

        Returns:
            int: index of selected next position.
        """
        cumulative_sum = np.cumsum(weights)
        return np.argmax(cumulative_sum >= (self.rand() * cumulative_sum[-1]))

    def seekingMode(self, task, cat, fcat, pop, fpop, fxb):
        r"""Seeking mode.

        Args:
            task (Task): Optimization task.
            cat (numpy.ndarray): Individual from population.
            fcat (float): Current individual's fitness/function value.
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
        for j in range(self.SMP - 1 if self.SPC else self.SMP):
            cat_copies.append(cat.copy())
            indexes = np.arange(task.D)
            self.Rand.shuffle(indexes)
            to_vary_indexes = indexes[:int(task.D * self.CDC)]
            if self.randint(2) == 1:
                cat_copies[j][to_vary_indexes] += cat_copies[j][to_vary_indexes] * self.SRD
            else:
                cat_copies[j][to_vary_indexes] -= cat_copies[j][to_vary_indexes] * self.SRD
            cat_copies[j] = task.repair(cat_copies[j])
            cat_copies_fs.append(task.eval(cat_copies[j]))
        if self.SPC:
            cat_copies.append(cat.copy())
            cat_copies_fs.append(fcat)

        cat_copies_select_probs = np.ones(len(cat_copies))
        fmax = np.max(cat_copies_fs)
        fmin = np.min(cat_copies_fs)
        if any(x != cat_copies_fs[0] for x in cat_copies_fs):
            fb = fmax
            if math.isinf(fb):
                cat_copies_select_probs = np.full(len(cat_copies), fb)
            else:
                cat_copies_select_probs = np.abs(cat_copies_fs - fb) / (fmax - fmin)
        if fmin < fxb:
            fxb = fmin
            ind = self.randint(self.NP, 1, 0)
            pop[ind] = cat_copies[np.where(cat_copies_fs == fmin)[0][0]]
            fpop[ind] = fmin
        sel_index = self.weightedSelection(cat_copies_select_probs)
        return cat_copies[sel_index], cat_copies_fs[sel_index], pop, fpop

    def tracingMode(self, task, cat, velocity, xb):
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
        Vnew = self.repair(velocity + (self.uniform(0, 1, len(velocity)) * self.C1 * (xb - cat)), np.full(task.D, -self.vMax), np.full(task.D, self.vMax))
        cat_new = task.repair(cat + Vnew)
        return cat_new, task.eval(cat_new), Vnew

    def runIteration(self, task, pop, fpop, xb, fxb, velocities, modes, **dparams):
        r"""Core function of Cat Swarm Optimization algorithm.

        Args:
            task (Task): Optimization task.
            pop (numpy.ndarray): Current population.
            fpop (numpy.ndarray): Current population fitness/function values.
            xb (numpy.ndarray): Current best individual.
            fxb (float): Current best cat fitness/function value.
            velocities (numpy.ndarray): Velocities of individuals.
            modes (numpy.ndarray): Flag of each individual.
            **dparams (Dict[str, Any]): Additional function arguments.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, float, Dict[str, Any]]:
                1. New population.
                2. New population fitness/function values.
                3. New global best solution.
                4. New global best solutions fitness/objective value.
                5. Additional arguments:
                    * Dictionary of modes (seek or trace) and velocities for each cat.
        """
        pop_copies = pop.copy()
        for k in range(len(pop_copies)):
            if modes[k] == 0:
                pop_copies[k], fpop[k], pop_copies[:], fpop[:] = self.seekingMode(task, pop_copies[k], fpop[k], pop_copies, fpop, fxb)
            else:  # if cat in tracing mode
                pop_copies[k], fpop[k], velocities[k] = self.tracingMode(task, pop_copies[k], velocities[k], xb)
        ib = np.argmin(fpop)
        if fpop[ib] < fxb: xb, fxb = pop_copies[ib].copy(), fpop[ib]
        return pop_copies, fpop, xb, fxb, {'velocities': velocities, 'modes': self.randomSeekTrace()}

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
