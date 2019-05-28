# encoding=utf8
# pylint: disable=mixed-indentation, trailing-whitespace, multiple-statements, attribute-defined-outside-init, logging-not-lazy, no-self-use, line-too-long, arguments-differ, bad-continuation
import logging

from numpy import zeros, where, ones, full, max, min, abs
import math
from NiaPy.algorithms.algorithm import Algorithm
from NiaPy.util import OptimizationType

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
        'C1': lambda x: isinstance(x, (int, float)) and x > 0,
        'SMP': lambda x: isinstance(x, int) and x > 0,
        'SPC': lambda x: isinstance(x, bool),
        'CDC': lambda x: isinstance(x, (int, float)) and 0 <= x <= 1,
        'SRD': lambda x: isinstance(x, (int, float)) and 0 <= x <= 1,
        'vMax': lambda x: isinstance(x, (int, float)) and x > 0
    }

    def setParameters(self, NP=20, MR=0.05, C1=2.0, SMP=3, SPC=True, CDC=1, SRD=0.2, vMax=0.9, **ukwargs):
        r"""Set the algorithm parameters.

        Arguments:
            NP (int): Number of individuals in population

            MR {float}: Mixture ratio

            C1 {float}: Constant in tracing mode

            SMP {int}: Seeking memory pool

            SPC {boolean}: Self-position considering

            CDC {float}: Counts of dimension to change

            SRD {float}: Seeking range of the selected dimension

            vMax {float}: Maximal velocity

            See Also:
                * :func:`NiaPy.algorithms.Algorithm.setParameters`
        """
        Algorithm.setParameters(self, NP=NP, **ukwargs)
        self.MR, self.C1, self.SMP, self.SPC, self.CDC, self.SRD, self.vMax = MR, C1, SMP, SPC, CDC, SRD, vMax
        if ukwargs: logger.info('Unused arguments: %s' % (ukwargs))

    def initPopulation(self, task):
        r"""Initialize population.

        Args:
            task (Task): Optimization task.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray[float], Dict[str, Any]]:
                1. Initialized population.
                2. Initialized populations fitness/function values.
                3. Modes (seek or trace) and velocities for each cat
        See Also:
            * :func:`NiaPy.algorithms.Algorithm.initPopulation`
        """
        pop, fpop, d = Algorithm.initPopulation(self, task)
        d['modes'] = self.randomSeekTrace()
        d['velocities'] = []
        for _ in range(len(pop)):
            d['velocities'].append(self.uniform(-self.vMax, self.vMax, task.D))
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
        ir = where(x < l)
        x[ir] = l[ir]
        ir = where(x > u)
        x[ir] = u[ir]
        return x

    def randomSeekTrace(self):
        r"""Set cats into seeking/tracing mode.

        Returns:
            List[int]:
                1. One or zero. One means tracing mode. Zero means seeking mode. Length of list is equal to NP.
        """
        lista = zeros((self.NP,), dtype=int)
        choose_from = []
        while not int(self.NP * self.MR) == len(choose_from):
            r = self.randint(len(lista), 0)
            if r not in choose_from:
                lista[r] = 1
                choose_from.append(r)
        return lista

    def weightedSelection(self, weights):
        r"""Random selection considering the weights.

        Args:
            weights (numpy.ndarray): weight for each potential position.

        Returns:
            int: index of selected next position.
        """
        totals = []
        running_total = 0
        for w in weights:
            running_total += w
            totals.append(running_total)

        rnd = self.uniform(0, 1) * running_total
        for i, total in enumerate(totals):
            if rnd < total:
                return i
        return len(weights) - 1

    def seekingMode(self, task, cat, fcat):
        r"""Seeking mode.

        Args:
            task (Task): Optimization task.
            cat (numpy.ndarray[float]): Individual from population.
            fcat (float): Current individual's fitness/function value.
        Returns:
            cat_new (numpy.ndarray[float]): Updated individual's position.
            fcat_new (float): Updated individual's fitness/function value.
        """

        cat_copies = []  # potential next positions
        cat_copies_fs = []  # their fitness values
        for j in range(self.SMP - 1 if self.SPC else self.SMP):
            cat_copies.append(cat.copy())
            to_vary_indexes = []
            while not int(task.D * self.CDC) == len(to_vary_indexes):
                r = self.randint(task.D, 0)
                if r not in to_vary_indexes:
                    to_vary_indexes.append(r)
            if self.randint(2) == 1:
                cat_copies[j][to_vary_indexes] += cat_copies[j][to_vary_indexes] * self.SRD
            else:
                cat_copies[j][to_vary_indexes] -= cat_copies[j][to_vary_indexes] * self.SRD
            cat_copies[j] = self.repair(cat_copies[j], task.Lower, task.Upper)
            cat_copies_fs.append(task.eval(cat_copies[j]))
        if self.SPC:
            cat_copies.append(cat.copy())
            cat_copies_fs.append(fcat)

        cat_copies_select_probs = ones(len(cat_copies))
        if any(x != cat_copies_fs[0] for x in cat_copies_fs):  # if all fitness values are not equal calculate the weights
            fmax = max(cat_copies_fs)
            fmin = min(cat_copies_fs)
            fb = fmax if task.optType == OptimizationType.MINIMIZATION else fmin
            if math.isinf(fb):
                cat_copies_select_probs = full(len(cat_copies), fb)
            else:
                cat_copies_select_probs = abs(cat_copies_fs - fb) / (fmax - fmin)
        sel_index = self.weightedSelection(cat_copies_select_probs)
        cat_new = cat_copies[sel_index]
        fcat_new = cat_copies_fs[sel_index]
        return cat_new, fcat_new

    def tracingMode(self, task, cat, velocity, xb):
        r"""Tracing mode.

        Args:
            task (Task): Optimization task.
            cat (numpy.ndarray[float]): Individual from population.
            velocity (list [float]): Velocity of individual.
            xb (numpy.ndarray): Current best individual.
        Returns:
            cat_new (numpy.ndarray[float]): Updated individual's position.
            fcat_new (float): Updated individual's fitness/function value.
        """

        r = self.uniform(0, 1, len(velocity))
        Vnew = velocity.copy() + (r * self.C1 * (xb.copy() - cat.copy()))
        ifx = where(Vnew > self.vMax)
        Vnew[ifx] = self.vMax
        ifx = where(Vnew < -self.vMax)
        Vnew[ifx] = -self.vMax
        cat_new = cat + Vnew
        cat_new = self.repair(cat_new, task.Lower, task.Upper)
        fcat_new = task.eval(cat_new)
        return cat_new, fcat_new, Vnew

    def runIteration(self, task, pop, fpop, xb, fxb, velocities, modes, **dparams):
        r"""Core function of Cat Swarm Optimization algorithm.

        Args:
            task (Task): Optimization task.
            pop (numpy.ndarray): Current population.
            fpop (numpy.ndarray[float]): Current population fitness/function values.
            xb (numpy.ndarray): Current best individual.
            fxb (float): Current best cat fitness/function value.
            velocities (list of lists): Velocities of individuals.
            modes (numpy.ndarray): Flag of each individual.
            **dparams (Dict[str, Any]): Additional function arguments.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray[float], Dict[str, Any]]:
                1. New population
                2. New population fitness/function values
                3. Additional arguments:
                    * Dictionary of modes (seek or trace) and velocities for each cat
        """
        pop_copies = pop.copy()
        for k in range(len(pop_copies)):  # for each cat
            if modes[k] == 0:  # if cat in seeking mode
                pop_copies[k], fpop[k] = self.seekingMode(task, pop_copies[k], fpop[k])  # Seek
            else:  # if cat in tracing mode
                pop_copies[k], fpop[k], velocities[k] = self.tracingMode(task, pop_copies[k], velocities[k], xb)  # Trace
        return pop_copies, fpop, {'velocities': velocities, 'modes': self.randomSeekTrace()}

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
