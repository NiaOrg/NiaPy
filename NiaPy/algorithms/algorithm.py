# encoding=utf8

import numpy as np
from numpy import random as rand

from NiaPy.algorithms.individual import (
    Individual,
    defaultNumPyInit
)
from NiaPy.util.exception import (
    FesException,
    GenException,
    TimeException,
    RefException
)

__all__ = ['Algorithm']

class Algorithm:
    r"""Class for implementing algorithms.

    Date:
        2018

    Author
        Klemen Berkovič

    License:
        MIT

    Attributes:
        Name (List[str]): List of names for algorithm.
        Rand (rand.RandomState): Random generator.
        NP (int): Number of inidividuals in populatin.
        InitPopFunc (Callable[[Task, int, Optional[rand.RandomState], Dict[str, Any]], Tuple[numpy.ndarray, numpy.ndarray]]): Idividual initialization function.
        itype (Individual): Type of individuals used in population, default value is None for Numpy arrays.
    """
    Name = ['Algorithm', 'AAA']
    Rand = rand.RandomState(None)
    NP = 50
    InitPopFunc = defaultNumPyInit
    itype = None

    @staticmethod
    def typeParameters():
        r"""Return functions for checking values of parameters.

        Return:
            Dict[str, Callable[[Any], bool]]:
                * NP: Check if number of individuals is :math:`\in [0, \infty]`.
        """
        return {'NP': lambda x: isinstance(x, int) and x > 0}

    def __init__(self, seed=None, **kwargs):
        r"""Initialize algorithm and create name for an algorithm.

        Args:
            seed (Optional[int]): Starting seed for random generator.
            kwargs (Dict[str, Any]): Additional arguments.

        See Also:
            * :func:`NiaPy.algorithms.Algorithm.setParameters`
        """
        self.Rand, self.exception = rand.RandomState(seed), None
        self.setParameters(**kwargs)

    @staticmethod
    def algorithmInfo():
        r"""Get algorithm information.

        Returns:
            str: Bit item.
        """
        return '''Basic algorithm. No implementation!!!'''

    def setParameters(self, NP=50, InitPopFunc=defaultNumPyInit, itype=None, **kwargs):
        r"""Set the parameters/arguments of the algorithm.

        Args:
            NP (Optional[int]): Number of individuals in population :math:`\in [1, \infty]`.
            InitPopFunc (Optional[Callable[[Task, int, Optional[rand.RandomState], Dict[str, Any]], Tuple[numpy.ndarray, numpy.ndarray]]]): Type of individuals used by algorithm.
            itype (Individual): Individual type used in population, default is Numpy array.
            kwargs (Dict[str, Any]): Additional arguments.

        See Also:
            * :func:`NiaPy.algorithms.defaultNumPyInit`
            * :func:`NiaPy.algorithms.defaultIndividualInit`
        """
        self.NP, self.InitPopFunc, self.itype = NP, InitPopFunc, itype

    def getParameters(self):
        r"""Get parameters of the algorithm.

        Returns:
            Dict[str, Any]:
                * Parameter name: Represents a parameter name
                * Value of parameter: Represents the value of the parameter
        """
        return {
            'NP': self.NP,
            'InitPopFunc': self.InitPopFunc,
            'itype': self.itype
        }

    def rand(self, D=1):
        r"""Get random distribution of shape D in range from 0 to 1.

        Args:
            D (Optional[int]): Shape of returned random distribution.

        Returns:
            Union[float, numpy.ndarray]: Random number or numbers :math:`\in [0, 1]`.
        """
        if isinstance(D, (np.ndarray, list)): return self.Rand.rand(*D)
        elif D > 1: return self.Rand.rand(D)
        else: return self.Rand.rand()

    def uniform(self, Lower, Upper, D=None):
        r"""Get uniform random distribution of shape D in range from "Lower" to "Upper".

        Args:
            Lower (Union[float, numpy.ndarray]): Lower bound.
            Upper (Union[float, numpy.ndarray]): Upper bound.
            D (Optional[Union[int, Iterable[int]]]): Shape of returned uniform random distribution.

        Returns:
            Union[float, numpy.ndarray]: Array of numbers :math:`\in [\mathit{Lower}, \mathit{Upper}]`.
        """
        return self.Rand.uniform(Lower, Upper, D) if D is not None else self.Rand.uniform(Lower, Upper)

    def normal(self, loc, scale, D=None):
        r"""Get normal random distribution of shape D with mean "loc" and standard deviation "scale".

        Args:
            loc (float): Mean of the normal random distribution.
            scale (float): Standard deviation of the normal random distribution.
            D (Optional[Union[int, Iterable[int]]]): Shape of returned normal random distribution.

        Returns:
            Union[numpy.ndarray, float]: Array of numbers.
        """
        return self.Rand.normal(loc, scale, D) if D is not None else self.Rand.normal(loc, scale)

    def randn(self, D=None):
        r"""Get standard normal distribution of shape D.

        Args:
            D (Optional[Union[int, Iterable[int]]]): Shape of returned standard normal distribution.

        Returns:
            Union[numpy.ndarray, float]: Random generated numbers or one random generated number :math:`\in [0, 1]`.
        """
        if D is None: return self.Rand.randn()
        elif isinstance(D, int): return self.Rand.randn(D)
        return self.Rand.randn(*D)

    def randint(self, Nmax, D=1, Nmin=0, skip=None):
        r"""Get discrete uniform (integer) random distribution of D shape in range from "Nmin" to "Nmax".

        Args:
            Nmin (int): Lower integer bound.
            D (Optional[Union[int, Iterable[int]]]): shape of returned discrete uniform random distribution.
            Nmax (Optional[int]): One above upper integer bound.
            skip (Optional[Union[int, Iterable[int]]]): numbers to skip.

        Returns:
            Union[int, numpy.ndarray]: Random generated integer number.
        """
        r = None
        if isinstance(D, (list, tuple, np.ndarray)): r = self.Rand.randint(Nmin, Nmax, D)
        elif D > 1: r = self.Rand.randint(Nmin, Nmax, D)
        else: r = self.Rand.randint(Nmin, Nmax)
        return r if skip is None or r not in skip else self.randint(Nmax, D, Nmin, skip)

    def getBest(self, X, X_f, xb=None, xb_f=np.inf):
        r"""Get the best individual for population.

        Args:
            X (numpy.ndarray): Current population.
            X_f (numpy.ndarray): Current populations fitness/function values of aligned individuals.
            xb (Optional[numpy.ndarray]): Best individual.
            xb_f (Optional[float]): Fitness value of best individual.

        Returns:
            Tuple[numpy.ndarray, float]:
                1. Coordinates of best solution.
                2. beset fitness/function value.
        """
        ib = np.argmin(X_f)
        if isinstance(X_f, (float, int)) and xb_f >= X_f: xb, xb_f = X, X_f
        elif isinstance(X_f, (np.ndarray, list)) and xb_f >= X_f[ib]: xb, xb_f = X[ib], X_f[ib]
        return (xb.x.copy() if isinstance(xb, Individual) else xb.copy()), xb_f

    def initPopulation(self, task):
        r"""Initialize starting population of optimization algorithm.

        Args:
            task (Task): Optimization task.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray, Dict[str, Any]]:
                1. New population.
                2. New population fitness values.
                3. Additional arguments.

        See Also:
            * :func:`NiaPy.algorithms.Algorithm.setParameters`
        """
        pop, fpop = self.InitPopFunc(task=task, NP=self.NP, rnd=self.Rand, itype=self.itype)
        return pop, fpop, {}

    def runIteration(self, task, pop, fpop, xb, fxb, **dparams):
        r"""Core functionality of algorithm.

        This function is called on every algorithm iteration.

        Args:
            task (Task): Optimization task.
            pop (numpy.ndarray): Current population coordinates.
            fpop (numpy.ndarray): Current population fitness value.
            xb (numpy.ndarray): Current generation best individuals coordinates.
            xb_f (float): current generation best individuals fitness value.
            dparams (Dict[str, Any]): Additional arguments for algorithms.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, float, Dict[str, Any]]:
                1. New populations coordinates.
                2. New populations fitness values.
                3. New global best position/solution
                4. New global best fitness/objective value
                5. Additional arguments of the algorithm.

        See Also:
            * :func:`NiaPy.algorithms.Algorithm.runYield`
        """
        return pop, fpop, xb, fxb, dparams

    def runYield(self, task):
        r"""Run the algorithm for a single iteration and return the best solution.

        Args:
            task (Task): Task with bounds and objective function for optimization.

        Returns:
            Generator[Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, float, Dict[str, Any]], Tuple[numpy.ndarray, numpy.ndarray], None]: Generator getting new/old optimal global values.

        Yield:
            Tuple[numpy.ndarray, numpy.ndarray]:
                1. New population best individuals coordinates.
                2. Fitness value of the best solution.

        See Also:
            * :func:`NiaPy.algorithms.Algorithm.initPopulation`
            * :func:`NiaPy.algorithms.Algorithm.runIteration`
        """
        pop, fpop, dparams = self.initPopulation(task)
        xb, fxb = self.getBest(pop, fpop)
        yield xb, fxb
        while True:
            pop, fpop, xb, fxb, dparams = self.runIteration(task, pop, fpop, xb, fxb, **dparams)
            yield xb, fxb

    def runTask(self, task):
        r"""Start the optimization.

        Args:
            task (Task): Task with bounds and objective function for optimization.

        Returns:
            Tuple[numpy.ndarray, float]:
                1. Best individuals components found in optimization process.
                2. Best fitness value found in optimization process.

        See Also:
            * :func:`NiaPy.algorithms.Algorithm.runYield`
        """
        algo, xb, fxb = self.runYield(task), None, np.inf
        while not task.stopCond():
            xb, fxb = next(algo)
            task.nextIter()
        return xb, fxb

    def run(self, task):
        r"""Start the optimization.

        Args:
            task (Task): Optimization task.

        Returns:
            Tuple[numpy.ndarray, float]:
                1. Best individuals components found in optimization process.
                2. Best fitness value found in optimization process.

        See Also:
            * :func:`NiaPy.algorithms.Algorithm.runTask`
        """
        try:
            # task.start()
            r = self.runTask(task)
            return r[0], r[1] * task.optType.value
        except (FesException, GenException, TimeException, RefException): return task.x, task.x_f * task.optType.value
        except Exception as e: self.exception = e
        return None, None

    def __call__(self, task):
        r"""Start the optimization.

        Args:
            task (Task): Optimization task.

        Returns:
            Tuple[numpy.ndarray, float]:
                1. Best individuals components found in optimization process.
                2. Best fitness value found in optimization process.

        See Also:
            * :func:`NiaPy.algorithms.Algorithm.run`
        """
        return self.run(task)

    def bad_run(self):
        r"""Check if some exeptions where thrown when the algorithm was running.

        Returns:
            bool: True if some error where detected at runtime of the algorithm, otherwise False
        """
        return self.exception is not None
