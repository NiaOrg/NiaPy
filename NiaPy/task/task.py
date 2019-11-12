# encoding=utf8

"""The implementation of tasks."""

import logging
from enum import Enum

from matplotlib import pyplot as plt
from numpy import inf, random as rand

from NiaPy.util.utility import (
    limit_repair,
    fullArray
)
from NiaPy.util.exception import (
    FesException,
    GenException,
    RefException
)
from NiaPy.task.utility import Utility


logging.basicConfig()
logger = logging.getLogger("NiaPy.task.Task")
logger.setLevel("INFO")


class OptimizationType(Enum):
    r"""Enum representing type of optimization.

    Attributes:
            MINIMIZATION (int): Represents minimization problems and is default optimization type of all algorithms.
            MAXIMIZATION (int): Represents maximization problems.

    """

    MINIMIZATION = 1.0
    MAXIMIZATION = -1.0


class Task:
    r"""Class representing problem to solve with optimization.

    Date:
            2019

    Author:
            Klemen Berkovič and others

    Attributes:
            D (int): Dimension of the problem.
            Lower (numpy.ndarray): Lower bounds of the problem.
            Upper (numpy.ndarray): Upper bounds of the problem.
            bRange (numpy.ndarray): Search range between upper and lower limits.
            optType (OptimizationType): Optimization type to use.

    See Also:
            * :class:`NiaPy.util.Utility`

    """

    D = 0
    benchmark = None
    Lower, Upper, bRange = inf, inf, inf
    optType = OptimizationType.MINIMIZATION

    def __init__(self, D=0, optType=OptimizationType.MINIMIZATION, benchmark=None, Lower=None, Upper=None, frepair=limit_repair, **kwargs):
        r"""Initialize task class for optimization.

        Arguments:
                D (Optional[int]): Number of dimensions.
                optType (Optional[OptimizationType]): Set the type of optimization.
                benchmark (Union[str, Benchmark]): Problem to solve with optimization.
                Lower (Optional[numpy.ndarray]): Lower limits of the problem.
                Upper (Optional[numpy.ndarray]): Upper limits of the problem.
                frepair (Optional[Callable[[numpy.ndarray, numpy.ndarray, numpy.ndarray, Dict[str, Any]], numpy.ndarray]]): Function for reparing individuals components to desired limits.

        See Also:
                * `func`:NiaPy.util.Utility.__init__`
                * `func`:NiaPy.util.Utility.repair`

        """

        # dimension of the problem
        self.D = D
        # set optimization type
        self.optType = optType
        # set optimization function
        self.benchmark = Utility().get_benchmark(benchmark) if benchmark is not None else None

        if self.benchmark is not None:
            self.Fun = self.benchmark.function() if self.benchmark is not None else None

        # set Lower limits
        if Lower is not None:
            self.Lower = fullArray(Lower, self.D)
        elif Lower is None and benchmark is not None:
            self.Lower = fullArray(self.benchmark.Lower, self.D)
        else:
            self.Lower = fullArray(0, self.D)

        # set Upper limits
        if Upper is not None:
            self.Upper = fullArray(Upper, self.D)
        elif Upper is None and benchmark is not None:
            self.Upper = fullArray(self.benchmark.Upper, self.D)
        else:
            self.Upper = fullArray(0, self.D)

        # set range
        self.bRange = self.Upper - self.Lower
        # set repair function
        self.frepair = frepair

    def dim(self):
        r"""Get the number of dimensions.

        Returns:
                int: Dimension of problem optimizing.

        """

        return self.D

    def bcLower(self):
        r"""Get the array of lower bound constraint.

        Returns:
                numpy.ndarray: Lower bound.

        """

        return self.Lower

    def bcUpper(self):
        r"""Get the array of upper bound constraint.

        Returns:
                numpy.ndarray: Upper bound.

        """

        return self.Upper

    def bcRange(self):
        r"""Get the range of bound constraint.

        Returns:
                numpy.ndarray: Range between lower and upper bound.

        """

        return self.Upper - self.Lower

    def repair(self, x, rnd=rand):
        r"""Repair solution and put the solution in the random position inside of the bounds of problem.

        Arguments:
                x (numpy.ndarray): Solution to check and repair if needed.
                rnd (mtrand.RandomState): Random number generator.

        Returns:
                numpy.ndarray: Fixed solution.

        See Also:
                * :func:`NiaPy.util.limitRepair`
                * :func:`NiaPy.util.limitInversRepair`
                * :func:`NiaPy.util.wangRepair`
                * :func:`NiaPy.util.randRepair`
                * :func:`NiaPy.util.reflectRepair`

        """

        return self.frepair(x, self.Lower, self.Upper, rnd=rnd)

    def nextIter(self):
        r"""Increments the number of algorithm iterations."""

    def start(self):
        r"""Start stopwatch."""

    def eval(self, A):
        r"""Evaluate the solution A.

        Arguments:
                A (numpy.ndarray): Solution to evaluate.

        Returns:
                float: Fitness/function values of solution.

        """

        return self.Fun(self.D, A) * self.optType.value

    def isFeasible(self, A):
        r"""Check if the solution is feasible.

        Arguments:
                A (Union[numpy.ndarray, Individual]): Solution to check for feasibility.

        Returns:
                bool: `True` if solution is in feasible space else `False`.

        """

        return False not in (A >= self.Lower) and False not in (A <= self.Upper)

    def stopCond(self):
        r"""Check if optimization task should stop.

        Returns:
                bool: `True` if stopping condition is meet else `False`.

        """

        return False


class CountingTask(Task):
    r"""Optimization task with added counting of function evaluations and algorithm iterations/generations.

    Attributes:
            Iters (int): Number of algorithm iterations/generations.
            Evals (int): Number of function evaluations.

    See Also:
            * :class:`NiaPy.util.Task`

    """

    def __init__(self, **kwargs):
        r"""Initialize counting task.

        Args:
                **kwargs (Dict[str, Any]): Additional arguments.

        See Also:
                * :func:`NiaPy.util.Task.__init__`

        """

        Task.__init__(self, **kwargs)
        self.Iters, self.Evals = 0, 0

    def eval(self, A):
        r"""Evaluate the solution A.

        This function increments function evaluation counter `self.Evals`.

        Arguments:
                A (numpy.ndarray): Solutions to evaluate.

        Returns:
                float: Fitness/function values of solution.

        See Also:
                * :func:`NiaPy.util.Task.eval`

        """

        r = Task.eval(self, A)
        self.Evals += 1
        return r

    def evals(self):
        r"""Get the number of evaluations made.

        Returns:
                int: Number of evaluations made.

        """

        return self.Evals

    def iters(self):
        r"""Get the number of algorithm iteratins made.

        Returns:
                int: Number of generations/iterations made by algorithm.

        """

        return self.Iters

    def nextIter(self):
        r"""Increases the number of algorithm iterations made.

        This function increments number of algorithm iterations/generations counter `self.Iters`.

        """

        self.Iters += 1


class StoppingTask(CountingTask):
    r"""Optimization task with implemented checking for stopping criterias.

    Attributes:
            nGEN (int): Maximum number of algorithm iterations/generations.
            nFES (int): Maximum number of function evaluations.
            refValue (float): Reference function/fitness values to reach in optimization.
            x (numpy.ndarray): Best found individual.
            x_f (float): Best found individual function/fitness value.

    See Also:
            * :class:`NiaPy.util.CountingTask`

    """

    def __init__(self, nFES=inf, nGEN=inf, refValue=None, logger=False, **kwargs):
        r"""Initialize task class for optimization.

        Arguments:
                nFES (Optional[int]): Number of function evaluations.
                nGEN (Optional[int]): Number of generations or iterations.
                refValue (Optional[float]): Reference value of function/fitness function.
                logger (Optional[bool]): Enable/disable logging of improvements.

        Note:
                Storing improvements during the evolutionary cycle is
                captured in self.n_evals and self.x_f_vals

        See Also:
                * :func:`NiaPy.util.CountingTask.__init__`

        """

        CountingTask.__init__(self, **kwargs)
        self.refValue = (-inf if refValue is None else refValue)
        self.logger = logger
        self.x, self.x_f = None, inf
        self.nFES, self.nGEN = nFES, nGEN
        self.n_evals = []
        self.x_f_vals = []

    def eval(self, A):
        r"""Evaluate solution.

        Args:
                A (numpy.ndarray): Solution to evaluate.

        Returns:
                float: Fitness/function value of solution.

        See Also:
                * :func:`NiaPy.util.StoppingTask.stopCond`
                * :func:`NiaPy.util.CountingTask.eval`

        """

        if self.stopCond():
            return inf * self.optType.value

        x_f = CountingTask.eval(self, A)

        if x_f < self.x_f:
            self.x_f = x_f
            self.n_evals.append(self.Evals)
            self.x_f_vals.append(x_f)
            if self.logger:
                logger.info('nFES:%d => %s' % (self.Evals, self.x_f))

        return x_f

    def stopCond(self):
        r"""Check if stopping condition reached.

        Returns:
                bool: `True` if number of function evaluations or number of algorithm iterations/generations or reference values is reach else `False`

        """

        return (self.Evals >= self.nFES) or (self.Iters >= self.nGEN) or (self.refValue > self.x_f)

    def stopCondI(self):
        r"""Check if stopping condition reached and increase number of iterations.

        Returns:
                bool: `True` if number of function evaluations or number of algorithm iterations/generations or reference values is reach else `False`.

        See Also:
                * :func:`NiaPy.util.StoppingTask.stopCond`
                * :func:`NiaPy.util.CountingTask.nextIter`

        """

        r = self.stopCond()
        CountingTask.nextIter(self)
        return r

    def return_conv(self):
        r"""Get values of x and y axis for plotting covariance graph.

        Returns:
                Tuple[List[int], List[float]]:
                    1. List of ints of function evaluations.
                    2. List of ints of function/fitness values.

        """
        r1, r2 = [], []
        for i, v in enumerate(self.n_evals):
            r1.append(v), r2.append(self.x_f_vals[i])
            if i >= len(self.n_evals) - 1: break
            diff = self.n_evals[i + 1] - v
            if diff <= 1: continue
            for j in range(diff - 1): r1.append(v + j + 1), r2.append(self.x_f_vals[i])
        return r1, r2

    def plot(self):
        """Plot a simple convergence graph."""
        fess, fitnesses = self.return_conv()
        plt.plot(fess, fitnesses)
        plt.xlabel('nFes')
        plt.ylabel('Fitness')
        plt.title('Convergence graph')
        plt.show()


class ThrowingTask(StoppingTask):
    r"""Task that throw exceptions when stopping condition is meet.

    See Also:
            * :class:`NiaPy.util.StoppingTask`

    """

    def __init__(self, **kwargs):
        r"""Initialize optimization task.

        Args:
                **kwargs (Dict[str, Any]): Additional arguments.

        See Also:
                * :func:`NiaPy.util.StoppingTask.__init__`

        """

        StoppingTask.__init__(self, **kwargs)

    def stopCondE(self):
        r"""Throw exception for the given stopping condition.

        Raises:
                * FesException: Thrown when the number of function/fitness evaluations is reached.
                * GenException: Thrown when the number of algorithms generations/iterations is reached.
                * RefException: Thrown when the reference values is reached.
                * TimeException: Thrown when algorithm exceeds time run limit.

        """

        # dtime = datetime.now() - self.startTime
        if self.Evals >= self.nFES:
            raise FesException()
        if self.Iters >= self.nGEN:
            raise GenException()
        # if self.runTime is not None and self.runTime >= dtime: raise TimeException()
        if self.refValue >= self.x_f:
            raise RefException()

    def eval(self, A):
        r"""Evaluate solution.

        Args:
                A (numpy.ndarray): Solution to evaluate.

        Returns:
                float: Function/fitness values of solution.

        See Also:
                * :func:`NiaPy.util.ThrowingTask.stopCondE`
                * :func:`NiaPy.util.StoppingTask.eval`

        """

        self.stopCondE()
        return StoppingTask.eval(self, A)
