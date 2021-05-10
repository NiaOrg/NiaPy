# encoding=utf8

"""The implementation of tasks."""

import logging
from enum import Enum

import numpy as np
from matplotlib import pyplot as plt
from niapy.benchmarks import Benchmark
from niapy.util.array import full_array
import niapy.util.repair as repair
from niapy.util.factory import get_benchmark
from niapy.util.exception import FesException, GenException, RefException

logging.basicConfig()
logger = logging.getLogger("niapy.task.Task")
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
        dimension (int): Dimension of the problem.
        lower (numpy.ndarray): Lower bounds of the problem.
        upper (numpy.ndarray): Upper bounds of the problem.
        range (numpy.ndarray): Search range between upper and lower limits.
        optimization_type (OptimizationType): Optimization type to use.

    """

    def __init__(self, dimension=0, optimization_type=OptimizationType.MINIMIZATION, benchmark=None, lower=None,
                 upper=None, repair_function=repair.limit):
        r"""Initialize task class for optimization.

        Args:
            dimension (Optional[int]): Number of dimensions.
            optimization_type (Optional[OptimizationType]): Set the type of optimization.
            benchmark (Union[str, Benchmark]): Problem to solve with optimization.
            lower (Optional[numpy.ndarray]): Lower limits of the problem.
            upper (Optional[numpy.ndarray]): Upper limits of the problem.
            repair_function (Optional[Callable[[numpy.ndarray, numpy.ndarray, numpy.ndarray, Dict[str, Any]], numpy.ndarray]]): Function for repairing individuals components to desired limits.

        See Also:
            * `func`:niapy.task.Utility.__init__`

        """
        # dimension of the problem
        self.dimension = dimension
        # set optimization type
        self.optimization_type = optimization_type
        # set optimization function
        self.benchmark = None
        if isinstance(benchmark, str):
            self.benchmark = get_benchmark(benchmark)
        elif isinstance(benchmark, Benchmark):
            self.benchmark = benchmark

        if self.benchmark is not None:
            self.function = self.benchmark.function() if self.benchmark is not None else None

        # set Lower limits
        if lower is not None:
            self.lower = full_array(lower, self.dimension)
        elif lower is None and benchmark is not None:
            self.lower = full_array(self.benchmark.lower, self.dimension)
        else:
            self.lower = full_array(0, self.dimension)

        # set Upper limits
        if upper is not None:
            self.upper = full_array(upper, self.dimension)
        elif upper is None and benchmark is not None:
            self.upper = full_array(self.benchmark.upper, self.dimension)
        else:
            self.upper = full_array(0, self.dimension)

        # set range
        self.range = self.upper - self.lower
        # set repair function
        self.repair_function = repair_function

    def repair(self, x, rng=None):
        r"""Repair solution and put the solution in the random position inside of the bounds of problem.

        Args:
            x (numpy.ndarray): Solution to check and repair if needed.
            rng (numpy.random.Generator): Random number generator.

        Returns:
            numpy.ndarray: Fixed solution.

        See Also:
            * :func:`niapy.util.repair.limit`
            * :func:`niapy.util.repair.limit_inverse`
            * :func:`niapy.util.repair.wang`
            * :func:`niapy.util.repair.rand`
            * :func:`niapy.util.repair.reflect`

        """
        return self.repair_function(x, self.lower, self.upper, rng=rng)

    def next_iter(self):
        r"""Increments the number of algorithm iterations."""

    def start(self):
        r"""Start stopwatch."""

    def eval(self, x):
        r"""Evaluate the solution A.

        Args:
            x (numpy.ndarray): Solution to evaluate.

        Returns:
            float: Fitness/function values of solution.

        """
        return self.function(self.dimension, x) * self.optimization_type.value

    def is_feasible(self, x):
        r"""Check if the solution is feasible.

        Args:
            x (Union[numpy.ndarray, Individual]): Solution to check for feasibility.

        Returns:
            bool: `True` if solution is in feasible space else `False`.

        """
        return np.all((x >= self.lower) & (x <= self.upper))

    def stopping_condition(self):
        r"""Check if optimization task should stop.

        Returns:
            bool: `True` if stopping condition is meet else `False`.

        """
        return False


class CountingTask(Task):
    r"""Optimization task with added counting of function evaluations and algorithm iterations/generations.

    Attributes:
        iters (int): Number of algorithm iterations/generations.
        evals (int): Number of function evaluations.

    See Also:
        * :class:`niapy.task.Task`

    """

    def __init__(self, dimension=0, optimization_type=OptimizationType.MINIMIZATION, benchmark=None, lower=None,
                 upper=None, repair_function=repair.limit):
        r"""Initialize counting task.

        Args:
            dimension (Optional[int]): Number of dimensions.
            optimization_type (Optional[OptimizationType]): Set the type of optimization.
            benchmark (Union[str, Benchmark]): Problem to solve with optimization.
            lower (Optional[numpy.ndarray]): Lower limits of the problem.
            upper (Optional[numpy.ndarray]): Upper limits of the problem.
            repair_function (Optional[Callable[[numpy.ndarray, numpy.ndarray, numpy.ndarray, Dict[str, Any]], numpy.ndarray]]): Function for repairing individuals components to desired limits.

        See Also:
            * :func:`niapy.task.Task.__init__`

        """
        super().__init__(dimension, optimization_type, benchmark, lower, upper, repair_function)
        self.iters = 0
        self.evals = 0

    def eval(self, x):
        r"""Evaluate the solution x.

        This function increments function evaluation counter `self.Evals`.

        Args:
            x (numpy.ndarray): Solutions to evaluate.

        Returns:
            float: Fitness/function values of solution.

        See Also:
            * :func:`niapy.task.Task.eval`

        """
        r = super().eval(x)
        self.evals += 1
        return r

    def next_iter(self):
        r"""Increases the number of algorithm iterations made.

        This function increments number of algorithm iterations/generations counter `self.Iters`.

        """
        self.iters += 1


class StoppingTask(CountingTask):
    r"""Optimization task with implemented checking for stopping criteria.

    Attributes:
        max_iters (int): Maximum number of algorithm iterations/generations.
        max_evals (int): Maximum number of function evaluations.
        cutoff_value (float): Reference function/fitness values to reach in optimization.
        x (numpy.ndarray): Best found individual.
        x_f (float): Best found individual function/fitness value.

    See Also:
        * :class:`niapy.task.CountingTask`

    """

    def __init__(self, max_evals=np.inf, max_iters=np.inf, cutoff_value=None, enable_logging=False, dimension=0,
                 optimization_type=OptimizationType.MINIMIZATION, benchmark=None, lower=None, upper=None,
                 repair_function=repair.limit):
        r"""Initialize task class for optimization.

        Args:
            max_evals (Optional[int]): Number of function evaluations.
            max_iters (Optional[int]): Number of generations or iterations.
            cutoff_value (Optional[float]): Reference value of function/fitness function.
            enable_logging (Optional[bool]): Enable/disable logging of improvements.
            dimension (Optional[int]): Number of dimensions.
            optimization_type (Optional[OptimizationType]): Set the type of optimization.
            benchmark (Union[str, Benchmark]): Problem to solve with optimization.
            lower (Optional[numpy.ndarray]): Lower limits of the problem.
            upper (Optional[numpy.ndarray]): Upper limits of the problem.
            repair_function (Optional[Callable[[numpy.ndarray, numpy.ndarray, numpy.ndarray, Dict[str, Any]], numpy.ndarray]]): Function for repairing individuals components to desired limits.

        Note:
            Storing improvements during the evolutionary cycle is
            captured in self.n_evals and self.x_f_vals

        See Also:
            * :func:`niapy.task.CountingTask.__init__`

        """
        super().__init__(dimension, optimization_type, benchmark, lower, upper, repair_function)
        self.cutoff_value = -np.inf * optimization_type.value if cutoff_value is None else cutoff_value
        self.enable_logging = enable_logging
        self.x = None
        self.x_f = np.inf * optimization_type.value
        self.max_evals = max_evals
        self.max_iters = max_iters
        self.n_evals = []
        self.x_f_vals = []

    def eval(self, x):
        r"""Evaluate solution.

        Args:
            x (numpy.ndarray): Solution to evaluate.

        Returns:
            float: Fitness/function value of solution.

        See Also:
            * :func:`niapy.task.StoppingTask.stopping_condition`
            * :func:`niapy.task.CountingTask.eval`

        """
        if self.stopping_condition():
            return np.inf * self.optimization_type.value

        x_f = super().eval(x)

        if x_f < self.x_f:
            self.x_f = x_f
            self.n_evals.append(self.evals)
            self.x_f_vals.append(x_f)
            if self.enable_logging:
                logger.info('evals:%d => %s' % (self.evals, self.x_f))

        return x_f

    def stopping_condition(self):
        r"""Check if stopping condition reached.

        Returns:
            bool: `True` if number of function evaluations or number of algorithm iterations/generations or reference values is reach else `False`.

        """
        return (self.evals >= self.max_evals) or (self.iters >= self.max_iters) or (self.cutoff_value > self.x_f)

    def stopping_condition_iter(self):
        r"""Check if stopping condition reached and increase number of iterations.

        Returns:
            bool: `True` if number of function evaluations or number of algorithm iterations/generations or reference values is reach else `False`.

        See Also:
            * :func:`niapy.task.StoppingTask.stopping_condition`
            * :func:`niapy.task.CountingTask.next_iter`

        """
        r = self.stopping_condition()
        self.next_iter()
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
            if i >= len(self.n_evals) - 1:
                break
            diff = self.n_evals[i + 1] - v
            if diff <= 1:
                continue
            for j in range(diff - 1):
                r1.append(v + j + 1), r2.append(self.x_f_vals[i])
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
        * :class:`niapy.task.StoppingTask`

    """

    def stopping_condition_throw(self):
        r"""Throw exception for the given stopping condition.

        Raises:
            * FesException: Thrown when the number of function/fitness evaluations is reached.
            * GenException: Thrown when the number of algorithms generations/iterations is reached.
            * RefException: Thrown when the reference values is reached.
            * TimeException: Thrown when algorithm exceeds time run limit.

        """
        if self.evals >= self.max_evals:
            raise FesException()
        if self.iters >= self.max_iters:
            raise GenException()
        if self.cutoff_value >= self.x_f:
            raise RefException()

    def eval(self, x):
        r"""Evaluate solution.

        Args:
            x (numpy.ndarray): Solution to evaluate.

        Returns:
            float: Function/fitness values of solution.

        See Also:
            * :func:`niapy.task.ThrowingTask.stopping_condition_throw`
            * :func:`niapy.task.StoppingTask.eval`

        """
        self.stopping_condition_throw()
        return super().eval(x)
