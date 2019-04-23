# encoding=utf8

"""Implementation of benchmarks utility function."""

import logging

from numpy import ndarray, asarray, full, empty, where, random as rand, ceil, amin, amax

from NiaPy.benchmarks import Benchmark, Rastrigin, Rosenbrock, Griewank, Sphere, Ackley, Schwefel, Schwefel221, Schwefel222, Whitley, Alpine1, Alpine2, HappyCat, Ridge, ChungReynolds, Csendes, Pinter, Qing, Quintic, Salomon, SchumerSteiglitz, Step, Step2, Step3, Stepint, SumSquares, StyblinskiTang, BentCigar, Discus, Elliptic, ExpandedGriewankPlusRosenbrock, HGBat, Katsuura, ExpandedSchaffer, ModifiedSchwefel, Weierstrass, Michalewichz, Levy, Sphere2, Sphere3, Trid, Perm, Zakharov, DixonPrice, Powell, CosineMixture, Infinity, SchafferN2, SchafferN4

logging.basicConfig()
logger = logging.getLogger("NiaPy.util.utility")
logger.setLevel("INFO")

__all__ = [
    "limit_repair",
    "limitInversRepair",
    "objects2array",
    "wangRepair",
    "randRepair",
    "fullArray",
    "reflectRepair"
]


class Utility:
    r"""Base class with string mappings to benchmarks and algorithms.

    Attributes:
        classes (Dict[str, Benchmark]): Mapping from stings to benchmark.

    """

    def __init__(self):
        r"""Initializing the algorithm and benchmark objects."""

        self.benchmark_classes = {
            "ackley": Ackley,
            "alpine1": Alpine1,
            "alpine2": Alpine2,
            "bentcigar": BentCigar,
            "chungReynolds": ChungReynolds,
            "cosinemixture": CosineMixture,
            "csendes": Csendes,
            "discus": Discus,
            "dixonprice": DixonPrice,
            "conditionedellptic": Elliptic,
            "elliptic": Elliptic,
            "expandedgriewankplusrosenbrock": ExpandedGriewankPlusRosenbrock,
            "expandedschaffer": ExpandedSchaffer,
            "griewank": Griewank,
            "happyCat": HappyCat,
            "hgbat": HGBat,
            "infinity": Infinity,
            "katsuura": Katsuura,
            "levy": Levy,
            "michalewicz": Michalewichz,
            "modifiedscwefel": ModifiedSchwefel,
            "perm": Perm,
            "pinter": Pinter,
            "powell": Powell,
            "qing": Qing,
            "quintic": Quintic,
            "rastrigin": Rastrigin,
            "ridge": Ridge,
            "rosenbrock": Rosenbrock,
            "salomon": Salomon,
            "schaffer2": SchafferN2,
            "schaffer4": SchafferN4,
            "schumerSteiglitz": SchumerSteiglitz,
            "schwefel": Schwefel,
            "schwefel221": Schwefel221,
            "schwefel222": Schwefel222,
            "sphere": Sphere,
            "sphere2": Sphere2,
            "sphere3": Sphere3,
            "step": Step,
            "step2": Step2,
            "step3": Step3,
            "stepint": Stepint,
            "styblinskiTang": StyblinskiTang,
            "sumSquares": SumSquares,
            "trid": Trid,
            "weierstrass": Weierstrass,
            "whitley": Whitley,
            "zakharov": Zakharov
        }

        self.algorithm_classes = {}


    def get_benchmark(self, benchmark):
        r"""Get the optimization problem.

        Arguments:
            benchmark (Union[str, Benchmark]): String or class that represents the optimization problem.

        Returns:
            Benchmark: Optimization function with limits.

        """

        if issubclass(type(benchmark), Benchmark):
            return benchmark
        elif benchmark in self.benchmark_classes.keys():
            return self.benchmark_classes[benchmark]()
        else:
            raise TypeError("Passed benchmark is not defined!")

    @classmethod
    def __raiseLowerAndUpperNotDefined(cls):
        r"""Trow exception if lower and upper bounds are not defined in benchmark.

        Raises:
            TypeError: Type error.
        """
        raise TypeError("Upper and Lower value must be defined!")


def limit_repair(x, Lower, Upper, **kwargs):
    r"""Repair solution and put the solution in the random position inside of the bounds of problem.

    Arguments:
            x (numpy.ndarray): Solution to check and repair if needed.
            Lower (numpy.ndarray): Lower bounds of search space.
            Upper (numpy.ndarray): Upper bounds of search space.
            kwargs (Dict[str, Any]): Additional arguments.

    Returns:
            numpy.ndarray: Solution in search space.
    """
    ir = where(x < Lower)
    x[ir] = Lower[ir]
    ir = where(x > Upper)
    x[ir] = Upper[ir]
    return x


def limitInversRepair(x, Lower, Upper, **kwargs):
    r"""Repair solution and put the solution in the random position inside of the bounds of problem.

    Arguments:
            x (numpy.ndarray): Solution to check and repair if needed.
            Lower (numpy.ndarray): Lower bounds of search space.
            Upper (numpy.ndarray): Upper bounds of search space.
            kwargs (Dict[str, Any]): Additional arguments.

    Returns:
            numpy.ndarray: Solution in search space.
    """
    ir = where(x < Lower)
    x[ir] = Upper[ir]
    ir = where(x > Upper)
    x[ir] = Lower[ir]
    return x


def wangRepair(x, Lower, Upper, **kwargs):
    r"""Repair solution and put the solution in the random position inside of the bounds of problem.

    Arguments:
            x (numpy.ndarray): Solution to check and repair if needed.
            Lower (numpy.ndarray): Lower bounds of search space.
            Upper (numpy.ndarray): Upper bounds of search space.
            kwargs (Dict[str, Any]): Additional arguments.

    Returns:
            numpy.ndarray: Solution in search space.
    """
    ir = where(x < Lower)
    x[ir] = amin([Upper[ir], 2 * Lower[ir] - x[ir]], axis=0)
    ir = where(x > Upper)
    x[ir] = amax([Lower[ir], 2 * Upper[ir] - x[ir]], axis=0)
    return x


def randRepair(x, Lower, Upper, rnd=rand, **kwargs):
    r"""Repair solution and put the solution in the random position inside of the bounds of problem.

    Arguments:
            x (numpy.ndarray): Solution to check and repair if needed.
            Lower (numpy.ndarray): Lower bounds of search space.
            Upper (numpy.ndarray): Upper bounds of search space.
            rnd (mtrand.RandomState): Random generator.
            kwargs (Dict[str, Any]): Additional arguments.

    Returns:
            numpy.ndarray: Fixed solution.
    """
    ir = where(x < Lower)
    x[ir] = rnd.uniform(Lower[ir], Upper[ir])
    ir = where(x > Upper)
    x[ir] = rnd.uniform(Lower[ir], Upper[ir])
    return x


def reflectRepair(x, Lower, Upper, **kwargs):
    r"""Repair solution and put the solution in search space with reflection of how much the solution violates a bound.

    Args:
            x (numpy.ndarray): Solution to be fixed.
            Lower (numpy.ndarray): Lower bounds of search space.
            Upper (numpy.ndarray): Upper bounds of search space.
            kwargs (Dict[str, Any]): Additional arguments.

    Returns:
            numpy.ndarray: Fix solution.
    """
    ir = where(x > Upper)
    x[ir] = Lower[ir] + x[ir] % (Upper[ir] - Lower[ir])
    ir = where(x < Lower)
    x[ir] = Lower[ir] + x[ir] % (Upper[ir] - Lower[ir])
    return x


def fullArray(a, D):
    r"""Fill or create array of length D, from value or value form a.

    Arguments:
        a (Union[int, float, numpy.ndarray], Iterable[Any]): Input values for fill.
        D (int): Length of new array.

    Returns:
        numpy.ndarray: Array filled with passed values or value.

    """

    A = []

    if isinstance(a, (int, float)):
        A = full(D, a)
    elif isinstance(a, (ndarray, list, tuple)):
        if len(a) == D:
            A = a if isinstance(a, ndarray) else asarray(a)
        elif len(a) > D:
            A = a[:D] if isinstance(a, ndarray) else asarray(a[:D])
        else:
            for i in range(int(ceil(float(D) / len(a)))):
                A.extend(a[:D if (D - i * len(a)) >= len(a) else D - i * len(a)])
            A = asarray(A)
    return A


def objects2array(objs):
    r"""Convert `Iterable` array or list to `NumPy` array.

    Args:
        objs (Iterable[Any]): Array or list to convert.

    Returns:
        numpy.ndarray: Array of objects.

    """

    a = empty(len(objs), dtype=object)
    for i, e in enumerate(objs):
        a[i] = e
    return a
