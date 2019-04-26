# encoding=utf8

"""Implementation of various utility functions."""

import logging

from numpy import (
    ndarray,
    asarray,
    full,
    empty,
    where,
    random as rand,
    ceil,
    amin,
    amax
)

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
