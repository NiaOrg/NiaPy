# encoding=utf8

"""Implementation of various utility functions."""

import inspect

import numpy as np
from numpy import random as rand

__all__ = [
    "limit_repair",
    "limitInversRepair",
    "objects2array",
    "wangRepair",
    "randRepair",
    "fullArray",
    "reflectRepair",
    "explore_package_for_classes"
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

    ir = np.where(x < Lower)
    x[ir] = Lower[ir]
    ir = np.where(x > Upper)
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

    ir = np.where(x < Lower)
    x[ir] = Upper[ir]
    ir = np.where(x > Upper)
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

    ir = np.where(x < Lower)
    x[ir] = np.amin([Upper[ir], 2 * Lower[ir] - x[ir]], axis=0)
    ir = np.where(x > Upper)
    x[ir] = np.amax([Lower[ir], 2 * Upper[ir] - x[ir]], axis=0)
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

    ir = np.where(x < Lower)
    x[ir] = rnd.uniform(Lower[ir], Upper[ir])
    ir = np.where(x > Upper)
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

    ir = np.where(x > Upper)
    x[ir] = Lower[ir] + x[ir] % (Upper[ir] - Lower[ir])
    ir = np.where(x < Lower)
    x[ir] = Lower[ir] + x[ir] % (Upper[ir] - Lower[ir])
    return x


def fullArray(a, D):
    r"""Fill or create array of length D, from value or value form a.

    Arguments:
        a (Union[int, float, Any, numpy.ndarray, Iterable[Union[int, float, Any]]]): Input values for fill.
        D (int): Length of new array.

    Returns:
        numpy.ndarray: Array filled with passed values or value.

    """

    A = []

    if isinstance(a, (int, float)):
        A = np.full(D, a)
    elif isinstance(a, (np.ndarray, list, tuple)):
        if len(a) == D:
            A = a if isinstance(a, np.ndarray) else np.asarray(a)
        elif len(a) > D:
            A = a[:D] if isinstance(a, np.ndarray) else np.asarray(a[:D])
        else:
            for i in range(int(np.ceil(float(D) / len(a)))):
                A.extend(a[:D if (D - i * len(a)) >= len(a) else D - i * len(a)])
            A = np.asarray(A)
    return A


def objects2array(objs):
    r"""Convert `Iterable` array or list to `NumPy` array.

    Args:
        objs (Iterable[Any]): Array or list to convert.

    Returns:
        numpy.ndarray: Array of objects.

    """

    a = np.empty(len(objs), dtype=object)
    for i, e in enumerate(objs):
        a[i] = e
    return a


def explore_package_for_classes(module, stype=object, subdir=False):
    r"""Explore the python package for classes.

    Args:
        module (Any): Module to inspect for classes.
        stype (Union[class, type]): Super type of search.
        subdir (bool): Go thrue

    Returns:
        Dict[str, Any]: Mapping for classes in package.
    """

    tmp = {}
    for key, data in inspect.getmembers(module, inspect.isclass):
        if isinstance(data, stype) or issubclass(data, stype):
            tmp[key] = data
    return tmp
