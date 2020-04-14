# encoding=utf8

"""Implementation of various utility functions."""

import numpy as np
from numpy import random as rand

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
        a (Union[int, float, numpy.ndarray], Iterable[Any]): Input values for fill.
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


def groupdatabylabel(data, labels, lt):
    r"""Get gruped data based on labels.

    Args:
        data (numpy.ndarray): Dataset of individuals.
        labels (numpy.ndarray): Labels of individuals.
        lt (LabelEncoder): Label transformer.

    Returns:
        numpy.ndarray: Gruped data based on labels.
    """

    G = [[] for _ in range(len(np.unique(labels)))]
    for i, e in enumerate(data):
        G[lt.transform([labels[i]])[0]].append(e)
    return np.asarray(G)

def clusters2labels(G_c, G_l):
    r"""Get mapping from clusters to classes/labels.

    Args:
        G_c (numpy.ndarray): Clusters centers.
        G_l (numpy.ndarray): Centers of labeld data.

    Returns:
        numpy.ndarray: Labels maped to clusters.
    """

    a, G_ll, inds = np.full(len(G_c), -1), [gl for gl in G_l], [i for i in range(len(G_l))]
    for i, gc in enumerate(G_c):
        e = np.argmin([np.sqrt(np.sum((gc - np.mean(gl, axis=0)) ** 2)) for gl in G_ll])
        a[i] = inds[e]
        del G_ll[e]
        del inds[e]
    return a

def classifie(o, C):
    r"""Classfie individua based on centers.

    Args:
        o (numpy.ndarray): Individual to classifie.
        C (numpy.ndarray): Center of clusters.

    Returns:
        int: Index of class.
    """

    return np.argmin([np.sqrt(np.sum((o - c) ** 2)) for c in C])
