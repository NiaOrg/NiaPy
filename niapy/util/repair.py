import numpy as np
from numpy.random import default_rng

__all__ = ['limit', 'limit_inverse', 'wang', 'rand', 'reflect']


def limit(x, lower, upper, **_kwargs):
    r"""Repair solution and put the solution in the random position inside of the bounds of problem.

    Args:
        x (numpy.ndarray): Solution to check and repair if needed.
        lower (numpy.ndarray): Lower bounds of search space.
        upper (numpy.ndarray): Upper bounds of search space.

    Returns:
        numpy.ndarray: Solution in search space.

    """
    return np.clip(x, lower, upper, out=x)


def limit_inverse(x, lower, upper, **_kwargs):
    r"""Repair solution and put the solution in the random position inside of the bounds of problem.

    Args:
        x (numpy.ndarray): Solution to check and repair if needed.
        lower (numpy.ndarray): Lower bounds of search space.
        upper (numpy.ndarray): Upper bounds of search space.

    Returns:
        numpy.ndarray: Solution in search space.

    """
    ir = np.where(x < lower)
    x[ir] = upper[ir]
    ir = np.where(x > upper)
    x[ir] = lower[ir]
    return x


def wang(x, lower, upper, **_kwargs):
    r"""Repair solution and put the solution in the random position inside of the bounds of problem.

    Args:
        x (numpy.ndarray): Solution to check and repair if needed.
        lower (numpy.ndarray): Lower bounds of search space.
        upper (numpy.ndarray): Upper bounds of search space.

    Returns:
        numpy.ndarray: Solution in search space.

    """
    ir = np.where(x < lower)
    x[ir] = np.amin([upper[ir], 2 * lower[ir] - x[ir]], axis=0)
    ir = np.where(x > upper)
    x[ir] = np.amax([lower[ir], 2 * upper[ir] - x[ir]], axis=0)
    return x


def rand(x, lower, upper, rng=None, **_kwargs):
    r"""Repair solution and put the solution in the random position inside of the bounds of problem.

    Args:
        x (numpy.ndarray): Solution to check and repair if needed.
        lower (numpy.ndarray): Lower bounds of search space.
        upper (numpy.ndarray): Upper bounds of search space.
        rng (numpy.random.Generator): Random generator.

    Returns:
        numpy.ndarray: Fixed solution.

    """
    rng = default_rng(rng)
    ir = np.where(x < lower)
    x[ir] = rng.uniform(lower[ir], upper[ir])
    ir = np.where(x > upper)
    x[ir] = rng.uniform(lower[ir], upper[ir])
    return x


def reflect(x, lower, upper, **_kwargs):
    r"""Repair solution and put the solution in search space with reflection of how much the solution violates a bound.

    Args:
        x (numpy.ndarray): Solution to be fixed.
        lower (numpy.ndarray): Lower bounds of search space.
        upper (numpy.ndarray): Upper bounds of search space.

    Returns:
        numpy.ndarray: Fix solution.

    """
    ir = np.where(x > upper)
    x[ir] = lower[ir] + x[ir] % (upper[ir] - lower[ir])
    ir = np.where(x < lower)
    x[ir] = lower[ir] + x[ir] % (upper[ir] - lower[ir])
    return x
