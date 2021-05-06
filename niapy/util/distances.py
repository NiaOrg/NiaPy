import numpy as np

__all__ = ['euclidean']


def euclidean(u, v):
    """Compute the euclidean distance between two numpy arrays.

    Args:
        u (numpy.ndarray): Input array.
        v (numpy.ndarray): Input array.

    Returns:
        float: Euclidean distance between u and v.

    """
    return np.sqrt(np.sum(np.square(u - v), axis=-1))
