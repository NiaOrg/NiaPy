import numpy as np

__all__ = ['full_array', 'objects_to_array']


def full_array(a, dimension):
    r"""Fill or create array of length dimension, from value or value form a.

    Args:
        a (Union[int, float, numpy.ndarray, Iterable[Any]]): Input values for fill.
        dimension (int): Length of new array.

    Returns:
        numpy.ndarray: Array filled with passed values or value.

    """
    if isinstance(a, (int, float)):
        out = np.ones(dimension) * a
    elif isinstance(a, (np.ndarray, list, tuple)):
        if len(a) == dimension:
            out = np.asarray(a)
        elif len(a) > dimension:
            out = np.asarray(a[:dimension])
        else:
            out = np.tile(a, int(np.ceil(dimension / len(a))))[:dimension]
    else:
        raise TypeError('`a` must be a scalar or an Iterable.')
    return out


def objects_to_array(objs):
    r"""Convert `Iterable` array or list to `NumPy` array with dtype object.

    Args:
        objs (Iterable[Any]): Array or list to convert.

    Returns:
        numpy.ndarray: Array of objects.

    """
    a = np.empty(len(objs), dtype=object)
    for i, e in enumerate(objs):
        a[i] = e
    return a
