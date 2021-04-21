import numpy as np

__all__ = ['full_array', 'objects_to_array']


def full_array(a, dimension):
    r"""Fill or create array of length dimension, from value or value form a.

    Arguments:
        a (Union[int, float, numpy.ndarray], Iterable[Any]): Input values for fill.
        dimension (int): Length of new array.

    Returns:
        numpy.ndarray: Array filled with passed values or value.

    """

    out = []

    if isinstance(a, (int, float)):
        out = np.full(dimension, a)
    elif isinstance(a, (np.ndarray, list, tuple)):
        if len(a) == dimension:
            out = a if isinstance(a, np.ndarray) else np.asarray(a)
        elif len(a) > dimension:
            out = a[:dimension] if isinstance(a, np.ndarray) else np.asarray(a[:dimension])
        else:
            for i in range(int(np.ceil(float(dimension) / len(a)))):
                out.extend(a[:dimension if (dimension - i * len(a)) >= len(a) else dimension - i * len(a)])
            out = np.asarray(out)
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
