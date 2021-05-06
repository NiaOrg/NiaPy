import math

import numpy as np

__all__ = ['levy_flight']


def levy_flight(rng, alpha=0.01, beta=1.5, size=None):
    """Compute levy flight.

    Args:
        alpha (float): Scaling factor.
        beta (float): Stability parameter in range (0, 2).
        size (Optional[Union[int, Iterable[int]]]: Output size.
        rng (numpy.random.Generator): Random number generator.

    Returns:
        Union[float, numpy.ndarray]: Sample(s) from a truncated levy distribution.

    """
    sigma = (math.gamma(1 + beta) * np.sin(np.pi * beta / 2) / (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = rng.normal(0, sigma, size)
    v = rng.normal(0, 1, size)
    sample = alpha * u / (np.abs(v) ** (1 / beta))
    return sample
