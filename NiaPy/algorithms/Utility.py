from NiaPy.algorithms import Algorithm
from NiaPy.algorithms.basic import DifferentialEvolution

class AlgorithmUtility:
    r"""Base class with string mappings to algorithms.

    Attributes:
        classes (Dict[str, Algorithm]): Mapping from stings to algorithms.

    """

    def __init__(self):
        r"""Initializing the algorithms."""

        self.algorithm_classes = {
            "DifferentialEvolution": DifferentialEvolution
        }

    def get_algorithm(self, algorithm):
        r"""Get the algorithm.

        Arguments:
            algorithm (Union[str, Algorithm]): String or class that represents the algorithm.

        Returns:
            Algorithm: Instance of an Algorithm.
        """

        if issubclass(type(algorithm), Algorithm):
            return algorithm
        elif algorithm in self.algorithm_classes:
            return self.algorithm_classes[algorithm]()
        else:
            raise TypeError("Passed algorithm is not defined!")