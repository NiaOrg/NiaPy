# encoding=utf8

"""Implementation of benchmarks utility function."""

from NiaPy import (
    benchmarks,
    algorithms
)
from NiaPy.util.utility import explore_package_for_classes


class Factory:
    r"""Base class with string mappings to benchmarks and algorithms.

    Author:
        Klemen Berkovic

    Date:
        2020

    License:
        MIT

    Attributes:
        benchmark_classes (Dict[str, Benchmark]): Mapping for fetching Benchmark classes.
        algorithm_classes (Dict[str, Algorithm]): Mapping for fetching Algorithm classes.
    """

    def __init__(self):
        r"""Init benchmark classes."""
        self.benchmark_classes = self.__init_factory(benchmarks, benchmarks.Benchmark)
        self.algorithm_classes = self.__init_factory(algorithms.basic, algorithms.Algorithm)
        self.algorithm_classes.update(self.__init_factory(algorithms.modified, algorithms.Algorithm))
        self.algorithm_classes.update(self.__init_factory(algorithms.other, algorithms.Algorithm))

    def __init_factory(self, module, dtype):
        tmp = {}
        for cc in explore_package_for_classes(module, dtype).values():
            for val in cc.Name:
                tmp[val] = cc
        return tmp

    def get_benchmark(self, benchmark, **kwargs):
        r"""Get the optimization problem.

        Arguments:
            benchmark (Union[str, Benchmark]): String or class that represents the optimization problem.
            kwargs (dict): Additional arguments for passed benchmark.

        Raises:
            TypeError: If benchmark is not defined.

        Returns:
            Benchmark: Optimization function with limits.
        """

        if isinstance(benchmark, benchmarks.Benchmark):
            return benchmark
        elif issubclass(type(benchmark), benchmarks.Benchmark):
            return benchmark(**kwargs)
        elif benchmark in self.benchmark_classes.keys():
            return self.benchmark_classes[benchmark](**kwargs)
        else:
            raise TypeError("Passed benchmark '%s' is not defined!" % benchmark)

    def get_algorithm(self, algorithm, **kwargs):
        r"""Get the algorithm for optimization.

        Args:
            algorithm (Union[str, Algorithm]): Algorithm to use.
            kwargs (dict): Additional arguments for algorithm.

        Raises:
            TypeError: If algorithm is not defined.

        Returns:
            Algorithm: Initialized algorithm.
        """

        if isinstance(algorithm, algorithms.Algorithm):
            return algorithm
        elif issubclass(type(algorithms), algorithms.Algorithm):
            return algorithm(**kwargs)
        elif algorithm in self.algorithm_classes.keys():
            return self.algorithm_classes[algorithm](**kwargs)
        else:
            raise TypeError("Passed algorithm '%s' is not defined!" % algorithm)

    @classmethod
    def __raiseLowerAndUpperNotDefined(cls):
        r"""Trow exception if lower and upper bounds are not defined in benchmark.

        Raises:
            TypeError: Type error.

        """

        raise TypeError("Upper and Lower value must be defined!")
