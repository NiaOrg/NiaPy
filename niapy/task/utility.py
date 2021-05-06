# encoding=utf8

"""Implementation of benchmarks utility function."""

from niapy import benchmarks


class Utility:
    r"""Base class with string mappings to benchmarks.

    Attributes:
        benchmark_classes (Dict[str, Benchmark]): Mapping from stings to benchmark.

    """

    def __init__(self):
        r"""Initialize benchmarks."""
        self.benchmark_classes = {
            "ackley": benchmarks.Ackley,
            "alpine1": benchmarks.Alpine1,
            "alpine2": benchmarks.Alpine2,
            "bent_cigar": benchmarks.BentCigar,
            "chung_reynolds": benchmarks.ChungReynolds,
            "cosine_mixture": benchmarks.CosineMixture,
            "csendes": benchmarks.Csendes,
            "discus": benchmarks.Discus,
            "dixon_price": benchmarks.DixonPrice,
            "conditioned_elliptic": benchmarks.Elliptic,
            "elliptic": benchmarks.Elliptic,
            "expanded_griewank_plus_rosenbrock": benchmarks.ExpandedGriewankPlusRosenbrock,
            "expanded_schaffer": benchmarks.ExpandedSchaffer,
            "griewank": benchmarks.Griewank,
            "happy_cat": benchmarks.HappyCat,
            "hgbat": benchmarks.HGBat,
            "infinity": benchmarks.Infinity,
            "katsuura": benchmarks.Katsuura,
            "levy": benchmarks.Levy,
            "michalewicz": benchmarks.Michalewichz,
            "modified_schwefel": benchmarks.ModifiedSchwefel,
            "perm": benchmarks.Perm,
            "pinter": benchmarks.Pinter,
            "powell": benchmarks.Powell,
            "qing": benchmarks.Qing,
            "quintic": benchmarks.Quintic,
            "rastrigin": benchmarks.Rastrigin,
            "ridge": benchmarks.Ridge,
            "rosenbrock": benchmarks.Rosenbrock,
            "salomon": benchmarks.Salomon,
            "schaffer2": benchmarks.SchafferN2,
            "schaffer4": benchmarks.SchafferN4,
            "schumer_steiglitz": benchmarks.SchumerSteiglitz,
            "schwefel": benchmarks.Schwefel,
            "schwefel221": benchmarks.Schwefel221,
            "schwefel222": benchmarks.Schwefel222,
            "sphere": benchmarks.Sphere,
            "sphere2": benchmarks.Sphere2,
            "sphere3": benchmarks.Sphere3,
            "step": benchmarks.Step,
            "step2": benchmarks.Step2,
            "step3": benchmarks.Step3,
            "stepint": benchmarks.Stepint,
            "styblinski_tang": benchmarks.StyblinskiTang,
            "sumSquares": benchmarks.SumSquares,
            "trid": benchmarks.Trid,
            "weierstrass": benchmarks.Weierstrass,
            "whitley": benchmarks.Whitley,
            "zakharov": benchmarks.Zakharov
        }

        self.algorithm_classes = {}

    def get_benchmark(self, benchmark):
        r"""Get the optimization problem.

        Args:
            benchmark (Union[str, Benchmark]): String or class that represents the optimization problem.

        Returns:
            Benchmark: Optimization function with limits.

        """
        if issubclass(type(benchmark), benchmarks.Benchmark) or isinstance(benchmark, benchmarks.Benchmark):
            return benchmark
        elif benchmark in self.benchmark_classes.keys():
            return self.benchmark_classes[benchmark]()
        else:
            raise TypeError("Passed benchmark is not defined!")
