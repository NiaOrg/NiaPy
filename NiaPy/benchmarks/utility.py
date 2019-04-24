"""Implementation of benchmarks utility function."""

from NiaPy.benchmarks import (
    Ackley,
    Alpine1,
    Alpine2,
    BentCigar,
    ChungReynolds,
    CosineMixture,
    Csendes,
    Discus,
    DixonPrice,
    Elliptic,
    ExpandedGriewankPlusRosenbrock,
    ExpandedSchaffer,
    Griewank,
    HappyCat,
    HGBat,
    Infinity,
    Katsuura,
    Levy,
    Michalewichz,
    ModifiedSchwefel,
    Perm,
    Pinter,
    Powell,
    Qing,
    Quintic,
    Rastrigin,
    Ridge,
    Rosenbrock,
    Salomon,
    SchafferN2,
    SchafferN4,
    SchumerSteiglitz,
    Schwefel,
    Schwefel221,
    Schwefel222,
    Sphere,
    Sphere2,
    Sphere3,
    Step,
    Step2,
    Step3,
    Stepint,
    StyblinskiTang,
    SumSquares,
    Trid,
    Weierstrass,
    Whitley,
    Zakharov,
    Benchmark
)


class Utility:
    r"""Base class with string mappings to benchmarks.

    Attributes:
        classes (Dict[str, Benchmark]): Mapping from stings to benchmark.

    """

    def __init__(self):
        r"""Initialize benchmarks."""

        self.benchmark_classes = {
            "ackley": Ackley,
            "alpine1": Alpine1,
            "alpine2": Alpine2,
            "bentcigar": BentCigar,
            "chungReynolds": ChungReynolds,
            "cosinemixture": CosineMixture,
            "csendes": Csendes,
            "discus": Discus,
            "dixonprice": DixonPrice,
            "conditionedellptic": Elliptic,
            "elliptic": Elliptic,
            "expandedgriewankplusrosenbrock": ExpandedGriewankPlusRosenbrock,
            "expandedschaffer": ExpandedSchaffer,
            "griewank": Griewank,
            "happyCat": HappyCat,
            "hgbat": HGBat,
            "infinity": Infinity,
            "katsuura": Katsuura,
            "levy": Levy,
            "michalewicz": Michalewichz,
            "modifiedscwefel": ModifiedSchwefel,
            "perm": Perm,
            "pinter": Pinter,
            "powell": Powell,
            "qing": Qing,
            "quintic": Quintic,
            "rastrigin": Rastrigin,
            "ridge": Ridge,
            "rosenbrock": Rosenbrock,
            "salomon": Salomon,
            "schaffer2": SchafferN2,
            "schaffer4": SchafferN4,
            "schumerSteiglitz": SchumerSteiglitz,
            "schwefel": Schwefel,
            "schwefel221": Schwefel221,
            "schwefel222": Schwefel222,
            "sphere": Sphere,
            "sphere2": Sphere2,
            "sphere3": Sphere3,
            "step": Step,
            "step2": Step2,
            "step3": Step3,
            "stepint": Stepint,
            "styblinskiTang": StyblinskiTang,
            "sumSquares": SumSquares,
            "trid": Trid,
            "weierstrass": Weierstrass,
            "whitley": Whitley,
            "zakharov": Zakharov
        }

        self.algorithm_classes = {}

    def get_benchmark(self, benchmark):
        r"""Get the optimization problem.

        Arguments:
            benchmark (Union[str, Benchmark]): String or class that represents the optimization problem.

        Returns:
            Benchmark: Optimization function with limits.

        """

        if issubclass(type(benchmark), Benchmark):
            return benchmark
        elif benchmark in self.benchmark_classes.keys():
            return self.benchmark_classes[benchmark]()
        else:
            raise TypeError("Passed benchmark is not defined!")

    @classmethod
    def __raiseLowerAndUpperNotDefined(cls):
        r"""Trow exception if lower and upper bounds are not defined in benchmark.

        Raises:
            TypeError: Type error.

        """

        raise TypeError("Upper and Lower value must be defined!")
