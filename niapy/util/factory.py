"""Factory functions for getting algorithms and benchmarks by name."""

__all__ = ['get_algorithm', 'get_benchmark']


def get_benchmark(name, *args, **kwargs):
    r"""Get benchmark by name.

    Args:
        name (str): Name of the benchmark.

    Returns:
        Benchmark: An instance of benchmark instantiated with \*args and \*\*kwargs.

    Raises:
        KeyError: If an invalid name is provided.

    """
    benchmark = _benchmark_options().pop(name.lower())
    return benchmark(*args, **kwargs)


def get_algorithm(name, *args, **kwargs):
    r"""Get algorithm by name.

    Args:
        name (str): Name of the algorithm.

    Returns:
        Algorithm: An instance of the algorithm instantiated \*args and \*\*kwargs.

    Raises:
        KeyError: If an invalid name is provided.

    """
    algorithm = _algorithm_options().pop(name)
    return algorithm(*args, **kwargs)


def _benchmark_options():
    import niapy.benchmarks as benchmarks

    benchmarks_dict = {
        'ackley': benchmarks.Ackley,
        'alpine1': benchmarks.Alpine1,
        'alpine2': benchmarks.Alpine2,
        'bent_cigar': benchmarks.BentCigar,
        'chung_reynolds': benchmarks.ChungReynolds,
        'cosine_mixture': benchmarks.CosineMixture,
        'csendes': benchmarks.Csendes,
        'discus': benchmarks.Discus,
        "dixon_price": benchmarks.DixonPrice,
        "elliptic": benchmarks.Elliptic,
        "conditioned_elliptic": benchmarks.Elliptic,
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
        "sum_squares": benchmarks.SumSquares,
        "trid": benchmarks.Trid,
        "weierstrass": benchmarks.Weierstrass,
        "whitley": benchmarks.Whitley,
        "zakharov": benchmarks.Zakharov,
    }
    return benchmarks_dict


def _algorithm_options():
    import niapy.algorithms.basic as basic_algorithms
    import niapy.algorithms.modified as modified_algorithms
    import niapy.algorithms.other as other_algorithms

    algorithms = {
        "BatAlgorithm": basic_algorithms.BatAlgorithm,
        "FireflyAlgorithm": basic_algorithms.FireflyAlgorithm,
        "DifferentialEvolution": basic_algorithms.DifferentialEvolution,
        "CrowdingDifferentialEvolution": basic_algorithms.CrowdingDifferentialEvolution,
        "AgingNpDifferentialEvolution": basic_algorithms.AgingNpDifferentialEvolution,
        "DynNpDifferentialEvolution": basic_algorithms.DynNpDifferentialEvolution,
        "MultiStrategyDifferentialEvolution": basic_algorithms.MultiStrategyDifferentialEvolution,
        "DynNpMultiStrategyDifferentialEvolution": basic_algorithms.DynNpMultiStrategyDifferentialEvolution,
        # "AgingNpMultiMutationDifferentialEvolution": basic_algorithms.AgingNpMultiMutationDifferentialEvolution,
        "FlowerPollinationAlgorithm": basic_algorithms.FlowerPollinationAlgorithm,
        "GreyWolfOptimizer": basic_algorithms.GreyWolfOptimizer,
        "GeneticAlgorithm": basic_algorithms.GeneticAlgorithm,
        "ArtificialBeeColonyAlgorithm": basic_algorithms.ArtificialBeeColonyAlgorithm,
        "ParticleSwarmAlgorithm": basic_algorithms.ParticleSwarmAlgorithm,
        "BareBonesFireworksAlgorithm": basic_algorithms.BareBonesFireworksAlgorithm,
        "CamelAlgorithm": basic_algorithms.CamelAlgorithm,
        "MonkeyKingEvolutionV1": basic_algorithms.MonkeyKingEvolutionV1,
        "MonkeyKingEvolutionV2": basic_algorithms.MonkeyKingEvolutionV2,
        "MonkeyKingEvolutionV3": basic_algorithms.MonkeyKingEvolutionV3,
        "EvolutionStrategy1p1": basic_algorithms.EvolutionStrategy1p1,
        "EvolutionStrategyMp1": basic_algorithms.EvolutionStrategyMp1,
        "EvolutionStrategyMpL": basic_algorithms.EvolutionStrategyMpL,
        "EvolutionStrategyML": basic_algorithms.EvolutionStrategyML,
        "CovarianceMatrixAdaptionEvolutionStrategy": basic_algorithms.CovarianceMatrixAdaptionEvolutionStrategy,
        "SineCosineAlgorithm": basic_algorithms.SineCosineAlgorithm,
        "GlowwormSwarmOptimization": basic_algorithms.GlowwormSwarmOptimization,
        "GlowwormSwarmOptimizationV1": basic_algorithms.GlowwormSwarmOptimizationV1,
        "GlowwormSwarmOptimizationV2": basic_algorithms.GlowwormSwarmOptimizationV2,
        "GlowwormSwarmOptimizationV3": basic_algorithms.GlowwormSwarmOptimizationV3,
        "HarmonySearch": basic_algorithms.HarmonySearch,
        "HarmonySearchV1": basic_algorithms.HarmonySearchV1,
        "KrillHerdV1": basic_algorithms.KrillHerdV1,
        "KrillHerdV2": basic_algorithms.KrillHerdV2,
        "KrillHerdV3": basic_algorithms.KrillHerdV3,
        "KrillHerdV4": basic_algorithms.KrillHerdV4,
        "KrillHerdV11": basic_algorithms.KrillHerdV11,
        "FireworksAlgorithm": basic_algorithms.FireworksAlgorithm,
        "EnhancedFireworksAlgorithm": basic_algorithms.EnhancedFireworksAlgorithm,
        "DynamicFireworksAlgorithm": basic_algorithms.DynamicFireworksAlgorithm,
        "DynamicFireworksAlgorithmGauss": basic_algorithms.DynamicFireworksAlgorithmGauss,
        "GravitationalSearchAlgorithm": basic_algorithms.GravitationalSearchAlgorithm,
        "MothFlameOptimizer": basic_algorithms.MothFlameOptimizer,
        "FishSchoolSearch": basic_algorithms.FishSchoolSearch,
        "CuckooSearch": basic_algorithms.CuckooSearch,
        "CoralReefsOptimization": basic_algorithms.CoralReefsOptimization,
        "ForestOptimizationAlgorithm": basic_algorithms.ForestOptimizationAlgorithm,
        "MonarchButterflyOptimization": basic_algorithms.MonarchButterflyOptimization,
        "HybridBatAlgorithm": modified_algorithms.HybridBatAlgorithm,
        "DifferentialEvolutionMTS": modified_algorithms.DifferentialEvolutionMTS,
        "DifferentialEvolutionMTSv1": modified_algorithms.DifferentialEvolutionMTSv1,
        "DynNpDifferentialEvolutionMTS": modified_algorithms.DynNpDifferentialEvolutionMTS,
        "DynNpDifferentialEvolutionMTSv1": modified_algorithms.DynNpDifferentialEvolutionMTSv1,
        "MultiStrategyDifferentialEvolutionMTS": modified_algorithms.MultiStrategyDifferentialEvolutionMTS,
        "MultiStrategyDifferentialEvolutionMTSv1": modified_algorithms.MultiStrategyDifferentialEvolutionMTSv1,
        "SelfAdaptiveDifferentialEvolution": modified_algorithms.SelfAdaptiveDifferentialEvolution,
        # "DynNpSelfAdaptiveDifferentialEvolutionAlgorithm": modified_algorithms.DynNpSelfAdaptiveDifferentialEvolutionAlgorithm,
        "MultiStrategySelfAdaptiveDifferentialEvolution": modified_algorithms.MultiStrategySelfAdaptiveDifferentialEvolution,
        # "AgingSelfAdaptiveDifferentialEvolution": modified_algorithms.AgingSelfAdaptiveDifferentialEvolution,
        "NelderMeadMethod": other_algorithms.NelderMeadMethod,
        "HillClimbAlgorithm": other_algorithms.HillClimbAlgorithm,
        "SimulatedAnnealing": other_algorithms.SimulatedAnnealing,
        "MultipleTrajectorySearch": other_algorithms.MultipleTrajectorySearch,
        "MultipleTrajectorySearchV1": other_algorithms.MultipleTrajectorySearchV1,
        "AnarchicSocietyOptimization": other_algorithms.AnarchicSocietyOptimization,
        "RandomSearch": other_algorithms.RandomSearch,
        "BacterialForagingOptimizationAlgorithm": basic_algorithms.BacterialForagingOptimizationAlgorithm
    }
    return algorithms
