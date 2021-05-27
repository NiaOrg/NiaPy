"""Factory functions for getting algorithms and problems by name."""

__all__ = ['get_algorithm', 'get_problem']


def get_problem(name, *args, **kwargs):
    r"""Get problem by name.

    Args:
        name (str): Name of the problem.

    Returns:
        Problem: An instance of Problem, instantiated with \*args and \*\*kwargs.

    Raises:
        KeyError: If an invalid name is provided.

    """
    problem = _problem_options().pop(name.lower())
    return problem(*args, **kwargs)


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


def _problem_options():
    import niapy.problems as problems

    problems_dict = {
        'ackley': problems.Ackley,
        'alpine1': problems.Alpine1,
        'alpine2': problems.Alpine2,
        'bent_cigar': problems.BentCigar,
        'chung_reynolds': problems.ChungReynolds,
        'cosine_mixture': problems.CosineMixture,
        'csendes': problems.Csendes,
        'discus': problems.Discus,
        "dixon_price": problems.DixonPrice,
        "elliptic": problems.Elliptic,
        "conditioned_elliptic": problems.Elliptic,
        "expanded_griewank_plus_rosenbrock": problems.ExpandedGriewankPlusRosenbrock,
        "expanded_schaffer": problems.ExpandedSchaffer,
        "griewank": problems.Griewank,
        "happy_cat": problems.HappyCat,
        "hgbat": problems.HGBat,
        "infinity": problems.Infinity,
        "katsuura": problems.Katsuura,
        "levy": problems.Levy,
        "michalewicz": problems.Michalewichz,
        "modified_schwefel": problems.ModifiedSchwefel,
        "perm": problems.Perm,
        "pinter": problems.Pinter,
        "powell": problems.Powell,
        "qing": problems.Qing,
        "quintic": problems.Quintic,
        "rastrigin": problems.Rastrigin,
        "ridge": problems.Ridge,
        "rosenbrock": problems.Rosenbrock,
        "salomon": problems.Salomon,
        "schaffer2": problems.SchafferN2,
        "schaffer4": problems.SchafferN4,
        "schumer_steiglitz": problems.SchumerSteiglitz,
        "schwefel": problems.Schwefel,
        "schwefel221": problems.Schwefel221,
        "schwefel222": problems.Schwefel222,
        "sphere": problems.Sphere,
        "sphere2": problems.Sphere2,
        "sphere3": problems.Sphere3,
        "step": problems.Step,
        "step2": problems.Step2,
        "step3": problems.Step3,
        "stepint": problems.Stepint,
        "styblinski_tang": problems.StyblinskiTang,
        "sum_squares": problems.SumSquares,
        "trid": problems.Trid,
        "weierstrass": problems.Weierstrass,
        "whitley": problems.Whitley,
        "zakharov": problems.Zakharov,
    }
    return problems_dict


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
