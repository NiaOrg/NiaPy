# encoding=utf8
"""Anarchic Society Optimization algorithm."""

import numpy as np

from niapy.algorithms.algorithm import Algorithm
from niapy.util.array import full_array


def euclidean(u, v):
    r"""Calculate Euclidean distance between two vectors.

    Implemented with numpy only, since scipy is not a NiaPy dependency.

    Args:
        u (numpy.ndarray): First vector.
        v (numpy.ndarray): Second vector.

    Returns
    -------
        float: Euclidean distance between u and v.

    """
    return np.sqrt(np.sum((np.asarray(u) - np.asarray(v)) ** 2))


__all__ = [
    'AnarchicSocietyOptimization',
    'elitism',
    'sequential',
    'crossover',
]


def _mp_c(x, f, cr, mp, rng):
    r"""
    Get new position based on fickleness (Fickleness Index strategy).

    Args:
        x (numpy.ndarray): Current individual's position.
        f (float): Scale factor.
        cr (float): Crossover probability.
        mp (float): Fickleness index value.
        rng (numpy.random.Generator): Random number generator.

    Returns
    -------
        numpy.ndarray: New position.

    """
    xn = x.copy()
    if mp < 0.5:
        b = np.sort(rng.choice(len(x), 2, replace=False))
        xn[b[0]:b[1]] = xn[b[0]:b[1]] + f * rng.standard_normal(b[1] - b[0])
    else:
        mask = rng.random(len(x)) < cr
        xn[mask] = xn[mask] + f * rng.standard_normal(np.sum(mask))
    return xn


def _mp_s(x, xr, xb, cr, mp, rng):
    r"""
    Get new position based on external irregularity (External Irregularity Index strategy).

    Args:
        x (numpy.ndarray): Current individual's position.
        xr (numpy.ndarray): Random individual's position.
        xb (numpy.ndarray): Global best individual's position.
        cr (float): Crossover probability.
        mp (float): External irregularity index value.
        rng (numpy.random.Generator): Random number generator.

    Returns
    -------
        numpy.ndarray: New position.

    """
    xn = x.copy()
    if mp < 0.25:
        b = np.sort(rng.choice(len(x), 2, replace=False))
        xn[b[0]:b[1]] = xb[b[0]:b[1]]
    elif mp < 0.5:
        mask = rng.random(len(x)) < cr
        xn[mask] = xb[mask]
    elif mp < 0.75:
        b = np.sort(rng.choice(len(x), 2, replace=False))
        xn[b[0]:b[1]] = xr[b[0]:b[1]]
    else:
        mask = rng.random(len(x)) < cr
        xn[mask] = xr[mask]
    return xn


def _mp_p(x, xpb, cr, mp, rng):
    r"""
    Get new position based on internal irregularity (Internal Irregularity Index strategy).

    Args:
        x (numpy.ndarray): Current individual's position.
        xpb (numpy.ndarray): Individual's personal best position.
        cr (float): Crossover probability.
        mp (float): Internal irregularity index value.
        rng (numpy.random.Generator): Random number generator.

    Returns
    -------
        numpy.ndarray: New position.

    """
    xn = x.copy()
    if mp < 0.5:
        b = np.sort(rng.choice(len(x), 2, replace=False))
        xn[b[0]:b[1]] = xpb[b[0]:b[1]]
    else:
        mask = rng.random(len(x)) < cr
        xn[mask] = xpb[mask]
    return xn


def elitism(x, xpb, xb, xr, mp_c, mp_s, mp_p, f, cr, task, rng):
    r"""
    Select the best of all three movement strategies.

    Args:
        x (numpy.ndarray): Individual's current position.
        xpb (numpy.ndarray): Individual's personal best position.
        xb (numpy.ndarray): Global best position.
        xr (numpy.ndarray): Random individual's position.
        mp_c (float): Fickleness index value.
        mp_s (float): External irregularity index value.
        mp_p (float): Internal irregularity index value.
        f (float): Scale factor.
        cr (float): Crossover probability.
        task (Task): Optimization task.
        rng (numpy.random.Generator): Random number generator.

    Returns
    -------
        Tuple[numpy.ndarray, float]:
            1. New position of individual.
            2. New position's fitness value.

    """
    candidates = [
        task.repair(_mp_c(x, f, cr, mp_c, rng), rng=rng),
        task.repair(_mp_s(x, xr, xb, cr, mp_s, rng), rng=rng),
        task.repair(_mp_p(x, xpb, cr, mp_p, rng), rng=rng),
    ]
    fitnesses = np.apply_along_axis(task.eval, 1, candidates)
    best_idx = np.argmin(fitnesses)
    return candidates[best_idx], fitnesses[best_idx]


def sequential(x, xpb, xb, xr, mp_c, mp_s, mp_p, f, cr, task, rng):
    r"""
    Sequentially apply all three movement strategies.

    Args:
        x (numpy.ndarray): Individual's current position.
        xpb (numpy.ndarray): Individual's personal best position.
        xb (numpy.ndarray): Global best position.
        xr (numpy.ndarray): Random individual's position.
        mp_c (float): Fickleness index value.
        mp_s (float): External irregularity index value.
        mp_p (float): Internal irregularity index value.
        f (float): Scale factor.
        cr (float): Crossover probability.
        task (Task): Optimization task.
        rng (numpy.random.Generator): Random number generator.

    Returns
    -------
        Tuple[numpy.ndarray, float]:
            1. New position.
            2. New position's fitness value.

    """
    xn = task.repair(
        _mp_s(
            _mp_p(_mp_c(x, f, cr, mp_c, rng), xpb, cr, mp_p, rng),
            xr, xb, cr, mp_s, rng,
        ),
        rng=rng,
    )
    return xn, task.eval(xn)


def crossover(x, xpb, xb, xr, mp_c, mp_s, mp_p, f, cr, task, rng):
    r"""
    Create a crossover over all three movement strategies.

    Args:
        x (numpy.ndarray): Individual's current position.
        xpb (numpy.ndarray): Individual's personal best position.
        xb (numpy.ndarray): Global best position.
        xr (numpy.ndarray): Random individual's position.
        mp_c (float): Fickleness index value.
        mp_s (float): External irregularity index value.
        mp_p (float): Internal irregularity index value.
        f (float): Scale factor.
        cr (float): Crossover probability.
        task (Task): Optimization task.
        rng (numpy.random.Generator): Random number generator.

    Returns
    -------
        Tuple[numpy.ndarray, float]:
            1. New position.
            2. New position's fitness value.

    """
    candidates = [
        task.repair(_mp_c(x, f, cr, mp_c, rng), rng=rng),
        task.repair(_mp_s(x, xr, xb, cr, mp_s, rng), rng=rng),
        task.repair(_mp_p(x, xpb, cr, mp_p, rng), rng=rng),
    ]
    xn = np.array([
        candidates[rng.integers(len(candidates))][i] if rng.random() < cr else x[i]
        for i in range(len(x))
    ])
    return xn, task.eval(xn)


class AnarchicSocietyOptimization(Algorithm):
    r"""Implementation of Anarchic Society Optimization algorithm.

    Algorithm:
        Anarchic Society Optimization

    Date:
        2018

    Authors:
        Klemen Berkovič

    License:
        MIT

    Reference paper:
        Ahmadi-Javid, Amir. "Anarchic Society Optimization: A human-inspired method."
        Evolutionary Computation (CEC), 2011 IEEE Congress on. IEEE, 2011.

    Attributes:
        Name (List[str]): List of strings representing algorithm names.
        alpha (List[float]): Factor for fickleness index function :math:`\in [0, 1]`.
        gamma (List[float]): Factor for external irregularity index function :math:`\in [0, \infty)`.
        theta (List[float]): Factor for internal irregularity index function :math:`\in [0, \infty)`.
        d (Callable): Distance function for fitness values.
        dn (Callable): Distance function for positions in search space.
        nl (float): Normalized neighbourhood range :math:`\in (0, 1]`.
        f (float): Mutation scale factor.
        cr (float): Crossover probability :math:`\in [0, 1]`.
        combination (Callable): Strategy for combining movement operators.

    See Also:
        * :class:`niapy.algorithms.Algorithm`

    Examples:
        >>> from niapy.algorithms.other import AnarchicSocietyOptimization
        >>> from niapy.task import Task
        >>> from niapy.benchmarks import Sphere
        >>> task = Task(problem=Sphere(dimension=10), max_evals=10000)
        >>> algo = AnarchicSocietyOptimization(population_size=43)
        >>> best, best_fit = algo.run(task)
        >>> print(best_fit)
    """

    Name = ['AnarchicSocietyOptimization', 'ASO']

    def __init__(self, population_size=43, alpha=(1, 0.83), gamma=(1.17, 0.56),
                 theta=(0.932, 0.832), d=euclidean, dn=euclidean, nl=1.0,
                 f=1.2, cr=0.25, combination=elitism, *args, **kwargs):
        r"""Initialize AnarchicSocietyOptimization algorithm."""
        super().__init__(*args, population_size=population_size, **kwargs)
        self.set_parameters(population_size=population_size, alpha=alpha, gamma=gamma,
                            theta=theta, d=d, dn=dn, nl=nl, f=f, cr=cr,
                            combination=combination, **kwargs)

    @staticmethod
    def info():
        r"""
        Get basic information about the algorithm.

        Returns
        -------
            str: Basic information.

        """
        return r"""Ahmadi-Javid, Amir. "Anarchic Society Optimization: A human-inspired method."
        Evolutionary Computation (CEC), 2011 IEEE Congress on. IEEE, 2011."""

    @staticmethod
    def type_parameters():
        r"""
        Get functions for checking parameter values.

        Returns
        -------
            Dict[str, Callable]:
                * alpha: Check alpha parameter.
                * gamma: Check gamma parameter.
                * theta: Check theta parameter.
                * nl: Check neighbourhood range.
                * f: Check scale factor.
                * cr: Check crossover probability.

        """
        d = Algorithm.type_parameters()
        d.update({
            'alpha': lambda x: True,
            'gamma': lambda x: True,
            'theta': lambda x: True,
            'nl': lambda x: isinstance(x, float) and 0 < x <= 1,
            'f': lambda x: isinstance(x, (int, float)) and x > 0,
            'cr': lambda x: isinstance(x, float) and 0 <= x <= 1,
        })
        return d

    def set_parameters(self, population_size=43, alpha=(1, 0.83), gamma=(1.17, 0.56),
                       theta=(0.932, 0.832), d=euclidean, dn=euclidean, nl=1.0,
                       f=1.2, cr=0.25, combination=elitism, *args, **kwargs):
        r"""
        Set algorithm parameters.

        Args:
            population_size (Optional[int]): Number of individuals in population.
            alpha (Optional[List[float]]): Factors for fickleness index :math:`\in [0, 1]`.
            gamma (Optional[List[float]]): Factors for external irregularity index :math:`\in [0, \infty)`.
            theta (Optional[List[float]]): Factors for internal irregularity index :math:`\in [0, \infty)`.
            d (Optional[Callable]): Distance function for fitness values.
            dn (Optional[Callable]): Distance function for positions.
            nl (Optional[float]): Normalized neighbourhood range :math:`\in (0, 1]`.
            f (Optional[float]): Mutation scale factor.
            cr (Optional[float]): Crossover probability :math:`\in [0, 1]`.
            combination (Optional[Callable]): Movement strategy combination function.
                Choose from :func:`elitism`, :func:`sequential`, :func:`crossover`.

        See Also
        --------
            * :func:`niapy.algorithms.Algorithm.set_parameters`

        """
        super().set_parameters(*args, population_size=population_size, **kwargs)
        self.alpha = alpha
        self.gamma = gamma
        self.theta = theta
        self.d = d
        self.dn = dn
        self.nl = nl
        self.f = f
        self.cr = cr
        self.combination = combination

    def get_parameters(self):
        r"""
        Get parameter values of the algorithm.

        Returns
        -------
            Dict[str, Any]: Parameter values.

        """
        params = super().get_parameters()
        params.update({
            'alpha': self.alpha,
            'gamma': self.gamma,
            'theta': self.theta,
            'nl': self.nl,
            'f': self.f,
            'cr': self.cr,
            'combination': self.combination,
        })
        return params

    def _init_params(self, population_size):
        r"""
        Expand scalar or short list parameters to arrays of length population_size.

        Args:
            population_size (int): Population size.

        Returns
        -------
            Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
                1. Alpha values per individual.
                2. Gamma values per individual.
                3. Theta values per individual.

        """
        return (
            full_array(self.alpha, population_size),
            full_array(self.gamma, population_size),
            full_array(self.theta, population_size),
        )

    def _fi(self, x_f, xpb_f, xb_f, alpha):
        r"""
        Calculate fickleness index.

        A high fickleness index means the individual is likely to change its
        behaviour (move away from its current position).

        Args:
            x_f (float): Individual's current fitness value.
            xpb_f (float): Individual's personal best fitness value.
            xb_f (float): Global best fitness value.
            alpha (float): Fickleness scaling factor.

        Returns
        -------
            float: Fickleness index value.

        """
        return 1.0 - alpha * xb_f / x_f - (1.0 - alpha) * xpb_f / x_f

    def _ei(self, x_f, xnb_f, gamma):
        r"""
        Calculate external irregularity index.

        Measures how different the individual's fitness is from its neighbour's.

        Args:
            x_f (float): Individual's current fitness value.
            xnb_f (float): Neighbour's fitness value.
            gamma (float): External irregularity scaling factor.

        Returns
        -------
            float: External irregularity index value.

        """
        return 1.0 - np.exp(-gamma * abs(x_f - xnb_f))

    def _ii(self, x_f, xpb_f, theta):
        r"""
        Calculate internal irregularity index.

        Measures how different the individual's current fitness is from its
        personal best fitness.

        Args:
            x_f (float): Individual's current fitness value.
            xpb_f (float): Individual's personal best fitness value.
            theta (float): Internal irregularity scaling factor.

        Returns
        -------
            float: Internal irregularity index value.

        """
        return 1.0 - np.exp(-theta * abs(x_f - xpb_f))

    def _get_best_neighbor(self, i, population, population_fitness, rs):
        r"""
        Find the best neighbour of individual *i* within the neighbourhood radius.

        Neighbourhood is defined by ``self.nl`` (normalised distance threshold)
        and ``self.dn`` (distance function in solution space).

        Args:
            i (int): Index of the focal individual.
            population (numpy.ndarray): Current population positions.
            population_fitness (numpy.ndarray): Current population fitness values.
            rs (float): Search-space diameter used for normalisation.

        Returns
        -------
            int: Index of the best neighbour within the neighbourhood.

        """
        distances = np.array([
            self.dn(population[i], population[j]) / rs
            for j in range(len(population))
        ])
        neighbor_indices = np.where(distances <= self.nl)[0]
        return neighbor_indices[np.argmin(population_fitness[neighbor_indices])]

    def _update_personal_best(self, population, population_fitness, personal_best, personal_best_fitness):
        r"""
        Update personal best positions for all individuals.

        Args:
            population (numpy.ndarray): Current population positions.
            population_fitness (numpy.ndarray): Current population fitness values.
            personal_best (numpy.ndarray): Current personal best positions.
            personal_best_fitness (numpy.ndarray): Current personal best fitness values.

        Returns
        -------
            Tuple[numpy.ndarray, numpy.ndarray]:
                1. Updated personal best positions.
                2. Updated personal best fitness values.

        """
        improved = population_fitness < personal_best_fitness
        personal_best[improved] = population[improved].copy()
        personal_best_fitness[improved] = population_fitness[improved].copy()
        return personal_best, personal_best_fitness

    def init_population(self, task):
        r"""
        Initialize population and algorithm state.

        Args:
            task (Task): Optimization task.

        Returns
        -------
            Tuple[numpy.ndarray, numpy.ndarray, Dict[str, Any]]:
                1. Initial population positions.
                2. Initial population fitness values.
                3. Additional state:
                    * personal_best (numpy.ndarray): Personal best positions.
                    * personal_best_fitness (numpy.ndarray): Personal best fitness values.
                    * alpha (numpy.ndarray): Per-individual alpha values.
                    * gamma (numpy.ndarray): Per-individual gamma values.
                    * theta (numpy.ndarray): Per-individual theta values.
                    * rs (float): Search-space diameter.

        See Also
        --------
            * :func:`niapy.algorithms.Algorithm.init_population`

        """
        population, population_fitness, state = super().init_population(task)
        alpha, gamma, theta = self._init_params(self.population_size)
        personal_best = population.copy()
        personal_best_fitness = population_fitness.copy()
        rs = self.dn(task.upper, task.lower)
        state.update({
            'personal_best': personal_best,
            'personal_best_fitness': personal_best_fitness,
            'alpha': alpha,
            'gamma': gamma,
            'theta': theta,
            'rs': rs,
        })
        return population, population_fitness, state

    def run_iteration(self, task, population, population_fitness, best_x, best_fitness, **params):
        r"""
        Perform one iteration of the Anarchic Society Optimization algorithm.

        Args:
            task (Task): Optimization task.
            population (numpy.ndarray): Current population positions.
            population_fitness (numpy.ndarray): Current population fitness values.
            best_x (numpy.ndarray): Global best position found so far.
            best_fitness (float): Global best fitness value found so far.
            **params: Additional algorithm state (personal_best, personal_best_fitness,
                alpha, gamma, theta, rs). See :func:`AnarchicSocietyOptimization.init_population`.

        Returns
        -------
            Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, float, Dict[str, Any]]:
                1. New population positions.
                2. New population fitness values.
                3. New global best position.
                4. New global best fitness value.
                5. Updated state dictionary.

        """
        personal_best = params.pop('personal_best')
        personal_best_fitness = params.pop('personal_best_fitness')
        alpha = params.pop('alpha')
        gamma = params.pop('gamma')
        theta = params.pop('theta')
        rs = params.pop('rs')

        n = len(population)

        # Find best neighbour index for each individual
        neighbor_indices = [
            self._get_best_neighbor(i, population, population_fitness, rs)
            for i in range(n)
        ]

        # Compute movement probability indices
        mp_c = np.array([
            self._fi(population_fitness[i], personal_best_fitness[i], best_fitness, alpha[i])
            for i in range(n)
        ])
        mp_s = np.array([
            self._ei(population_fitness[i], population_fitness[neighbor_indices[i]], gamma[i])
            for i in range(n)
        ])
        mp_p = np.array([
            self._ii(population_fitness[i], personal_best_fitness[i], theta[i])
            for i in range(n)
        ])

        # Generate new positions using the chosen combination strategy
        new_population = np.empty_like(population)
        new_fitness = np.empty(n)
        for i in range(n):
            # Pick a random individual different from i
            rand_idx = self.integers(n, skip=[i])
            new_population[i], new_fitness[i] = self.combination(
                population[i],
                personal_best[i],
                best_x,
                population[rand_idx],
                mp_c[i], mp_s[i], mp_p[i],
                self.f, self.cr,
                task,
                self.rng,
            )

        # Update personal bests
        personal_best, personal_best_fitness = self._update_personal_best(
            new_population, new_fitness, personal_best, personal_best_fitness
        )

        # Update global best
        best_x, best_fitness = self.get_best(new_population, new_fitness, best_x, best_fitness)

        return new_population, new_fitness, best_x, best_fitness, {
            'personal_best': personal_best,
            'personal_best_fitness': personal_best_fitness,
            'alpha': alpha,
            'gamma': gamma,
            'theta': theta,
            'rs': rs,
        }
