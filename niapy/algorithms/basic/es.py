# encoding=utf8
import logging
from math import ceil

import numpy as np
from numpy.linalg import norm, cholesky, eig, solve, lstsq

from niapy.algorithms.algorithm import Algorithm, Individual, default_individual_init
from niapy.util import objects_to_array

logging.basicConfig()
logger = logging.getLogger('niapy.algorithms.basic')
logger.setLevel('INFO')

__all__ = ['EvolutionStrategy1p1', 'EvolutionStrategyMp1', 'EvolutionStrategyMpL', 'EvolutionStrategyML',
           'CovarianceMatrixAdaptionEvolutionStrategy']


class IndividualES(Individual):
    r"""Individual for Evolution Strategies.

    See Also:
        * :class:`niapy.algorithms.Individual`

    """

    def __init__(self, rho=1, **kwargs):
        r"""Initialize individual.

        Args:
            rho(Optional[int]): Rho parameter.

        See Also:
            * :func:`niapy.algorithms.Individual.__init__`

        """
        super().__init__(**kwargs)
        self.rho = rho


class EvolutionStrategy1p1(Algorithm):
    r"""Implementation of (1 + 1) evolution strategy algorithm. Uses just one individual.

    Algorithm:
        (1 + 1) Evolution Strategy Algorithm

    Date:
        2018

    Authors:
        Klemen Berkovič

    License:
        MIT

    Reference URL:

    Reference paper:
        KALYANMOY, Deb. "Multi-Objective optimization using evolutionary algorithms".
        John Wiley & Sons, Ltd. Kanpur, India. 2001.

    Attributes:
        Name (List[str]): List of strings representing algorithm names.
        mu (int): Number of parents.
        k (int): Number of iterations before checking and fixing rho.
        c_a (float): Search range amplification factor.
        c_r (float): Search range reduction factor.

    See Also:
        * :class:`niapy.algorithms.Algorithm`

    """

    Name = ['EvolutionStrategy1p1', 'EvolutionStrategy(1+1)', 'ES(1+1)']

    @staticmethod
    def info():
        r"""Get algorithms information.

        Returns:
            str: Algorithm information.

        See Also:
            * :func:`niapy.algorithms.Algorithm.info`

        """
        return r"""KALYANMOY, Deb. "Multi-Objective optimization using evolutionary algorithms". John Wiley & Sons, Ltd. Kanpur, India. 2001."""

    def __init__(self, mu=1, k=10, c_a=1.1, c_r=0.5, epsilon=1e-20, *args, **kwargs):
        """Initialize EvolutionStrategy1p1.

        Args:
            mu (Optional[int]): Number of parents
            k (Optional[int]): Number of iterations before checking and fixing rho
            c_a (Optional[float]): Search range amplification factor
            c_r (Optional[float]): Search range reduction factor
            epsilon (Optional[float]): Small number.

        See Also:
            * :func:`niapy.algorithms.Algorithm.__init__`

        """
        super().__init__(population_size=mu, individual_type=kwargs.pop('individual_type', IndividualES), *args, **kwargs)
        self.mu = mu
        self.k = k
        self.c_a = c_a
        self.c_r = c_r
        self.epsilon = epsilon

    def set_parameters(self, mu=1, k=10, c_a=1.1, c_r=0.5, epsilon=1e-20, **kwargs):
        r"""Set the arguments of an algorithm.

        Arguments:
            mu (Optional[int]): Number of parents
            k (Optional[int]): Number of iterations before checking and fixing rho
            c_a (Optional[float]): Search range amplification factor
            c_r (Optional[float]): Search range reduction factor
            epsilon (Optional[float]): Small number.

        See Also:
            * :func:`niapy.algorithms.Algorithm.set_parameters`

        """
        super().set_parameters(population_size=mu, individual_type=kwargs.pop('individual_type', IndividualES), **kwargs)
        self.mu = mu
        self.k = k
        self.c_a = c_a
        self.c_r = c_r
        self.epsilon = epsilon

    def mutate(self, x, rho):
        r"""Mutate individual.

        Args:
            x (numpy.ndarray): Current individual.
            rho (float): Current standard deviation.

        Returns:
            Individual: Mutated individual.

        """
        return x + self.normal(0, rho, len(x))

    def update_rho(self, rho, k):
        r"""Update standard deviation.

        Args:
            rho (float): Current standard deviation.
            k (int): Number of successful mutations.

        Returns:
            float: New standard deviation.

        """
        phi = k / self.k
        if phi < 0.2:
            return self.c_r * rho if rho > self.epsilon else 1
        elif phi > 0.2:
            return self.c_a * rho if rho > self.epsilon else 1
        else:
            return rho

    def init_population(self, task):
        r"""Initialize starting individual.

        Args:
            task (Task): Optimization task.

        Returns:
            Tuple[Individual, float, Dict[str, Any]]:
                1. Initialized individual.
                2. Initialized individual fitness/function value.
                3. Additional arguments:
                    * ki (int): Number of successful rho update.

        """
        print(task.dimension)
        c, ki = IndividualES(task=task, rng=self.rng), 0
        return c, c.f, {'ki': ki}

    def run_iteration(self, task, c, population_fitness, best_x, best_fitness, **params):
        r"""Core function of EvolutionStrategy(1+1) algorithm.

        Args:
            task (Task): Optimization task.
            c (Individual): Current position.
            population_fitness (float): Current position function/fitness value.
            best_x (numpy.ndarray): Global best position.
            best_fitness (float): Global best function/fitness value.
            **params (Dict[str, Any]): Additional arguments.

        Returns:
            Tuple[Individual, float, Individual, float, Dict[str, Any]]:
                1. Initialized individual.
                2. Initialized individual fitness/function value.
                3. New global best solution.
                4. New global best solutions fitness/objective value.
                5. Additional arguments:
                    * ki (int): Number of successful rho update.

        """
        ki = params.pop('ki')

        if (task.iters + 1) % self.k == 0:
            c.rho, ki = self.update_rho(c.rho, ki), 0
        cn = objects_to_array([task.repair(self.mutate(c.x, c.rho), self.rng) for _i in range(self.mu)])
        cn_f = np.asarray([task.eval(cn[i]) for i in range(len(cn))])
        ib = np.argmin(cn_f)
        if cn_f[ib] < c.f:
            c.x, c.f, ki = cn[ib], cn_f[ib], ki + 1
            if cn_f[ib] < best_fitness:
                best_x, best_fitness = self.get_best(cn[ib], cn_f[ib], best_x, best_fitness)
        return c, c.f, best_x, best_fitness, {'ki': ki}


class EvolutionStrategyMp1(EvolutionStrategy1p1):
    r"""Implementation of (mu + 1) evolution strategy algorithm. Algorithm creates mu mutants but into new generation goes only one individual.

    Algorithm:
        (:math:`\mu + 1`) Evolution Strategy Algorithm

    Date:
        2018

    Authors:
        Klemen Berkovič

    License:
        MIT

    Reference URL:

    Reference paper:

    Attributes:
        Name (List[str]): List of strings representing algorithm names.

    See Also:
        * :class:`niapy.algorithms.basic.EvolutionStrategy1p1`

    """

    Name = ['EvolutionStrategyMp1', 'EvolutionStrategy(mu+1)', 'ES(m+1)']

    @staticmethod
    def info():
        r"""Get algorithms information.

        Returns:
            str: Algorithm information.

        See Also:
            * :func:`niapy.algorithms.Algorithm.info`

        """
        return r"""TODO"""

    def __init__(self, mu=40, *args, **kwargs):
        """Initialize EvolutionStrategyMp1."""
        super().__init__(mu=mu, *args, **kwargs)

    def set_parameters(self, **kwargs):
        r"""Set core parameters of EvolutionStrategy(mu+1) algorithm.

        Args:
            **kwargs (Dict[str, Any]):

        See Also:
            * :func:`niapy.algorithms.basic.EvolutionStrategy1p1.set_parameters`

        """
        mu = kwargs.pop('mu', 40)
        super().set_parameters(self, mu=mu, **kwargs)


class EvolutionStrategyMpL(EvolutionStrategy1p1):
    r"""Implementation of (mu + lambda) evolution strategy algorithm. Mutation creates lambda individual. Lambda individual compete with mu individuals for survival, so only mu individual go to new generation.

    Algorithm:
        (:math:`\mu + \lambda`) Evolution Strategy Algorithm

    Date:
        2018

    Authors:
        Klemen Berkovič

    License:
        MIT

    Reference URL:

    Reference paper:

    Attributes:
        Name (List[str]): List of strings representing algorithm names
        lam (int): Lambda.

    See Also:
        * :class:`niapy.algorithms.basic.EvolutionStrategy1p1`

    """

    Name = ['EvolutionStrategyMpL', 'EvolutionStrategy(mu+lambda)', 'ES(m+l)']

    @staticmethod
    def info():
        r"""Get algorithms information.

        Returns:
            str: Algorithm information.

        See Also:
            * :func:`niapy.algorithms.Algorithm.info`

        """
        return r"""TODO"""

    def __init__(self, lam=45, *args, **kwargs):
        """Initialize EvolutionStrategyMpL.

        Args:
            lam (int): Number of new individual generated by mutation.

        """
        super().__init__(initialization_function=default_individual_init, *args, **kwargs)
        self.lam = lam

    def set_parameters(self, lam=45, **kwargs):
        r"""Set the arguments of an algorithm.

        Args:
            lam (int): Number of new individual generated by mutation.

        See Also:
            * :func:`niapy.algorithms.basic.es.EvolutionStrategy1p1.set_parameters`

        """
        super().set_parameters(initialization_function=default_individual_init, **kwargs)
        self.lam = lam

    def update_rho(self, pop, k):
        r"""Update standard deviation for population.

        Args:
            pop (numpy.ndarray[Individual]): Current population.
            k (int): Number of successful mutations.

        """
        phi = k / self.k
        if phi < 0.2:
            for i in pop:
                i.rho = self.c_r * i.rho
        elif phi > 0.2:
            for i in pop:
                i.rho = self.c_a * i.rho

    @staticmethod
    def change_count(c, cn):
        r"""Update number of successful mutations for population.

        Args:
            c (numpy.ndarray[Individual]): Current population.
            cn (numpy.ndarray[Individual]): New population.

        Returns:
            int: Number of successful mutations.

        """
        k = 0
        for e in cn:
            if e not in c.tolist():
                k += 1
        return k

    def mutate_rand(self, pop, task):
        r"""Mutate random individual form population.

        Args:
            pop (numpy.ndarray[Individual]): Current population.
            task (Task): Optimization task.

        Returns:
            numpy.ndarray: Random individual from population that was mutated.

        """
        i = self.integers(self.mu)
        return task.repair(self.mutate(pop[i].x, pop[i].rho), rng=self.rng)

    def init_population(self, task):
        r"""Initialize starting population.

        Args:
            task (Task): Optimization task.

        Returns:
            Tuple[numpy.ndarray[Individual], numpy.ndarray[float], Dict[str, Any]]:
                1. Initialized population.
                2. Initialized populations function/fitness values.
                3. Additional arguments:
                    * ki (int): Number of successful mutations.

        See Also:
            * :func:`niapy.algorithms.algorithm.Algorithm.init_population`

        """
        c, fc, d = Algorithm.init_population(self, task)
        d.update({'ki': 0})
        return c, fc, d

    def run_iteration(self, task, c, population_fitness, best_x, best_fitness, **params):
        r"""Core function of EvolutionStrategyMpL algorithm.

        Args:
            task (Task): Optimization task.
            c (numpy.ndarray): Current population.
            population_fitness (numpy.ndarray): Current populations fitness/function values.
            best_x (numpy.ndarray): Global best individual.
            best_fitness (float): Global best individuals fitness/function value.
            **params (Dict[str, Any]): Additional arguments.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, float, Dict[str, Any]]:
                1. New population.
                2. New populations function/fitness values.
                3. New global best solution.
                4. New global best solutions fitness/objective value.
                5. Additional arguments:
                    * ki (int): Number of successful mutations.

        """
        ki = params.pop('ki')

        if (task.iters + 1) % self.k == 0:
            _, ki = self.update_rho(c, ki), 0
        cn = objects_to_array(
            [IndividualES(x=self.mutate_rand(c, task), task=task, rng=self.rng) for _ in range(self.lam)])
        cn = np.append(cn, c)
        cn = objects_to_array([cn[i] for i in np.argsort([i.f for i in cn])[:self.mu]])
        ki += self.change_count(c, cn)
        fcn = np.asarray([x.f for x in cn])
        best_x, best_fitness = self.get_best(cn, fcn, best_x, best_fitness)
        return cn, fcn, best_x, best_fitness, {'ki': ki}


class EvolutionStrategyML(EvolutionStrategyMpL):
    r"""Implementation of (mu, lambda) evolution strategy algorithm. Algorithm is good for dynamic environments. Mu individual create lambda children. Only best mu children go to new generation. Mu parents are discarded.

    Algorithm:
        (:math:`\mu + \lambda`) Evolution Strategy Algorithm

    Date:
        2018

    Authors:
        Klemen Berkovič

    License:
        MIT

    Reference URL:

    Reference paper:

    Attributes:
        Name (List[str]): List of strings representing algorithm names

    See Also:
        * :class:`niapy.algorithm.basic.es.EvolutionStrategyMpL`

    """

    Name = ['EvolutionStrategyML', 'EvolutionStrategy(mu,lambda)', 'ES(m,l)']

    @staticmethod
    def info():
        r"""Get algorithms information.

        Returns:
            str: Algorithm information.

        See Also:
            * :func:`niapy.algorithms.Algorithm.info`

        """
        return r"""TODO"""

    def new_pop(self, pop):
        r"""Return new population.

        Args:
            pop (numpy.ndarray): Current population.

        Returns:
            numpy.ndarray: New population.

        """
        pop_s = np.argsort([i.f for i in pop])
        if self.mu < self.lam:
            return objects_to_array([pop[i] for i in pop_s[:self.mu]])
        new_population = list()
        for i in range(int(ceil(float(self.mu) / self.lam))):
            new_population.extend(pop[:self.lam if (self.mu - i * self.lam) >= self.lam else self.mu - i * self.lam])
        return objects_to_array(new_population)

    def init_population(self, task):
        r"""Initialize starting population.

        Args:
            task (Task): Optimization task.

        Returns:
            Tuple[numpy.ndarray[Individual], numpy.ndarray[float], Dict[str, Any]]:
                1. Initialized population.
                2. Initialized populations fitness/function values.
                3. Additional arguments.

        See Also:
            * :func:`niapy.algorithm.basic.es.EvolutionStrategyMpL.init_population`

        """
        c, fc, _ = EvolutionStrategyMpL.init_population(self, task)
        return c, fc, {}

    def run_iteration(self, task, c, population_fitness, best_x, best_fitness, **params):
        r"""Core function of EvolutionStrategyML algorithm.

        Args:
            task (Task): Optimization task.
            c (numpy.ndarray): Current population.
            population_fitness (numpy.ndarray): Current population fitness/function values.
            best_x (numpy.ndarray): Global best individual.
            best_fitness (float): Global best individuals fitness/function value.
            **params Dict[str, Any]: Additional arguments.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, float, Dict[str, Any]]:
                1. New population.
                2. New populations fitness/function values.
                3. New global best solution.
                4. New global best solutions fitness/objective value.
                5. Additional arguments.

        """
        cn = objects_to_array(
            [IndividualES(x=self.mutate_rand(c, task), task=task, rand=self.rng) for _ in range(self.lam)])
        c = self.new_pop(cn)
        fc = np.asarray([x.f for x in c])
        best_x, best_fitness = self.get_best(c, fc, best_x, best_fitness)
        return c, fc, best_x, best_fitness, {}


def cmaes_function(task, rng, epsilon=1e-20):
    lam, alpha_mu, hs, sigma0 = (4 + np.round(3 * np.log(task.dimension))) * 10, 2, 0, 0.3 * task.range
    mu = int(np.round(lam / 2))
    w = np.log(mu + 0.5) - np.log(range(1, mu + 1))
    w = w / np.sum(w)
    mu_eff = 1 / np.sum(w ** 2)
    cs = (mu_eff + 2) / (task.dimension + mu_eff + 5)
    ds = 1 + cs + 2 * max(np.sqrt((mu_eff - 1) / (task.dimension + 1)) - 1, 0)
    enn = np.sqrt(task.dimension) * (1 - 1 / (4 * task.dimension) + 1 / (21 * task.dimension ** 2))
    cc, c1 = (4 + mu_eff / task.dimension) / (4 + task.dimension + 2 * mu_eff / task.dimension), 2 / ((task.dimension + 1.3) ** 2 + mu_eff)
    cmu, hth = min(1 - c1, alpha_mu * (mu_eff - 2 + 1 / mu_eff) / ((task.dimension + 2) ** 2 + alpha_mu * mu_eff / 2)), (
                1.4 + 2 / (task.dimension + 1)) * enn
    ps = np.zeros(task.dimension)
    pc = np.zeros(task.dimension)
    c = np.eye(task.dimension)
    sigma = sigma0
    x = rng.uniform(task.lower, task.upper)
    x_f = task.eval(x)
    while not task.stopping_condition_iter():
        pop_step = np.asarray([rng.multivariate_normal(np.zeros(task.dimension), c) for _ in range(int(lam))])
        pop = np.asarray([task.repair(x + sigma * ps, rng=rng) for ps in pop_step])
        pop_f = np.apply_along_axis(task.eval, 1, pop)
        i_sort = np.argsort(pop_f)
        pop, pop_f, pop_step = pop[i_sort[:mu]], pop_f[i_sort[:mu]], pop_step[i_sort[:mu]]
        if pop_f[0] < x_f:
            x, x_f = pop[0], pop_f[0]
        m = np.sum(w * pop_step.T, axis=1)
        ps = solve(cholesky(c).conj() + epsilon, ((1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * m + epsilon).T)[0].T
        sigma *= np.exp(cs / ds * (norm(ps) / enn - 1)) ** 0.3
        i_fix = np.where(sigma == np.inf)
        if np.any(i_fix):
            sigma[i_fix] = sigma0
        if norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * ((task.iters + 1) + 1))) < hth:
            hs = 1
        else:
            hs = 0
        delta = (1 - hs) * cc * (2 - cc)
        pc = (1 - cc) * pc + hs * np.sqrt(cc * (2 - cc) * mu_eff) * m
        c = (1 - c1 - cmu) * c + c1 * (
                    np.tile(pc, [len(pc), 1]) * np.tile(pc.reshape([len(pc), 1]), [1, len(pc)]) + delta * c)
        for i in range(mu):
            c += cmu * w[i] * np.tile(pop_step[i], [len(pop_step[i]), 1]) * np.tile(pop_step[i].reshape([len(pop_step[i]), 1]), [1, len(pop_step[i])])
        e, v = eig(c)
        if np.any(e < epsilon):
            e = np.fmax(e, 0)
            c = lstsq(v.T, np.dot(v, np.diag(e)).T, rcond=None)[0].T
    return x, x_f


class CovarianceMatrixAdaptionEvolutionStrategy(Algorithm):
    r"""Implementation of CMAES.

    Algorithm:
        CMAES

    Date:
        2018

    Authors:
        Klemen Berkovič

    License:
        MIT

    Reference URL:
        https://arxiv.org/abs/1604.00772

    Reference paper:
        Hansen, Nikolaus. "The CMA evolution strategy: A tutorial." arXiv preprint arXiv:1604.00772 (2016).

    Attributes:
        Name (List[str]): List of names representing algorithm names
        epsilon (float): Small number.

    """

    Name = ['CovarianceMatrixAdaptionEvolutionStrategy', 'CMA-ES', 'CMAES']
    epsilon = 1e-20

    @staticmethod
    def info():
        r"""Get algorithms information.

        Returns:
            str: Algorithm information.

        See Also:
            * :func:`niapy.algorithms.Algorithm.info`

        """
        return r"""Hansen, Nikolaus. "The CMA evolution strategy: A tutorial." arXiv preprint arXiv:1604.00772 (2016)."""

    def __init__(self, epsilon=1e-20, *args, **kwargs):
        """Initialize CovarianceMatrixAdaptionEvolutionStrategy.

        Args:
            epsilon (float): Small number.

        """
        super().__init__(*args, **kwargs)
        self.epsilon = epsilon

    def set_parameters(self, epsilon=1e-20, **kwargs):
        r"""Set core parameters of CovarianceMatrixAdaptionEvolutionStrategy algorithm.

        Args:
            epsilon (float): Small number.

        """
        super().set_parameters(**kwargs)
        self.epsilon = epsilon

    def run_task(self, task):
        r"""Start the optimization.

        Args:
            task (Task): Task with bounds and objective function for optimization.

        Returns:
            Tuple[numpy.ndarray, float]:
                1. Best individuals components found in optimization process.
                2. Best fitness value found in optimization process.

        See Also:
            * :func:`niapy.algorithms.Algorithm.run_task`

        """
        return cmaes_function(task, rng=self.rng, epsilon=self.epsilon)

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
