# encoding=utf8

"""Particle swarm algorithm module."""

import numpy as np

from niapy.algorithms.algorithm import Algorithm
from niapy.util import full_array
from niapy.util.repair import reflect

__all__ = [
    'ParticleSwarmAlgorithm',
    'ParticleSwarmOptimization',
    'CenterParticleSwarmOptimization',
    'MutatedParticleSwarmOptimization',
    'MutatedCenterParticleSwarmOptimization',
    'ComprehensiveLearningParticleSwarmOptimizer',
    'MutatedCenterUnifiedParticleSwarmOptimization',
    'OppositionVelocityClampingParticleSwarmOptimization'
]


class ParticleSwarmAlgorithm(Algorithm):
    r"""Implementation of Particle Swarm Optimization algorithm.

    Algorithm:
        Particle Swarm Optimization algorithm

    Date:
        2018

    Authors:
        Lucija Brezočnik, Grega Vrbančič, Iztok Fister Jr. and Klemen Berkovič

    License:
        MIT

    Reference paper:
        Kennedy, J. and Eberhart, R. "Particle Swarm Optimization". Proceedings of IEEE International Conference on Neural Networks. IV. pp. 1942--1948, 1995.

    Attributes:
        Name (List[str]): List of strings representing algorithm names
        c1 (float): Cognitive component.
        c2 (float): Social component.
        w (Union[float, numpy.ndarray[float]]): Inertial weight.
        min_velocity (Union[float, numpy.ndarray[float]]): Minimal velocity.
        max_velocity (Union[float, numpy.ndarray[float]]): Maximal velocity.
        repair (Callable[[numpy.ndarray, numpy.ndarray, numpy.ndarray, Optional[numpy.random.Generator]], numpy.ndarray]): Repair method for velocity.

    See Also:
        * :class:`niapy.algorithms.Algorithm`

    """

    Name = ['WeightedVelocityClampingParticleSwarmAlgorithm', 'WVCPSO']

    @staticmethod
    def info():
        r"""Get basic information of algorithm.

        Returns:
            str: Basic information of algorithm.

        See Also:
            * :func:`niapy.algorithms.Algorithm.info`

        """
        return r"""Kennedy, J. and Eberhart, R. "Particle Swarm Optimization". Proceedings of IEEE International Conference on Neural Networks. IV. pp. 1942--1948, 1995."""

    def __init__(self, population_size=25, c1=2.0, c2=2.0, w=0.7, min_velocity=-1.5, max_velocity=1.5, repair=reflect,
                 *args, **kwargs):
        """Initialize ParticleSwarmAlgorithm.

        Args:
            population_size (int): Population size
            c1 (float): Cognitive component.
            c2 (float): Social component.
            w (Union[float, numpy.ndarray]): Inertial weight.
            min_velocity (Union[float, numpy.ndarray]): Minimal velocity.
            max_velocity (Union[float, numpy.ndarray]): Maximal velocity.
            repair (Callable[[np.ndarray, np.ndarray, np.ndarray, dict], np.ndarray]): Repair method for velocity.

        See Also:
            * :func:`niapy.algorithms.Algorithm.__init__`

        """
        super().__init__(population_size, *args, **kwargs)
        self.c1 = c1
        self.c2 = c2
        self.w = w
        self.min_velocity = min_velocity
        self.max_velocity = max_velocity
        self.repair = repair

    def set_parameters(self, population_size=25, c1=2.0, c2=2.0, w=0.7, min_velocity=-1.5, max_velocity=1.5,
                       repair=reflect, **kwargs):
        r"""Set Particle Swarm Algorithm main parameters.

        Args:
            population_size (int): Population size
            c1 (float): Cognitive component.
            c2 (float): Social component.
            w (Union[float, numpy.ndarray]): Inertial weight.
            min_velocity (Union[float, numpy.ndarray]): Minimal velocity.
            max_velocity (Union[float, numpy.ndarray]): Maximal velocity.
            repair (Callable[[np.ndarray, np.ndarray, np.ndarray, dict], np.ndarray]): Repair method for velocity.

        See Also:
            * :func:`niapy.algorithms.Algorithm.set_parameters`

        """
        super().set_parameters(population_size=population_size, **kwargs)
        self.c1 = c1
        self.c2 = c2
        self.w = w
        self.min_velocity = min_velocity
        self.max_velocity = max_velocity
        self.repair = repair

    def get_parameters(self):
        r"""Get value of parameters for this instance of algorithm.

        Returns:
            Dict[str, Union[int, float, numpy.ndarray]]: Dictionary which has parameters mapped to values.

        See Also:
            * :func:`niapy.algorithms.Algorithm.get_parameters`

        """
        d = super().get_parameters()
        d.update({
            'c1': self.c1,
            'c2': self.c2,
            'w': self.w,
            'min_velocity': self.min_velocity,
            'max_velocity': self.max_velocity
        })
        return d

    def init(self, task):
        r"""Initialize dynamic arguments of Particle Swarm Optimization algorithm.

        Args:
            task (Task): Optimization task.

        Returns:
            Dict[str, Union[float, numpy.ndarray]]:
                * w (numpy.ndarray): Inertial weight.
                * min_velocity (numpy.ndarray): Minimal velocity.
                * max_velocity (numpy.ndarray): Maximal velocity.
                * v (numpy.ndarray): Initial velocity of particle.

        """
        return {
            'w': full_array(self.w, task.dimension),
            'min_velocity': full_array(self.min_velocity, task.dimension),
            'max_velocity': full_array(self.max_velocity, task.dimension),
            'v': np.zeros((self.population_size, task.dimension))
        }

    def init_population(self, task):
        r"""Initialize population and dynamic arguments of the Particle Swarm Optimization algorithm.

        Args:
            task: Optimization task.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray, list, dict]:
                1. Initial population.
                2. Initial population fitness/function values.
                3. Additional arguments.
                4. Additional keyword arguments:
                    * personal_best (numpy.ndarray): particles best population.
                    * personal_best_fitness (numpy.ndarray[float]): particles best positions function/fitness value.
                    * w (numpy.ndarray): Inertial weight.
                    * min_velocity (numpy.ndarray): Minimal velocity.
                    * max_velocity (numpy.ndarray): Maximal velocity.
                    * v (numpy.ndarray): Initial velocity of particle.

        See Also:
            * :func:`niapy.algorithms.Algorithm.init_population`

        """
        pop, fpop, d = super().init_population(task)
        d.update(self.init(task))
        d.update({'personal_best': pop.copy(), 'personal_best_fitness': fpop.copy()})
        return pop, fpop, d

    def update_velocity(self, v, p, pb, gb, w, min_velocity, max_velocity, task, **kwargs):
        r"""Update particle velocity.

        Args:
            v (numpy.ndarray): Current velocity of particle.
            p (numpy.ndarray): Current position of particle.
            pb (numpy.ndarray): Personal best position of particle.
            gb (numpy.ndarray): Global best position of particle.
            w (Union[float, numpy.ndarray]): Weights for velocity adjustment.
            min_velocity (numpy.ndarray): Minimal velocity allowed.
            max_velocity (numpy.ndarray): Maximal velocity allowed.
            task (Task): Optimization task.
            kwargs: Additional arguments.

        Returns:
            numpy.ndarray: Updated velocity of particle.

        """
        return self.repair(
            w * v + self.c1 * self.random(task.dimension) * (pb - p) + self.c2 * self.random(task.dimension) * (gb - p),
            min_velocity, max_velocity)

    def run_iteration(self, task, pop, fpop, xb, fxb, **params):
        r"""Core function of Particle Swarm Optimization algorithm.

        Args:
            task (Task): Optimization task.
            pop (numpy.ndarray): Current populations.
            fpop (numpy.ndarray): Current population fitness/function values.
            xb (numpy.ndarray): Current best particle.
            fxb (float): Current best particle fitness/function value.
            params (dict): Additional function keyword arguments.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, float, dict]:
                1. New population.
                2. New population fitness/function values.
                3. New global best position.
                4. New global best positions function/fitness value.
                5. Additional arguments.
                6. Additional keyword arguments:
                    * personal_best (numpy.ndarray): Particles best population.
                    * personal_best_fitness (numpy.ndarray[float]): Particles best positions function/fitness value.
                    * w (numpy.ndarray): Inertial weight.
                    * min_velocity (numpy.ndarray): Minimal velocity.
                    * max_velocity (numpy.ndarray): Maximal velocity.
                    * v (numpy.ndarray): Initial velocity of particle.

        See Also:
            * :class:`niapy.algorithms.algorithm.Algorithm.run_iteration`

        """
        personal_best = params.pop('personal_best')
        personal_best_fitness = params.pop('personal_best_fitness')
        w = params.pop('w')
        min_velocity = params.pop('min_velocity')
        max_velocity = params.pop('max_velocity')
        v = params.pop('v')

        for i in range(len(pop)):
            v[i] = self.update_velocity(v[i], pop[i], personal_best[i], xb, w, min_velocity, max_velocity, task)
            pop[i] = task.repair(pop[i] + v[i], rng=self.rng)
            fpop[i] = task.eval(pop[i])
            if fpop[i] < personal_best_fitness[i]:
                personal_best[i], personal_best_fitness[i] = pop[i].copy(), fpop[i]
            if fpop[i] < fxb:
                xb, fxb = pop[i].copy(), fpop[i]
        return pop, fpop, xb, fxb, {'personal_best': personal_best, 'personal_best_fitness': personal_best_fitness,
                                    'w': w, 'min_velocity': min_velocity, 'max_velocity': max_velocity, 'v': v}


class ParticleSwarmOptimization(ParticleSwarmAlgorithm):
    r"""Implementation of Particle Swarm Optimization algorithm.

    Algorithm:
        Particle Swarm Optimization algorithm

    Date:
        2018

    Authors:
        Lucija Brezočnik, Grega Vrbančič, Iztok Fister Jr. and Klemen Berkovič

    License:
        MIT

    Reference paper:
        Kennedy, J. and Eberhart, R. "Particle Swarm Optimization". Proceedings of IEEE International Conference on Neural Networks. IV. pp. 1942--1948, 1995.

    Attributes:
        Name (List[str]): List of strings representing algorithm names

    See Also:
        * :class:`niapy.algorithms.basic.WeightedVelocityClampingParticleSwarmAlgorithm`

    """

    Name = ['ParticleSwarmAlgorithm', 'PSO']

    @staticmethod
    def info():
        r"""Get basic information of algorithm.

        Returns:
            str: Basic information of algorithm.

        See Also:
            * :func:`niapy.algorithms.Algorithm.info`

        """
        return r"""Kennedy, J. and Eberhart, R. "Particle Swarm Optimization". Proceedings of IEEE International Conference on Neural Networks. IV. pp. 1942--1948, 1995."""

    def __init__(self, *args, **kwargs):
        """Initialize ParticleSwarmOptimization."""
        super().__init__(*args, **kwargs)
        self.w = 1.0
        self.min_velocity = -np.inf
        self.max_velocity = np.inf

    def set_parameters(self, **kwargs):
        r"""Set core parameters of algorithm.

        See Also:
            * :func:`niapy.algorithms.basic.WeightedVelocityClampingParticleSwarmAlgorithm.set_parameters`

        """
        kwargs.pop('w', None), kwargs.pop('vMin', None), kwargs.pop('vMax', None)
        super().set_parameters(w=1, min_velocity=-np.inf, max_velocity=np.inf, **kwargs)


class OppositionVelocityClampingParticleSwarmOptimization(ParticleSwarmAlgorithm):
    r"""Implementation of Opposition-Based Particle Swarm Optimization with Velocity Clamping.

    Algorithm:
        Opposition-Based Particle Swarm Optimization with Velocity Clamping

    Date:
        2019

    Authors:
        Klemen Berkovič

    License:
        MIT

    Reference paper:
        Shahzad, Farrukh, et al. "Opposition-based particle swarm optimization with velocity clamping (OVCPSO)." Advances in Computational Intelligence. Springer, Berlin, Heidelberg, 2009. 339-348

    Attributes:
        p0: Probability of opposite learning phase.
        w_min: Minimum inertial weight.
        w_max: Maximum inertial weight.
        sigma: Velocity scaling factor.

    See Also:
        * :class:`niapy.algorithms.basic.ParticleSwarmAlgorithm`

    """

    Name = ['OppositionVelocityClampingParticleSwarmOptimization', 'OVCPSO']

    @staticmethod
    def info():
        r"""Get basic information of algorithm.

        Returns:
            str: Basic information of algorithm.

        See Also:
            * :func:`niapy.algorithms.Algorithm.info`

        """
        return r"""Shahzad, Farrukh, et al. "Opposition-based particle swarm optimization with velocity clamping (OVCPSO)." Advances in Computational Intelligence. Springer, Berlin, Heidelberg, 2009. 339-348"""

    def __init__(self, p0=.3, w_min=.4, w_max=.9, sigma=.1, c1=1.49612, c2=1.49612, *args, **kwargs):
        """Initialize OppositionVelocityClampingParticleSwarmOptimization.

        Args:
            p0 (float): Probability of running Opposite learning.
            w_min (numpy.ndarray): Minimal value of weights.
            w_max (numpy.ndarray): Maximum value of weights.
            sigma (numpy.ndarray): Velocity range factor.
            c1 (float): Cognitive component.
            c2 (float): Social component.

        See Also:
            * :func:`niapy.algorithm.basic.ParticleSwarmAlgorithm.__init__`

        """
        kwargs.pop('w', None)
        super().__init__(w=w_max, c1=c1, c2=c2, *args, **kwargs)
        self.p0 = p0
        self.w_min = w_min
        self.w_max = w_max
        self.sigma = sigma

    def set_parameters(self, p0=.3, w_min=.4, w_max=.9, sigma=.1, c1=1.49612, c2=1.49612, **kwargs):
        r"""Set core algorithm parameters.

        Args:
            p0 (float): Probability of running Opposite learning.
            w_min (numpy.ndarray): Minimal value of weights.
            w_max (numpy.ndarray): Maximum value of weights.
            sigma (numpy.ndarray): Velocity range factor.
            c1 (float): Cognitive component.
            c2 (float): Social component.

        See Also:
            * :func:`niapy.algorithm.basic.ParticleSwarmAlgorithm.set_parameters`

        """
        kwargs.pop('w', None)
        super().set_parameters(w=w_max, c1=c1, c2=c2, **kwargs)
        self.p0 = p0
        self.w_min = w_min
        self.w_max = w_max
        self.sigma = sigma

    def get_parameters(self):
        r"""Get value of parameters for this instance of algorithm.

        Returns:
            Dict[str, Union[int, float, numpy.ndarray]]: Dictionary which has parameters mapped to values.

        See Also:
            * :func:`niapy.algorithms.basic.ParticleSwarmAlgorithm.get_parameters`

        """
        d = ParticleSwarmAlgorithm.get_parameters(self)
        d.pop('min_velocity', None), d.pop('max_velocity', None)
        d.update({
            'p0': self.p0, 'w_min': self.w_min, 'w_max': self.w_max, 'sigma': self.sigma
        })
        return d

    @staticmethod
    def opposite_learning(s_l, s_h, pop, fpop, task):
        r"""Run opposite learning phase.

        Args:
            s_l (numpy.ndarray): lower limit of opposite particles.
            s_h (numpy.ndarray): upper limit of opposite particles.
            pop (numpy.ndarray): Current populations positions.
            fpop (numpy.ndarray): Current populations functions/fitness values.
            task (Task): Optimization task.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, float]:
                1. New particles position
                2. New particles function/fitness values
                3. New best position of opposite learning phase
                4. new best function/fitness value of opposite learning phase

        """
        s_r = s_l + s_h
        s = np.asarray([s_r - e for e in pop])
        s_f = np.asarray([task.eval(e) for e in s])
        s, s_f = np.concatenate([pop, s]), np.concatenate([fpop, s_f])
        sorted_indices = np.argsort(s_f)
        return s[sorted_indices[:len(pop)]], s_f[sorted_indices[:len(pop)]], s[sorted_indices[0]], s_f[
            sorted_indices[0]]

    def init_population(self, task):
        r"""Init starting population and dynamic parameters.

        Args:
            task (Task): Optimization task.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray, list, dict]:
                1. Initialized population.
                2. Initialized populations function/fitness values.
                3. Additional arguments.
                4. Additional keyword arguments:
                    * personal_best (numpy.ndarray): particles best population.
                    * personal_best_fitness (numpy.ndarray[float]): particles best positions function/fitness value.
                    * vMin (numpy.ndarray): Minimal velocity.
                    * vMax (numpy.ndarray): Maximal velocity.
                    * V (numpy.ndarray): Initial velocity of particle.
                    * S_u (numpy.ndarray): upper bound for opposite learning.
                    * S_l (numpy.ndarray): lower bound for opposite learning.

        """
        pop, fpop, d = super().init_population(task)
        s_l, s_h = task.lower, task.upper
        pop, fpop, _, _ = self.opposite_learning(s_l, s_h, pop, fpop, task)
        pb_indices = np.where(fpop < d['personal_best_fitness'])
        d['personal_best'][pb_indices], d['personal_best_fitness'][pb_indices] = pop[pb_indices], fpop[pb_indices]
        d['min_velocity'], d['max_velocity'] = self.sigma * (task.upper - task.lower), self.sigma * (
                task.lower - task.upper)
        d.update({'s_l': s_l, 's_h': s_h})
        return pop, fpop, d

    def run_iteration(self, task, pop, fpop, xb, fxb, **params):
        r"""Core function of Opposite-based Particle Swarm Optimization with velocity clamping algorithm.

        Args:
            task (Task): Optimization task.
            pop (numpy.ndarray): Current population.
            fpop (numpy.ndarray): Current populations function/fitness values.
            xb (numpy.ndarray): Current global best position.
            fxb (float): Current global best positions function/fitness value.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, float, list, dict]:
                1. New population.
                2. New populations function/fitness values.
                3. New global best position.
                4. New global best positions function/fitness value.
                5. Additional arguments.
                6. Additional keyword arguments:
                    * personal_best: particles best population.
                    * personal_best_fitness: particles best positions function/fitness value.
                    * min_velocity: Minimal velocity.
                    * max_velocity: Maximal velocity.
                    * v: Initial velocity of particle.
                    * s_h: upper bound for opposite learning.
                    * s_l: lower bound for opposite learning.

        """
        personal_best = params.pop('personal_best')
        personal_best_fitness = params.pop('personal_best_fitness')
        min_velocity = params.pop('min_velocity')
        max_velocity = params.pop('max_velocity')
        v = params.pop('v')
        s_l = params.pop('s_l')
        s_h = params.pop('s_h')

        if self.random() < self.p0:
            pop, fpop, nb, fnb = self.opposite_learning(s_l, s_h, pop, fpop, task)
            pb_indices = np.where(fpop < personal_best_fitness)
            personal_best[pb_indices], personal_best_fitness[pb_indices] = pop[pb_indices], fpop[pb_indices]
            if fnb < fxb:
                xb, fxb = nb.copy(), fnb
        else:
            w = self.w_max - ((self.w_max - self.w_min) / task.max_iters) * (task.iters + 1)
            for i in range(len(pop)):
                v[i] = self.update_velocity(v[i], pop[i], personal_best[i], xb, w, min_velocity, max_velocity, task)
                pop[i] = task.repair(pop[i] + v[i], rng=self.rng)
                fpop[i] = task.eval(pop[i])
                if fpop[i] < personal_best_fitness[i]:
                    personal_best[i], personal_best_fitness[i] = pop[i].copy(), fpop[i]
                    if fpop[i] < fxb:
                        xb, fxb = pop[i].copy(), fpop[i]
            min_velocity, max_velocity = self.sigma * np.min(pop, axis=0), self.sigma * np.max(pop, axis=0)
        return pop, fpop, xb, fxb, {'personal_best': personal_best, 'personal_best_fitness': personal_best_fitness,
                                    'min_velocity': min_velocity,
                                    'max_velocity': max_velocity, 'v': v, 's_l': s_l, 's_h': s_h}


class CenterParticleSwarmOptimization(ParticleSwarmAlgorithm):
    r"""Implementation of Center Particle Swarm Optimization.

    Algorithm:
        Center Particle Swarm Optimization

    Date:
        2019

    Authors:
        Klemen Berkovič

    License:
        MIT

    Reference paper:
        H.-C. Tsai, Predicting strengths of concrete-type specimens using hybrid multilayer perceptrons with center-Unified particle swarm optimization, Adv. Eng. Softw. 37 (2010) 1104–1112.

    See Also:
        * :class:`niapy.algorithms.basic.WeightedVelocityClampingParticleSwarmAlgorithm`

    """

    Name = ['CenterParticleSwarmOptimization', 'CPSO']

    @staticmethod
    def info():
        r"""Get basic information of algorithm.

        Returns:
            str: Basic information of algorithm.

        See Also:
            * :func:`niapy.algorithms.Algorithm.info`

        """
        return r"""H.-C. Tsai, Predicting strengths of concrete-type specimens using hybrid multilayer perceptrons with center-Unified particle swarm optimization, Adv. Eng. Softw. 37 (2010) 1104–1112."""

    def __init__(self, *args, **kwargs):
        """Initialize CPSO."""
        kwargs.pop('min_velocity', None), kwargs.pop('max_velocity', None)
        super().__init__(min_velocity=-np.inf, max_velocity=np.inf, *args, **kwargs)

    def set_parameters(self, **kwargs):
        r"""Set core algorithm parameters.

        Args:
            **kwargs: Additional arguments.

        See Also:
            :func:`niapy.algorithm.basic.WeightedVelocityClampingParticleSwarmAlgorithm.set_parameters`

        """
        kwargs.pop('min_velocity', None), kwargs.pop('max_velocity', None)
        super().set_parameters(min_velocity=-np.inf, max_velocity=np.inf, **kwargs)

    def get_parameters(self):
        r"""Get value of parameters for this instance of algorithm.

        Returns:
            Dict[str, Union[int, float, numpy.ndarray]]: Dictionary which has parameters mapped to values.

        See Also:
            * :func:`niapy.algorithms.basic.ParticleSwarmAlgorithm.get_parameters`

        """
        d = super().get_parameters()
        d.pop('min_velocity', None), d.pop('max_velocity', None)
        return d

    def run_iteration(self, task, pop, fpop, xb, fxb, **params):
        r"""Core function of algorithm.

        Args:
            task (Task): Optimization task.
            pop (numpy.ndarray): Current population of particles.
            fpop (numpy.ndarray): Current particles function/fitness values.
            xb (numpy.ndarray): Current global best particle.
            fxb (numpy.float): Current global best particles function/fitness value.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, float, dict]:
                1. New population of particles.
                2. New populations function/fitness values.
                3. New global best particle.
                4. New global best particle function/fitness value.
                5. Additional arguments.
                6. Additional keyword arguments.

        See Also:
            * :func:`niapy.algorithm.basic.WeightedVelocityClampingParticleSwarmAlgorithm.run_iteration`

        """
        pop, fpop, xb, fxb, d = super().run_iteration(task, pop, fpop, xb, fxb, **params)
        c = np.sum(pop, axis=0) / len(pop)
        fc = task.eval(c)
        if fc <= fxb:
            xb, fxb = c, fc
        return pop, fpop, xb, fxb, d


class MutatedParticleSwarmOptimization(ParticleSwarmAlgorithm):
    r"""Implementation of Mutated Particle Swarm Optimization.

    Algorithm:
        Mutated Particle Swarm Optimization

    Date:
        2019

    Authors:
        Klemen Berkovič

    License:
        MIT

    Reference paper:
        H. Wang, C. Li, Y. Liu, S. Zeng, a hybrid particle swarm algorithm with cauchy mutation, Proceedings of the 2007 IEEE Swarm Intelligence Symposium (2007) 356–360.

    Attributes:
        num_mutations (int): Number of mutations of global best particle.

    See Also:
        * :class:`niapy.algorithms.basic.WeightedVelocityClampingParticleSwarmAlgorithm`

    """

    Name = ['MutatedParticleSwarmOptimization', 'MPSO']

    @staticmethod
    def info():
        r"""Get basic information of algorithm.

        Returns:
            str: Basic information of algorithm.

        See Also:
            * :func:`niapy.algorithms.Algorithm.info`

        """
        return r"""H. Wang, C. Li, Y. Liu, S. Zeng, a hybrid particle swarm algorithm with cauchy mutation, Proceedings of the 2007 IEEE Swarm Intelligence Symposium (2007) 356–360."""

    def __init__(self, num_mutations=10, *args, **kwargs):
        """Initialize MPSO."""
        kwargs.pop('min_velocity', None), kwargs.pop('max_velocity', None)
        super().__init__(min_velocity=-np.inf, max_velocity=np.inf, *args, **kwargs)
        self.num_mutations = num_mutations

    def set_parameters(self, num_mutations=10, **kwargs):
        r"""Set core algorithm parameters.

        Args:
            num_mutations (int): Number of mutations of global best particle.
            **kwargs: Additional arguments.

        See Also:
            * :func:`niapy.algorithm.basic.WeightedVelocityClampingParticleSwarmAlgorithm.set_parameters`

        """
        kwargs.pop('min_velocity', None), kwargs.pop('max_velocity', None)
        ParticleSwarmAlgorithm.set_parameters(self, min_velocity=-np.inf, max_velocity=np.inf, **kwargs)
        self.num_mutations = num_mutations

    def get_parameters(self):
        r"""Get value of parameters for this instance of algorithm.

        Returns:
            Dict[str, Union[int, float, numpy.ndarray]]: Dictionary which has parameters mapped to values.

        See Also:
            * :func:`niapy.algorithms.basic.ParticleSwarmAlgorithm.get_parameters`

        """
        d = ParticleSwarmAlgorithm.get_parameters(self)
        d.pop('min_velocity', None), d.pop('max_velocity', None)
        d.update({'num_mutations': self.num_mutations})
        return d

    def run_iteration(self, task, pop, fpop, xb, fxb, **params):
        r"""Core function of algorithm.

        Args:
            task (Task): Optimization task.
            pop (numpy.ndarray): Current population of particles.
            fpop (numpy.ndarray): Current particles function/fitness values.
            xb (numpy.ndarray): Current global best particle.
            fxb (float): Current global best particles function/fitness value.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, float, list, dict]:
                1. New population of particles.
                2. New populations function/fitness values.
                3. New global best particle.
                4. New global best particle function/fitness value.
                5. Additional arguments.
                6. Additional keyword arguments.

        See Also:
            * :func:`niapy.algorithm.basic.WeightedVelocityClampingParticleSwarmAlgorithm.run_iteration`

        """
        pop, fpop, xb, fxb, d = ParticleSwarmAlgorithm.run_iteration(self, task, pop, fpop, xb, fxb, **params)
        v = d['v']
        v_a = (np.sum(v, axis=0) / len(v))
        v_a = v_a / np.max(np.abs(v_a))
        for _ in range(self.num_mutations):
            g = task.repair(xb + v_a * self.uniform(task.lower, task.upper), self.rng)
            fg = task.eval(g)
            if fg <= fxb:
                xb, fxb = g, fg
        return pop, fpop, xb, fxb, d


class MutatedCenterParticleSwarmOptimization(CenterParticleSwarmOptimization):
    r"""Implementation of Mutated Particle Swarm Optimization.

    Algorithm:
        Mutated Center Particle Swarm Optimization

    Date:
        2019

    Authors:
        Klemen Berkovič

    License:
        MIT

    Reference paper:
        TODO find one

    Attributes:
        num_mutations (int): Number of mutations of global best particle.

    See Also:
        * :class:`niapy.algorithms.basic.CenterParticleSwarmOptimization`

    """

    Name = ['MutatedCenterParticleSwarmOptimization', 'MCPSO']

    @staticmethod
    def info():
        r"""Get basic information of algorithm.

        Returns:
            str: Basic information of algorithm.

        See Also:
            * :func:`niapy.algorithms.Algorithm.info`

        """
        return r"""TODO find one"""

    def __init__(self, num_mutations=10, *args, **kwargs):
        """Initialize MCPSO."""
        kwargs.pop('min_velocity', None), kwargs.pop('max_velocity', None)
        super().__init__(min_velocity=-np.inf, max_velocity=np.inf, *args, **kwargs)
        self.num_mutations = num_mutations

    def set_parameters(self, num_mutations=10, **kwargs):
        r"""Set core algorithm parameters.

        Args:
            num_mutations (int): Number of mutations of global best particle.
            **kwargs: Additional arguments.

        See Also:
            * :func:`niapy.algorithm.basic.CenterParticleSwarmOptimization.set_parameters`

        """
        kwargs.pop('min_velocity', None), kwargs.pop('max_velocity', None)
        ParticleSwarmAlgorithm.set_parameters(self, min_velocity=-np.inf, max_velocity=np.inf, **kwargs)
        self.num_mutations = num_mutations

    def get_parameters(self):
        r"""Get value of parameters for this instance of algorithm.

        Returns:
            Dict[str, Union[int, float, numpy.ndarray]]: Dictionary which has parameters mapped to values.

        See Also:
            * :func:`niapy.algorithms.basic.CenterParticleSwarmOptimization.get_parameters`

        """
        d = CenterParticleSwarmOptimization.get_parameters(self)
        d.update({'num_mutations': self.num_mutations})
        return d

    def run_iteration(self, task, pop, fpop, xb, fxb, **params):
        r"""Core function of algorithm.

        Args:
            task (Task): Optimization task.
            pop (numpy.ndarray): Current population of particles.
            fpop (numpy.ndarray): Current particles function/fitness values.
            xb (numpy.ndarray): Current global best particle.
            fxb (float: Current global best particles function/fitness value.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, float, list, dict]:
                1. New population of particles.
                2. New populations function/fitness values.
                3. New global best particle.
                4. New global best particle function/fitness value.
                5. Additional arguments.
                6. Additional keyword arguments.

        See Also:
            * :func:`niapy.algorithm.basic.WeightedVelocityClampingParticleSwarmAlgorithm.run_iteration`

        """
        pop, fpop, xb, fxb, d = CenterParticleSwarmOptimization.run_iteration(self, task, pop, fpop, xb, fxb, **params)
        v = d['v']
        v_a = (np.sum(v, axis=0) / len(v))
        v_a = v_a / np.max(np.abs(v_a))
        for _ in range(self.num_mutations):
            g = task.repair(xb + v_a * self.uniform(task.lower, task.upper), self.rng)
            fg = task.eval(g)
            if fg <= fxb:
                xb, fxb = g, fg
        return pop, fpop, xb, fxb, d


class MutatedCenterUnifiedParticleSwarmOptimization(MutatedCenterParticleSwarmOptimization):
    r"""Implementation of Mutated Particle Swarm Optimization.

    Algorithm:
        Mutated Center Unified Particle Swarm Optimization

    Date:
        2019

    Authors:
        Klemen Berkovič

    License:
        MIT

    Reference paper:
        Tsai, Hsing-Chih. "Unified particle swarm delivers high efficiency to particle swarm optimization." Applied Soft Computing 55 (2017): 371-383.

    Attributes:
        Name (List[str]): Names of algorithm.

    See Also:
        * :class:`niapy.algorithms.basic.CenterParticleSwarmOptimization`

    """

    Name = ['MutatedCenterUnifiedParticleSwarmOptimization', 'MCUPSO']

    @staticmethod
    def info():
        r"""Get basic information of algorithm.

        Returns:
            str: Basic information of algorithm.

        See Also:
            * :func:`niapy.algorithms.Algorithm.info`

        """
        return r"""Tsai, Hsing-Chih. "Unified particle swarm delivers high efficiency to particle swarm optimization." Applied Soft Computing 55 (2017): 371-383."""

    def update_velocity(self, v, p, pb, gb, w, min_velocity, max_velocity, task, **kwargs):
        r"""Update particle velocity.

        Args:
            v (numpy.ndarray): Current velocity of particle.
            p (numpy.ndarray): Current position of particle.
            pb (numpy.ndarray): Personal best position of particle.
            gb (numpy.ndarray): Global best position of particle.
            w (numpy.ndarray): Weights for velocity adjustment.
            min_velocity (numpy.ndarray): Minimal velocity allowed.
            max_velocity (numpy.ndarray): Maximal velocity allowed.
            task (Task): Optimization task.
            kwargs (dict): Additional arguments.

        Returns:
            numpy.ndarray: Updated velocity of particle.

        """
        r3 = self.random(task.dimension)
        return self.repair(
            w * v + self.c1 * self.random(task.dimension) * (pb - p) * r3 + self.c2 * self.random(task.dimension) * (
                        gb - p) * (1 - r3),
            min_velocity, max_velocity)


class ComprehensiveLearningParticleSwarmOptimizer(ParticleSwarmAlgorithm):
    r"""Implementation of Mutated Particle Swarm Optimization.

    Algorithm:
        Comprehensive Learning Particle Swarm Optimizer

    Date:
        2019

    Authors:
        Klemen Berkovič

    License:
        MIT

    Reference paper:
        J. J. Liang, a. K. Qin, P. N. Suganthan and S. Baskar, "Comprehensive learning particle swarm optimizer for global optimization of multimodal functions," in IEEE Transactions on Evolutionary Computation, vol. 10, no. 3, pp. 281-295, June 2006. doi: 10.1109/TEVC.2005.857610

    Reference URL:
        http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1637688&isnumber=34326

    Attributes:
        w0 (float): Inertia weight.
        w1 (float): Inertia weight.
        c (float): Velocity constant.
        m (int): Refresh rate.

    See Also:
        * :class:`niapy.algorithms.basic.ParticleSwarmAlgorithm`

    """

    Name = ['ComprehensiveLearningParticleSwarmOptimizer', 'CLPSO']

    @staticmethod
    def info():
        r"""Get basic information of algorithm.

        Returns:
            str: Basic information of algorithm.

        See Also:
            * :func:`niapy.algorithms.Algorithm.info`

        """
        return r"""J. J. Liang, a. K. Qin, P. N. Suganthan and S. Baskar, "Comprehensive learning particle swarm optimizer for global optimization of multimodal functions," in IEEE Transactions on Evolutionary Computation, vol. 10, no. 3, pp. 281-295, June 2006. doi: 10.1109/TEVC.2005.857610	"""

    def __init__(self, m=10, w0=.9, w1=.4, c=1.49445, *args, **kwargs):
        """Initialize CLPSO."""
        super().__init__(*args, **kwargs)
        self.m = m
        self.w0 = w0
        self.w1 = w1
        self.c = c

    def set_parameters(self, m=10, w0=.9, w1=.4, c=1.49445, **kwargs):
        r"""Set Particle Swarm Algorithm main parameters.

        Args:
            w0 (int): Inertia weight.
            w1 (float): Inertia weight.
            c (float): Velocity constant.
            m (float): Refresh rate.
            kwargs (dict): Additional arguments

        See Also:
            * :func:`niapy.algorithms.basic.ParticleSwarmAlgorithm.set_parameters`

        """
        ParticleSwarmAlgorithm.set_parameters(self, **kwargs)
        self.m = m
        self.w0 = w0
        self.w1 = w1
        self.c = c

    def get_parameters(self):
        r"""Get value of parameters for this instance of algorithm.

        Returns:
            Dict[str, Union[int, float, numpy.ndarray]]: Dictionary which has parameters mapped to values.

        See Also:
            * :func:`niapy.algorithms.basic.ParticleSwarmAlgorithm.get_parameters`

        """
        d = ParticleSwarmAlgorithm.get_parameters(self)
        d.update({
            'm': self.m,
            'w0': self.w0,
            'w1': self.w1,
            'c': self.c
        })
        return d

    def init(self, task):
        r"""Initialize dynamic arguments of Particle Swarm Optimization algorithm.

        Args:
            task (Task): Optimization task.

        Returns:
            Dict[str, numpy.ndarray]:
                * vMin: Minimal velocity.
                * vMax: Maximal velocity.
                * V: Initial velocity of particle.
                * flag: Refresh gap counter.

        """
        return {'min_velocity': full_array(self.min_velocity, task.dimension),
                'max_velocity': full_array(self.max_velocity, task.dimension),
                'v': np.full([self.population_size, task.dimension], 0.0), 'flag': np.full(self.population_size, 0),
                'pc': np.asarray(
                    [.05 + .45 * (np.exp(10 * (i - 1) / (self.population_size - 1)) - 1) / (np.exp(10) - 1) for i in
                     range(self.population_size)])}

    def generate_personal_best_cl(self, i, pc, personal_best, personal_best_fitness):
        r"""Generate new personal best position for learning.

        Args:
            i (int): Current particle.
            pc (float): Learning probability.
            personal_best (numpy.ndarray): Personal best positions for population.
            personal_best_fitness (numpy.ndarray): Personal best positions function/fitness values for personal best position.

        Returns:
            numpy.ndarray: Personal best for learning.

        """
        pbest = []
        for j in range(len(personal_best[i])):
            if self.random() > pc:
                pbest.append(personal_best[i, j])
            else:
                r1, r2 = int(self.random() * len(personal_best)), int(self.random() * len(personal_best))
                if personal_best_fitness[r1] < personal_best_fitness[r2]:
                    pbest.append(personal_best[r1, j])
                else:
                    pbest.append(personal_best[r2, j])
        return np.asarray(pbest)

    def update_velocity_cl(self, v, p, pb, w, min_velocity, max_velocity, task, **_kwargs):
        r"""Update particle velocity.

        Args:
            v (numpy.ndarray): Current velocity of particle.
            p (numpy.ndarray): Current position of particle.
            pb (numpy.ndarray): Personal best position of particle.
            w (numpy.ndarray): Weights for velocity adjustment.
            min_velocity (numpy.ndarray): Minimal velocity allowed.
            max_velocity (numpy.ndarray): Maximal velocity allowed.
            task (Task): Optimization task.

        Returns:
            numpy.ndarray: Updated velocity of particle.

        """
        return self.repair(w * v + self.c * self.random(task.dimension) * (pb - p), min_velocity, max_velocity)

    def run_iteration(self, task, pop, fpop, xb, fxb, **params):
        r"""Core function of algorithm.

        Args:
            task (Task): Optimization task.
            pop (numpy.ndarray): Current populations.
            fpop (numpy.ndarray): Current population fitness/function values.
            xb (numpy.ndarray): Current best particle.
            fxb (float): Current best particle fitness/function value.
            params (dict): Additional function keyword arguments.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, list, dict]:
                1. New population.
                2. New population fitness/function values.
                3. New global best position.
                4. New global best positions function/fitness value.
                5. Additional arguments.
                6. Additional keyword arguments:
                    * personal_best: Particles best population.
                    * personal_best_fitness: Particles best positions function/fitness value.
                    * min_velocity: Minimal velocity.
                    * max_velocity: Maximal velocity.
                    * V: Initial velocity of particle.
                    * flag: Refresh gap counter.
                    * pc: Learning rate.

        See Also:
            * :class:`niapy.algorithms.basic.ParticleSwarmAlgorithm.run_iteration`

        """
        personal_best = params.pop('personal_best')
        personal_best_fitness = params.pop('personal_best_fitness')
        min_velocity = params.pop('min_velocity')
        max_velocity = params.pop('max_velocity')
        v = params.pop('v')
        flag = params.pop('flag')
        pc = params.pop('pc')

        w = self.w0 * (self.w0 - self.w1) * (task.iters + 1) / task.max_iters
        for i in range(len(pop)):
            if flag[i] >= self.m:
                v[i] = self.update_velocity(v[i], pop[i], personal_best[i], xb, 1, min_velocity, max_velocity, task)
                pop[i] = task.repair(pop[i] + v[i], rng=self.rng)
                fpop[i] = task.eval(pop[i])
                if fpop[i] < personal_best_fitness[i]:
                    personal_best[i], personal_best_fitness[i] = pop[i].copy(), fpop[i]
                    if fpop[i] < fxb:
                        xb, fxb = pop[i].copy(), fpop[i]
                flag[i] = 0
            pbest = self.generate_personal_best_cl(i, pc[i], personal_best, personal_best_fitness)
            v[i] = self.update_velocity_cl(v[i], pop[i], pbest, w, min_velocity, max_velocity, task)
            pop[i] = pop[i] + v[i]
            if task.is_feasible(pop[i]):
                fpop[i] = task.eval(pop[i])
                if fpop[i] < personal_best_fitness[i]:
                    personal_best[i], personal_best_fitness[i] = pop[i].copy(), fpop[i]
                    if fpop[i] < fxb:
                        xb, fxb = pop[i].copy(), fpop[i]
        return pop, fpop, xb, fxb, {'personal_best': personal_best, 'personal_best_fitness': personal_best_fitness,
                                    'min_velocity': min_velocity,
                                    'max_velocity': max_velocity, 'v': v, 'flag': flag, 'pc': pc}

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
