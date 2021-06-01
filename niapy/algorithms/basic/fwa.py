# encoding=utf8
import logging

import numpy as np

from niapy.algorithms.algorithm import Algorithm
from niapy.util.distances import euclidean
import niapy.util.repair as repair

logging.basicConfig()
logger = logging.getLogger('niapy.algorithms.basic')
logger.setLevel('INFO')

__all__ = ['FireworksAlgorithm', 'EnhancedFireworksAlgorithm', 'DynamicFireworksAlgorithm',
           'DynamicFireworksAlgorithmGauss', 'BareBonesFireworksAlgorithm']


class BareBonesFireworksAlgorithm(Algorithm):
    r"""Implementation of Bare Bones Fireworks Algorithm.

    Algorithm:
        Bare Bones Fireworks Algorithm

    Date:
        2018

    Authors:
        Klemen Berkovič

    License:
        MIT

    Reference URL:
        https://www.sciencedirect.com/science/article/pii/S1568494617306609

    Reference paper:
        Junzhi Li, Ying Tan, The bare bones fireworks algorithm: A minimalist global optimizer, Applied Soft Computing, Volume 62, 2018, Pages 454-462, ISSN 1568-4946, https://doi.org/10.1016/j.asoc.2017.10.046.

    Attributes:
        Name (List[str]): List of strings representing algorithm names
        num_sparks (int): Number of sparks
        amplification_coefficient (float): amplification coefficient
        reduction_coefficient (float): reduction coefficient

    """

    Name = ['BareBonesFireworksAlgorithm', 'BBFWA']

    @staticmethod
    def info():
        r"""Get default information of algorithm.

        Returns:
            str: Basic information.

        See Also:
            * :func:`niapy.algorithms.Algorithm.info`

        """
        return r"""Junzhi Li, Ying Tan, The bare bones fireworks algorithm: A minimalist global optimizer, Applied Soft Computing, Volume 62, 2018, Pages 454-462, ISSN 1568-4946, https://doi.org/10.1016/j.asoc.2017.10.046."""

    def __init__(self, num_sparks=10, amplification_coefficient=1.5, reduction_coefficient=0.5, *args, **kwargs):
        r"""Initialize BareBonesFireworksAlgorithm.

        Args:
            num_sparks (int): Number of sparks :math:`\in[1, \infty)`.
            amplification_coefficient (float): Amplification coefficient :math:`\in [1, \infty)`.
            reduction_coefficient (float): Reduction coefficient :math:`\in (0, 1)`.

        """
        kwargs.pop('population_size', None)
        super().__init__(1, *args, **kwargs)
        self.num_sparks = num_sparks
        self.amplification_coefficient = amplification_coefficient
        self.reduction_coefficient = reduction_coefficient

    def set_parameters(self, num_sparks=10, amplification_coefficient=1.5, reduction_coefficient=0.5, **kwargs):
        r"""Set the arguments of an algorithm.

        Args:
            num_sparks (int): Number of sparks :math:`\in [1, \infty)`.
            amplification_coefficient (float): Amplification coefficient :math:`\in [1, \infty)`.
            reduction_coefficient (float): Reduction coefficient :math:`\in (0, 1)`.

        """
        kwargs.pop('population_size', None)
        super().set_parameters(population_size=1, **kwargs)
        self.num_sparks = num_sparks
        self.amplification_coefficient = amplification_coefficient
        self.reduction_coefficient = reduction_coefficient

    def init_population(self, task):
        r"""Initialize starting population.

        Args:
            task (Task): Optimization task.

        Returns:
            Tuple[numpy.ndarray, float, Dict[str, Any]]:
                1. Initial solution.
                2. Initial solution function/fitness value.
                3. Additional arguments:
                    * A (numpy.ndarray): Starting amplitude or search range.

        """
        x, x_fit, d = super().init_population(task)
        d.update({'amplitude': task.range})
        return x, x_fit, d

    def run_iteration(self, task, population, population_fitness, best_x, best_fitness, **params):
        r"""Core function of Bare Bones Fireworks Algorithm.

        Args:
            task (Task): Optimization task.
            population (numpy.ndarray): Current solution.
            population_fitness (float): Current solution fitness/function value.
            best_x (numpy.ndarray): Current best solution.
            best_fitness (float): Current best solution fitness/function value.
            params (Dict[str, Any]): Additional parameters.

        Returns:
            Tuple[numpy.ndarray, float, numpy.ndarray, float, Dict[str, Any]]:
                1. New solution.
                2. New solution fitness/function value.
                3. New global best solution.
                4. New global best solutions fitness/objective value.
                5. Additional arguments:
                    * amplitude (numpy.ndarray): Search range.

        """
        amplitude = params.pop('amplitude')

        sparks = self.uniform(population - amplitude, population + amplitude, (self.num_sparks, task.dimension))
        sparks = np.apply_along_axis(task.repair, 1, sparks, self.rng)
        sparks_fitness = np.apply_along_axis(task.eval, 1, sparks)
        best_index = np.argmin(sparks_fitness)
        if sparks_fitness[best_index] < population_fitness:
            population = sparks[best_index]
            population_fitness = sparks_fitness[best_index]
            amplitude = self.amplification_coefficient * amplitude
        else:
            amplitude = self.reduction_coefficient * amplitude
        return population, population_fitness, population.copy(), population_fitness, {'amplitude': amplitude}


class FireworksAlgorithm(Algorithm):
    r"""Implementation of fireworks algorithm.

    Algorithm:
        Fireworks Algorithm

    Date:
        2018

    Authors:
        Klemen Berkovič

    License:
        MIT

    Reference URL:
        https://www.springer.com/gp/book/9783662463529

    Reference paper:
        Tan, Ying. "Fireworks algorithm." Heidelberg, Germany: Springer 10 (2015): 978-3

    Attributes:
        Name (List[str]): List of strings representing algorithm names.

    """

    Name = ['FireworksAlgorithm', 'FWA']

    @staticmethod
    def info():
        r"""Get default information of algorithm.

        Returns:
            str: Basic information.

        See Also:
            * :func:`niapy.algorithms.Algorithm.info`

        """
        return r"""Tan, Ying. "Fireworks algorithm." Heidelberg, Germany: Springer 10 (2015): 978-3."""

    def __init__(self, population_size=5, num_sparks=50, a=0.04, b=0.8, max_amplitude=40, num_gaussian=5, *args, **kwargs):
        """Initialize FWA.

        Args:
            population_size (int): Number of Fireworks
            num_sparks (int): Number of sparks
            a (float): Limitation of sparks
            b (float): Limitation of sparks
            max_amplitude (float): Initial amplitude.
            num_gaussian (int): Number of sparks to apply gaussian mutation to.

        """
        super().__init__(population_size, *args, **kwargs)
        self.num_sparks = num_sparks
        self.a = a
        self.b = b
        self.max_amplitude = max_amplitude
        self.num_gaussian = num_gaussian
        self.epsilon = np.finfo(float).eps

    def set_parameters(self, population_size=5, num_sparks=50, a=0.04, b=0.8, max_amplitude=40, num_gaussian=5,
                       **kwargs):
        r"""Set the arguments of an algorithm.

        Args:
            population_size (int): Number of Fireworks
            num_sparks (int): Number of sparks
            a (float): Limitation of sparks
            b (float): Limitation of sparks
            max_amplitude (float): Initial amplitude.
            num_gaussian (int): Number of sparks to apply gaussian mutation to.

        """
        super().set_parameters(population_size=population_size, **kwargs)
        self.num_sparks = num_sparks
        self.a = a
        self.b = b
        self.max_amplitude = max_amplitude
        self.num_gaussian = num_gaussian
        self.epsilon = np.finfo(float).eps

    def sparks_num(self, population_fitness):
        r"""Calculate number of sparks.

        Args:
            population_fitness (numpy.ndarray): Population fitness values.

        Returns:
            numpy.ndarray: Number of sparks that for all fireworks.

        """
        worst_fitness = np.amax(population_fitness)
        sparks_num = self.num_sparks * (worst_fitness - population_fitness + self.epsilon)
        sparks_num /= np.sum(worst_fitness - population_fitness) + self.epsilon

        cond = [sparks_num < self.a * self.num_sparks, (sparks_num > self.b * self.num_sparks) * (self.a < self.b < 1)]
        choices = [round(self.a * self.num_sparks), round(self.b * self.num_sparks)]
        return np.select(cond, choices, default=np.round(sparks_num)).astype(int)

    def explosion_amplitudes(self, population_fitness, task=None):
        r"""Calculate explosion amplitude.

        Args:
            population_fitness (numpy.ndarray): Population fitness values.
            task (Optional[Task]): Optimization task (Unused in this version of the algorithm).

        Returns:
            numpy.ndarray: Explosion amplitude of sparks.

        """
        best_fitness = np.amin(population_fitness)
        amplitudes = self.max_amplitude * (population_fitness - best_fitness + self.epsilon)
        amplitudes /= np.sum(population_fitness - best_fitness) + self.epsilon
        return amplitudes

    def explosion_spark(self, x, amplitude, task):
        r"""Explode a spark.

        Args:
            x (numpy.ndarray): Individuals creating spark.
            amplitude (float): Amplitude of spark.
            task (Task): Optimization task.

        Returns:
            numpy.ndarray: Sparks exploded in with specified amplitude.

        """
        z = self.rng.choice(task.dimension, self.rng.integers(task.dimension), replace=False)
        x[z] = x[z] + amplitude * self.uniform(-1, 1)
        return self.mapping(x, task)

    def gaussian_spark(self, x, task, best_x=None):
        r"""Create gaussian spark.

        Args:
            x (numpy.ndarray): Individual creating a spark.
            task (Task): Optimization task.
            best_x (numpy.ndarray): Current best individual. Unused in this version of the algorithm.

        Returns:
            numpy.ndarray: Spark exploded based on gaussian amplitude.

        """
        z = self.rng.choice(task.dimension, self.rng.integers(task.dimension), replace=False)
        x[z] = x[z] * self.normal(1, 1)
        return self.mapping(x, task)

    def mapping(self, x, task):
        r"""Fix value to bounds.

        Args:
            x (numpy.ndarray): Individual to fix.
            task (Task): Optimization task.

        Returns:
            numpy.ndarray: Individual in search range.

        """
        return repair.reflect(x, task.lower, task.upper)

    def selection(self, population, population_fitness, sparks, task):
        r"""Generate new generation of individuals.

        Args:
            population (numpy.ndarray): Current population.
            population_fitness (numpy.ndarray[float]): Currents population fitness/function values.
            sparks (numpy.ndarray): New population.
            task (Task): Optimization task.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray[float], numpy.ndarray, float]:
                1. New population.
                2. New populations fitness/function values.
                3. New global best individual.
                4. New global best fitness.

        """
        sparks_fitness = np.apply_along_axis(task.eval, 1, sparks)
        best_index = np.argmin(sparks_fitness)
        best_x = sparks[best_index].copy()
        best_fitness = sparks_fitness[best_index]

        all_sparks = np.delete(sparks, best_index, axis=0)
        fitness = np.delete(sparks_fitness, best_index)

        distances = np.sum(euclidean(all_sparks[:, np.newaxis, :], all_sparks[np.newaxis, :, :]), axis=0)
        probabilities = distances / np.sum(distances)

        selected_indices = self.rng.choice(len(all_sparks), self.population_size - 1, replace=False, p=probabilities)

        population[0] = best_x
        population[1:] = all_sparks[selected_indices]
        population_fitness[0] = best_fitness
        population_fitness[1:] = fitness[selected_indices]
        return population, population_fitness, best_x, best_fitness

    def run_iteration(self, task, population, population_fitness, best_x, best_fitness, **params):
        r"""Core function of Fireworks algorithm.

        Args:
            task (Task): Optimization task.
            population (numpy.ndarray): Current population.
            population_fitness (numpy.ndarray[float]): Current populations function/fitness values.
            best_x (numpy.ndarray): Global best individual.
            best_fitness (float): Global best individuals fitness/function value.
            **params (Dict[str, Any)]: Additional arguments

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, float, Dict[str, Any]]:
                1. Initialized population.
                2. Initialized populations function/fitness values.
                3. New global best solution.
                4. New global best solutions fitness/objective value.
                5. Additional arguments:
                    * Ah (numpy.ndarray): Initialized amplitudes.

        See Also:
            * :func:`FireworksAlgorithm.sparks_num`.
            * :func:`FireworksAlgorithm.explosion_amplitudes`
            * :func:`FireworksAlgorithm.explosion_spark`
            * :func:`FireworksAlgorithm.gaussian_spark`
            * :func:`FireworksAlgorithm.selection`

        """
        sparks_num = self.sparks_num(population_fitness)
        amplitudes = self.explosion_amplitudes(population_fitness, task=task)

        all_sparks = population.copy()
        for i in range(self.population_size):
            si = sparks_num[i]
            ai = amplitudes[i]

            sparks_i = np.empty((si, task.dimension))
            for s in range(si):
                sparks_i[s] = population[i]
                sparks_i[s] = self.explosion_spark(sparks_i[s], ai, task)
            all_sparks = np.concatenate((all_sparks, sparks_i), axis=0)

        gaussian_idx = self.rng.choice(len(all_sparks), self.num_gaussian, replace=False)
        gaussian_sparks = np.array(all_sparks[gaussian_idx])

        for i in range(self.num_gaussian):
            gaussian_sparks[i] = self.gaussian_spark(gaussian_sparks[i], task, best_x=best_x)

        all_sparks = np.concatenate((all_sparks, gaussian_sparks), axis=0)

        population, population_fitness, best_x, best_fitness = self.selection(population, population_fitness,
                                                                              all_sparks, task)

        return population, population_fitness, best_x, best_fitness, {}


class EnhancedFireworksAlgorithm(FireworksAlgorithm):
    r"""Implementation of enhanced fireworks algorithm.

    Algorithm:
        Enhanced Fireworks Algorithm

    Date:
        2018

    Authors:
        Klemen Berkovič

    License:
        MIT

    Reference URL:
        https://ieeexplore.ieee.org/document/6557813/

    Reference paper:
        S. Zheng, A. Janecek and Y. Tan, "Enhanced Fireworks Algorithm," 2013 IEEE Congress on Evolutionary Computation, Cancun, 2013, pp. 2069-2077. doi: 10.1109/CEC.2013.6557813

    Attributes:
        Name (List[str]): List of strings representing algorithm names.
        amplitude_init (float): Initial amplitude of sparks.
        amplitude_final (float): Maximal amplitude of sparks.

    """

    Name = ['EnhancedFireworksAlgorithm', 'EFWA']

    @staticmethod
    def info():
        r"""Get default information of algorithm.

        Returns:
            str: Basic information.

        See Also:
            * :func:`niapy.algorithms.Algorithm.info`

        """
        return r"""S. Zheng, A. Janecek and Y. Tan, "Enhanced Fireworks Algorithm," 2013 IEEE Congress on Evolutionary Computation, Cancun, 2013, pp. 2069-2077. doi: 10.1109/CEC.2013.6557813"""

    def __init__(self, amplitude_init=0.2, amplitude_final=0.01, *args, **kwargs):
        """Initialize EFWA.

        Args:
            amplitude_init (float): Initial amplitude.
            amplitude_final (float): Final amplitude.

        See Also:
            * :func:`FireworksAlgorithm.__init__`

        """
        super().__init__(*args, **kwargs)
        self.amplitude_init = amplitude_init
        self.amplitude_final = amplitude_final

    def set_parameters(self, amplitude_init=0.2, amplitude_final=0.01, **kwargs):
        r"""Set EnhancedFireworksAlgorithm algorithms core parameters.

        Args:
            amplitude_init (float): Initial amplitude.
            amplitude_final (float): Final amplitude.

        See Also:
            * :func:`FireworksAlgorithm.set_parameters`

        """
        super().set_parameters(**kwargs)
        self.amplitude_init = amplitude_init
        self.amplitude_final = amplitude_final

    def explosion_amplitudes(self, population_fitness, task=None):
        r"""Calculate explosion amplitude.

        Args:
            population_fitness (numpy.ndarray):
            task (Task): Optimization task.

        Returns:
            numpy.ndarray: New amplitude.

        """
        amplitudes = super().explosion_amplitudes(population_fitness, task)
        a_min = self.amplitude_init - np.sqrt(task.evals * (2 * task.max_evals - task.evals)) * (self.amplitude_init - self.amplitude_final) / task.max_evals
        amplitudes[amplitudes < a_min] = a_min
        return amplitudes

    def mapping(self, x, task):
        r"""Fix value to bounds.

        Args:
            x (numpy.ndarray): Individual to fix.
            task (Task): Optimization task.

        Returns:
            numpy.ndarray: Individual in search range.

        """
        return repair.rand(x, task.lower, task.upper, rng=self.rng)

    def explosion_spark(self, x, amplitude, task):
        r"""Explode a spark.

        Args:
            x (numpy.ndarray): Individuals creating spark.
            amplitude (float): Amplitude of spark.
            task (Task): Optimization task.

        Returns:
            numpy.ndarray: Sparks exploded in with specified amplitude.

        """
        z = self.rng.choice(task.dimension, self.rng.integers(task.dimension), replace=False)
        if isinstance(amplitude, np.ndarray):
            x[z] = x[z] + amplitude[z] * self.uniform(-1, 1, len(z))
        else:
            x[z] = x[z] + amplitude * self.uniform(-1, 1, len(z))
        return self.mapping(x, task)

    def gaussian_spark(self, x, task, best_x=None):
        r"""Create new individual.

        Args:
            x (numpy.ndarray):
            task (Task): Optimization task.
            best_x (numpy.ndarray): Current global best individual.

        Returns:
            numpy.ndarray: New individual generated by gaussian noise.

        """
        z = self.rng.choice(task.dimension, self.rng.integers(task.dimension), replace=False)
        e = self.standard_normal()
        x[z] = x[z] + (best_x[z] - x[z]) * e
        return self.mapping(x, task)

    def selection(self, population, population_fitness, sparks, task):
        r"""Generate new population.

        Args:
            population (numpy.ndarray): Current population.
            population_fitness (numpy.ndarray[float]): Current populations fitness/function values.
            sparks (numpy.ndarray): New population.
            task (Task): Optimization task.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray[float], numpy.ndarray, float]:
                1. New population.
                2. New populations fitness/function values.
                3. New global best individual.
                4. New global best fitness.

        """
        sparks_fitness = np.apply_along_axis(task.eval, 1, sparks)
        ib = np.argmin(sparks_fitness)
        best_x = sparks[ib].copy()
        best_fitness = sparks_fitness[ib]
        if sparks_fitness[ib] < population_fitness[0]:
            population[0], population_fitness[0] = best_x, best_fitness
        for i in range(1, self.population_size):
            r = self.integers(len(sparks))
            if sparks_fitness[r] < population_fitness[i]:
                population[i], population_fitness[i] = sparks[r], sparks_fitness[r]
        return population, population_fitness, best_x, best_fitness


class DynamicFireworksAlgorithmGauss(EnhancedFireworksAlgorithm):
    r"""Implementation of dynamic fireworks algorithm.

    Algorithm:
        Dynamic Fireworks Algorithm

    Date:
        2018

    Authors:
        Klemen Berkovič

    License:
        MIT

    Reference URL:
        http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6900485&isnumber=6900223

    Reference paper:
        S. Zheng, A. Janecek, J. Li and Y. Tan, "Dynamic search in fireworks algorithm," 2014 IEEE Congress on Evolutionary Computation (CEC), Beijing, 2014, pp. 3222-3229. doi: 10.1109/CEC.2014.6900485

    Attributes:
        Name (List[str]): List of strings representing algorithm names.
        amplitude_cf (Union[float, int]): Amplitude of the core firework.
        amplification_coeff (Union[float, int]): Amplification coefficient.
        reduction_coeff (Union[float, int]): Reduction coefficient.

    """

    Name = ['DynamicFireworksAlgorithmGauss', 'dynFWAG']

    @staticmethod
    def info():
        r"""Get default information of algorithm.

        Returns:
            str: Basic information.

        See Also:
            * :func:`niapy.algorithms.Algorithm.info`

        """
        return r"""S. Zheng, A. Janecek, J. Li and Y. Tan, "Dynamic search in fireworks algorithm," 2014 IEEE Congress on Evolutionary Computation (CEC), Beijing, 2014, pp. 3222-3229. doi: 10.1109/CEC.2014.6900485"""

    def __init__(self, amplification_coeff=1.2, reduction_coeff=0.9, *args, **kwargs):
        """Initialize dynFWAG.

        Args:
            amplification_coeff (Union[int, float]): Amplification coefficient.
            reduction_coeff (Union[int, float]): Reduction coefficient.

        See Also:
            * :func:`FireworksAlgorithm.__init__`

        """
        super().__init__(*args, **kwargs)
        self.amplification_coeff = amplification_coeff
        self.reduction_coeff = reduction_coeff

    def set_parameters(self, amplification_coeff=1.2, reduction_coeff=0.9, **kwargs):
        r"""Set core arguments of DynamicFireworksAlgorithmGauss.

        Args:
            amplification_coeff (Union[int, float]): Amplification coefficient.
            reduction_coeff (Union[int, float]): Reduction coefficient.

        See Also:
            * :func:`FireworksAlgorithm.set_parameters`

        """
        super().set_parameters(**kwargs)
        self.amplification_coeff = amplification_coeff
        self.reduction_coeff = reduction_coeff

    def update_cf(self, xnb, xcb, xcb_f, xb, xb_f, amplitude_cf, task):
        r"""Update the core firework.

        Args:
            xnb: Sparks generated by core fireworks.
            xcb: Current generations best spark.
            xcb_f: Current generations best fitness.
            xb: Global best individual.
            xb_f: Global best fitness.
            amplitude_cf: Amplitude of the core firework.
            task (Task): Optimization task.

        Returns:
            Tuple[numpy.ndarray, float, numpy.ndarray]:
                1. New core firework.
                2. New core firework's fitness.
                3. New core firework amplitude.

        """
        xnb_f = np.apply_along_axis(task.eval, 1, xnb)
        ib_f = np.argmin(xnb_f)
        if xnb_f[ib_f] <= xb_f:
            xb, xb_f = xnb[ib_f], xnb_f[ib_f]
        if xb_f >= xcb_f:
            xb, xb_f, amplitude_cf = xcb, xcb_f, amplitude_cf * self.amplification_coeff
        else:
            amplitude_cf = amplitude_cf * self.reduction_coeff
        return xb, xb_f, amplitude_cf

    def explosion_amplitudes(self, population_fitness, task=None):
        """Calculate explosion amplitude for other fireworks."""
        return FireworksAlgorithm.explosion_amplitudes(self, population_fitness)

    def init_population(self, task):
        r"""Initialize population.

        Args:
            task (Task): Optimization task.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray[float], Dict[str, Any]]:
                1. Initialized population.
                2. Initialized population function/fitness values.
                3. Additional arguments:
                    * amplitude_cf (numpy.ndarray): Initial amplitude of the core firework.

        """
        fireworks, fitnesses, _ = super().init_population(task)
        amplitude_cf = task.range
        return fireworks, fitnesses, {'amplitude_cf': amplitude_cf}

    def selection(self, population, population_fitness, sparks, task):
        """Select fireworks for the next generation."""
        sparks_fitness = np.apply_along_axis(task.eval, 1, sparks)
        ib = np.argmin(sparks_fitness)
        for i, f in enumerate(population_fitness):
            r = self.integers(len(sparks))
            if sparks_fitness[r] < f:
                population[i], population_fitness[i] = sparks[r], sparks_fitness[r]
        population[0], population_fitness[0] = sparks[ib], sparks_fitness[ib]
        return population, population_fitness

    def run_iteration(self, task, population, population_fitness, best_x, best_fitness, **params):
        r"""Core function of DynamicFireworksAlgorithmGauss algorithm.

        Args:
            task (Task): Optimization task.
            population (numpy.ndarray): Current population.
            population_fitness (numpy.ndarray): Current populations function/fitness values.
            best_x (numpy.ndarray): Global best individual.
            best_fitness (float): Global best fitness/function value.
            **params (Dict[str, Any]): Additional arguments.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, float, Dict[str, Any]]:
                1. New population.
                2. New populations fitness/function values.
                3. New global best solution.
                4. New global best solutions fitness/objective value.
                5. Additional arguments:
                    * amplitude_cf (numpy.ndarray): Amplitude of the core firework.

        """
        amplitude_cf = params.pop('amplitude_cf')

        sparks_num = self.sparks_num(population_fitness)
        amplitudes = self.explosion_amplitudes(population_fitness)

        cf_sparks_num = self.num_sparks * (np.amax(population_fitness) - best_fitness + self.epsilon) /\
            (np.sum(np.amax(population_fitness - best_fitness)) + self.epsilon)

        if cf_sparks_num < self.a * self.num_sparks:
            cf_sparks_num = round(self.a * self.num_sparks)
        elif cf_sparks_num > self.b * self.num_sparks and self.a < self.b < 1:
            cf_sparks_num = round(self.b * self.num_sparks)
        else:
            cf_sparks_num = round(cf_sparks_num)

        all_sparks = population.copy()
        for i in range(self.population_size):
            si = sparks_num[i]
            ai = amplitudes[i]

            sparks_i = np.empty((si, task.dimension))
            for s in range(si):
                sparks_i[s] = population[i]
                sparks_i[s] = self.explosion_spark(sparks_i[s], ai, task)
            all_sparks = np.concatenate((all_sparks, sparks_i), axis=0)

        gaussian_idx = self.rng.choice(len(all_sparks), self.num_gaussian, replace=False)
        gaussian_sparks = np.array(all_sparks[gaussian_idx])

        for i in range(self.num_gaussian):
            gaussian_sparks[i] = self.gaussian_spark(gaussian_sparks[i], task, best_x=best_x)

        all_sparks = np.concatenate((all_sparks, gaussian_sparks), axis=0)

        cf_sparks = np.empty((cf_sparks_num, task.dimension))
        for s in range(cf_sparks_num):
            cf_sparks[s] = best_x
            cf_sparks[s] = self.explosion_spark(cf_sparks[s], amplitude_cf, task)

        population, population_fitness = self.selection(population, population_fitness, all_sparks, task)
        best_x, best_fitness, amplitude_cf = self.update_cf(cf_sparks, population[0], population_fitness[0], best_x, best_fitness, amplitude_cf, task)
        return population, population_fitness, best_x, best_fitness, {'amplitude_cf': amplitude_cf}


class DynamicFireworksAlgorithm(DynamicFireworksAlgorithmGauss):
    r"""Implementation of dynamic fireworks algorithm.

    Algorithm:
        Dynamic Fireworks Algorithm

    Date:
        2018

    Authors:
        Klemen Berkovič

    License:
        MIT

    Reference URL:
        http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6900485&isnumber=6900223

    Reference paper:
        S. Zheng, A. Janecek, J. Li and Y. Tan, "Dynamic search in fireworks algorithm," 2014 IEEE Congress on Evolutionary Computation (CEC), Beijing, 2014, pp. 3222-3229. doi: 10.1109/CEC.2014.6900485

    Attributes:
        Name (List[str]): List of strings representing algorithm name.

    See Also:
        * :class:`niapy.algorithms.basic.DynamicFireworksAlgorithmGauss`

    """

    Name = ['DynamicFireworksAlgorithm', 'dynFWA']

    @staticmethod
    def info():
        r"""Get default information of algorithm.

        Returns:
            str: Basic information.

        See Also:
            * :func:`niapy.algorithms.Algorithm.info`

        """
        return r"""S. Zheng, A. Janecek, J. Li and Y. Tan, "Dynamic search in fireworks algorithm," 2014 IEEE Congress on Evolutionary Computation (CEC), Beijing, 2014, pp. 3222-3229. doi: 10.1109/CEC.2014.6900485"""

    def run_iteration(self, task, population, population_fitness, best_x, best_fitness, **params):
        r"""Co50re function of Dynamic Fireworks Algorithm.

        Args:
            task (Task): Optimization task
            population (numpy.ndarray): Current population
            population_fitness (numpy.ndarray[float]): Current population fitness/function values
            best_x (numpy.ndarray): Current best solution
            best_fitness (float): Current best solution's fitness/function value
            **params:

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray[float], Dict[str, Any]]:
                1. New population.
                2. New population function/fitness values.
                3. New global best solution.
                4. New global best fitness.
                5. Additional arguments.

        """
        amplitude_cf = params.pop('amplitude_cf')

        sparks_num = self.sparks_num(population_fitness)
        amplitudes = self.explosion_amplitudes(population_fitness)

        cf_sparks_num = self.num_sparks * (np.amax(population_fitness) - best_fitness + self.epsilon) / \
            (np.sum(np.amax(population_fitness - best_fitness)) + self.epsilon)

        if cf_sparks_num < self.a * self.num_sparks:
            cf_sparks_num = round(self.a * self.num_sparks)
        elif cf_sparks_num > self.b * self.num_sparks and self.a < self.b < 1:
            cf_sparks_num = round(self.b * self.num_sparks)
        else:
            cf_sparks_num = round(cf_sparks_num)

        all_sparks = population.copy()
        for i in range(self.population_size):
            si = sparks_num[i]
            ai = amplitudes[i]

            sparks_i = np.empty((si, task.dimension))
            for s in range(si):
                sparks_i[s] = population[i]
                sparks_i[s] = self.explosion_spark(sparks_i[s], ai, task)
            all_sparks = np.concatenate((all_sparks, sparks_i), axis=0)

        cf_sparks = np.empty((cf_sparks_num, task.dimension))
        for s in range(cf_sparks_num):
            cf_sparks[s] = best_x
            cf_sparks[s] = self.explosion_spark(cf_sparks[s], amplitude_cf, task)

        population, population_fitness = self.selection(population, population_fitness, all_sparks, task)
        best_x, best_fitness, amplitude_cf = self.update_cf(cf_sparks, population[0], population_fitness[0], best_x,
                                                            best_fitness, amplitude_cf, task)
        return population, population_fitness, best_x, best_fitness, {'amplitude_cf': amplitude_cf}

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
