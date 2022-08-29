# encoding=utf8
import logging

import numpy as np

from niapy.algorithms.algorithm import Individual
from niapy.algorithms.basic.de import DifferentialEvolution
from niapy.util import objects_to_array

logging.basicConfig()
logger = logging.getLogger('niapy.algorithms.modified')
logger.setLevel('INFO')

__all__ = [
    'parent_medium',
    'cross_curr2pbest1',
    'SolutionSHADE',
    'SuccessHistoryAdaptiveDifferentialEvolution',
    'LpsrSuccessHistoryAdaptiveDifferentialEvolution'
]


def parent_medium(x, p, lower, upper, **_kwargs):
    r"""Repair solution and put the solution to the medium of the parent's value.

    Args:
        x (numpy.ndarray): Solution to check and repair if needed.
        p (numpy.ndarray): The parent of the solution.
        lower (numpy.ndarray): Lower bounds of search space.
        upper (numpy.ndarray): Upper bounds of search space.

    Returns:
        numpy.ndarray: Solution in search space.

    """
    ir = np.where(x < lower)    # values under the range are repaired to a medium of x_min and parents value
    x[ir] = (lower[ir] + p[ir]) / 2.0
    ir = np.where(x > upper)    # values over the range are repaired to a medium of x_max and parents value
    x[ir] = (upper[ir] + p[ir]) / 2.0
    return x


def cross_curr2pbest1(pop, ic, f, cr, rng, p_num, archive, arc_ind_cnt, task, **_kwargs):
    r"""Mutation strategy with crossover.

    Mutation:
        Name: current-to-pbest/1

        :math:`\mathbf{v}_{i, G} = \mathbf{x}_{i, G} + differential_weight \cdot (\mathbf{x}_{pbest, G} - \mathbf{x}_{i, G}) + differential_weight \cdot (\mathbf{x}_{r_1, G} - \mathbf{x}_{r_2, G})`
        where individual :math:`\mathbf{x}_{pbest, G}` is randomly selected from the top :math:`N \cdot pbest_factor (pbest_factor \in [0,1])` current population members,
        :math:`r_1` is an index representing a random current population member and :math:`r_2` is an index representing a random member of :math:`N_{G} \cup A`

    Crossover:
        Name: Binomial crossover

        :math:`\mathbf{u}_{j, i, G} = \begin{cases} \mathbf{v}_{j, i, G}, & \text{if $rand[0,1) \leq crossover_rate$ or $j=j_{rand}$}, \\ \mathbf{x}_{j, i, G}, & \text{otherwise}. \end{cases}`
        where :math:`j_{rand}` is an index representing a random problem dimension.

    Args:
        pop (numpy.ndarray[Individual]): Current population.
        ic (int): Index of individual being mutated.
        f (float): Scale factor.
        cr (float): Crossover probability.
        rng (numpy.random.Generator): Random generator.
        pbest_factor (float): Greediness factor.
        archive (numpy.ndarray): External archive.
        arc_ind_cnt (int): Number of individuals in the archive.
        task (Task): Optimization task.

    Returns:
        numpy.ndarray: mutated and mixed individual.

    """
    # Note: the population passed in the argument must be sorted by fitness!
    x_pbest = rng.integers(p_num)
    # a random individual is selected from the best p_num individuals of the population rng.integers
    p = [1 / (len(pop) - 1.0) if i != ic else 0 for i in range(len(pop))]
    r1 = rng.choice(len(pop), p=p)  # a random individual != to the current individual is selected from the population
    p = [1 / (len(pop) + arc_ind_cnt - 2.0) if i != ic and i != r1 else 0 for i in range(len(pop) + arc_ind_cnt)]
    r2 = rng.choice(len(pop) + arc_ind_cnt, p=p)
    # a second random individual != to the current individual and r1 is selected from the population U archive
    j = rng.integers(task.dimension)
    if r2 >= len(pop):
        r2 -= len(pop)
        v = [pop[ic][i] + f * (pop[x_pbest][i] - pop[ic][i]) + f * (pop[r1][i] - archive[r2][i])
             if rng.random() < cr or i == j else pop[ic][i] for i in range(task.dimension)]
        return parent_medium(np.asarray(v), pop[ic].x, task.lower, task.upper)
        # the mutant vector is repaired if needed

    else:
        v = [pop[ic][i] + f * (pop[x_pbest][i] - pop[ic][i]) + f * (pop[r1][i] - pop[r2][i])
             if rng.random() < cr or i == j else pop[ic][i] for i in range(task.dimension)]
        return parent_medium(np.asarray(v), pop[ic].x, task.lower, task.upper)
        # the mutant vector is repaired if needed


class SolutionSHADE(Individual):
    r"""Individual for SHADE algorithm.

    Attributes:
        differential_weight (float): Scale factor.
        crossover_probability (float): Crossover probability.

    See Also:
        :class:`niapy.algorithms.Individual`

    """

    def __init__(self, differential_weight=0.5, crossover_probability=0.5, **kwargs):
        r"""Initialize SolutionSHADE.

        Attributes:
            differential_weight (float): Scale factor.
            crossover_probability (float): Crossover probability.

        See Also:
            :func:`niapy.algorithm.Individual.__init__`

        """
        super().__init__(**kwargs)
        self.differential_weight = differential_weight
        self.crossover_probability = crossover_probability


class SuccessHistoryAdaptiveDifferentialEvolution(DifferentialEvolution):
    r"""Implementation of Success-history based adaptive differential evolution algorithm.

    Algorithm:
        Success-history based adaptive differential evolution algorithm

    Date:
        2022

    Author:
        Aleš Gartner

    License:
        MIT

    Reference paper:
        Ryoji Tanabe and Alex Fukunaga: Improving the Search Performance of SHADE Using Linear Population Size Reduction,  Proc. IEEE Congress on Evolutionary Computation (CEC-2014), Beijing, July, 2014.

    Attributes:
        Name (List[str]): List of strings representing algorithm name
        extern_arc_rate (float): External archive size factor.
        pbest_factor (float): Greediness factor for current-to-pbest/1 mutation.
        hist_mem_size (int): Size of historical memory.

    See Also:
        * :class:`niapy.algorithms.basic.DifferentialEvolution`

    """

    Name = ['SuccessHistoryAdaptiveDifferentialEvolution', 'SHADE']

    @staticmethod
    def info():
        r"""Get algorithm information.

        Returns:
            str: Algorithm information.

        See Also:
            * :func:`niapy.algorithms.Algorithm.info`

        """
        return r"""Ryoji Tanabe and Alex Fukunaga: Improving the Search Performance of SHADE Using Linear Population Size Reduction,  Proc. IEEE Congress on Evolutionary Computation (CEC-2014), Beijing, July, 2014."""

    def __init__(self, population_size=540, extern_arc_rate=2.6, pbest_factor=0.11, hist_mem_size=6, *args, **kwargs):
        """Initialize SHADE.

        Args:
            population_size (Optional[int]): Population size.
            extern_arc_rate (Optional[float]): External archive size factor.
            pbest_factor (Optional[float]): Greediness factor for current-to-pbest/1 mutation.
            hist_mem_size (Optional[int]): Size of historical memory.

        See Also:
            * :func:`niapy.algorithms.basic.DifferentialEvolution.__init__`

        """
        super().__init__(population_size, individual_type=kwargs.pop('individual_type', SolutionSHADE), *args, **kwargs)
        self.extern_arc_rate = extern_arc_rate
        self.pbest_factor = pbest_factor
        self.hist_mem_size = hist_mem_size

    def set_parameters(self, population_size=540, extern_arc_rate=2.6, pbest_factor=0.11, hist_mem_size=6, **kwargs):
        r"""Set the parameters of an algorithm.

        Args:
            population_size (Optional[int]): Population size.
            extern_arc_rate (Optional[float]): External archive size factor.
            pbest_factor (Optional[float]): Greediness factor for current-to-pbest/1 mutation.
            hist_mem_size (Optional[int]): Size of historical memory.

        See Also:
            * :func:`niapy.algorithms.basic.DifferentialEvolution.set_parameters`

        """
        super().set_parameters(population_size=population_size,
                               individual_type=kwargs.pop('individual_type', SolutionSHADE), **kwargs)
        self.extern_arc_rate = extern_arc_rate
        self.pbest_factor = pbest_factor
        self.hist_mem_size = hist_mem_size

    def get_parameters(self):
        r"""Get algorithm parameters.

        Returns:
            Dict[str, Any]: Algorithm parameters.

        """
        d = DifferentialEvolution.get_parameters(self)
        d.update({
            'extern_arc_rate': self.extern_arc_rate,
            'pbest_factor': self.pbest_factor,
            'hist_mem_size': self.hist_mem_size,
        })
        return d

    def cauchy(self, loc, gamma):
        r"""Get cauchy random distribution with mean "loc" and standard deviation "gamma".

        Args:
            loc (float): Mean of the cauchy random distribution.
            gamma (float): Standard deviation of the cauchy random distribution.

        Returns:
            Union[numpy.ndarray[float], float]: Array of numbers.

        """
        c = loc + gamma * np.tan(np.pi * (self.random() - 0.5))
        return c if c > 0 else self.cauchy(loc, gamma)

    def gen_ind_params(self, x, hist_cr, hist_f):
        r"""Generate new individual with new scale factor and crossover probability.

        Args:
            x (IndividualSHADE): Individual to apply function on.
            hist_cr (numpy.ndarray[float]): Historic values of crossover probability.
            hist_f (numpy.ndarray[float]): Historic values of scale factor.

        Returns:
            Individual: New individual with new parameters

        """
        mi = self.integers(self.hist_mem_size)  # a random pair of f cr is selected form historical memory
        m_cr = hist_cr[mi]
        m_f = hist_f[mi]
        cr = self.normal(m_cr, 0.1) if m_cr != -1 else 0
        # cr is randomised from normal distribution and then repaired if needed
        cr = np.clip(cr, 0, 1)
        f = self.cauchy(m_f, 0.1)
        # f is randomised from cauchy distribution until the value is >0 and then repaired if needed
        f = np.clip(f, 0, 1)
        return self.individual_type(x=x.x, differential_weight=f, crossover_probability=cr, e=False)

    def evolve(self, pop, hist_cr, hist_f, archive, arc_ind_cnt, task, **_kwargs):
        r"""Evolve current population.

        Args:
            pop (numpy.ndarray[IndividualSHADE]): Current population.
            hist_cr (numpy.ndarray[float]): Historic values of crossover probability.
            hist_f (numpy.ndarray[float]): Historic values of scale factor.
            archive (numpy.ndarray): External archive.
            arc_ind_cnt (int): Number of individuals in the archive.
            task (Task): Optimization task.

        Returns:
            numpy.ndarray: New population.

        """
        new_pop = objects_to_array([self.gen_ind_params(xi, hist_cr, hist_f) for xi in pop])
        p_num = np.int_(np.around(len(pop) * self.pbest_factor))
        if p_num < 2:
            p_num = 2
        # cr and f for mutation are computed
        for i, xi in enumerate(new_pop):
            new_pop[i].x = cross_curr2pbest1(pop, i, xi.differential_weight, xi.crossover_probability, self.rng,
                                             p_num, archive, arc_ind_cnt, task)  # trial vectors are created
        for xi in new_pop:
            xi.evaluate(task, rng=self.random)

        return new_pop

    def selection(self, pop, new_pop, archive, arc_ind_cnt, best_x, best_fitness, task, **kwargs):
        r"""Operator for selection.

        Args:
            pop (numpy.ndarray): Current population.
            new_pop (numpy.ndarray): New Population.
            archive (numpy.ndarray): External archive.
            arc_ind_cnt (int): Number of individuals in the archive.
            best_x (numpy.ndarray): Current global best solution.
            best_fitness (float): Current global best solutions fitness/objective value.
            task (Task): Optimization task.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, int, numpy.ndarray, float]:
                1. New selected individuals.
                2. Scale factor values of successful new individuals.
                3. Crossover probability values of successful new individuals.
                4. Updated external archive.
                5. Updated number of individuals in the archive.
                6. New global best solution.
                7. New global best solutions fitness/objective value.

        """
        success_f = np.asarray([])  # array for storing successful f values
        success_cr = np.asarray([])  # array for storing successful cr values
        fitness_diff = np.asarray([])  # array for storing the difference of fitness of new individuals
        archive_size = np.int_(np.around(len(pop) * self.extern_arc_rate))
        arr = np.copy(pop)
        for i, vi in enumerate(new_pop):
            if vi.f == pop[i].f:
                arr[i] = vi
            elif vi.f < pop[i].f:
                # trial vectors that have a better or equal fitness value are selected for the next generation
                if archive_size > 1:
                    if arc_ind_cnt < archive_size:
                        archive[arc_ind_cnt] = pop[i].x
                        # parents that have worse fitness then their trial vector are stored into the external archive
                        arc_ind_cnt += 1
                    else:
                        rand_ind = self.integers(archive_size)
                        # if the archive is full random archive members are replaced
                        archive[rand_ind] = pop[i].x
                fitness_diff = np.append(fitness_diff, np.absolute(pop[i].f - vi.f))
                success_f = np.append(success_f, vi.differential_weight)
                success_cr = np.append(success_cr, vi.crossover_probability)
                arr[i] = vi
        best_x, best_fitness = self.get_best(arr, np.asarray([ui.f for ui in arr]), best_x, best_fitness)
        return arr, success_f, success_cr, fitness_diff, archive, arc_ind_cnt, best_x, best_fitness

    def post_selection(self, pop, arc, arc_ind_cnt, task, xb, fxb, **kwargs):
        r"""Post selection operator.

        Args:
            pop (numpy.ndarray): Current population.
            arc (numpy.ndarray): External archive.
            arc_ind_cnt (int): Number of individuals in the archive.
            task (Task): Optimization task.
            xb (numpy.ndarray): Global best solution.
            fxb (float): Global best fitness.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray, int, numpy.ndarray, float]:
                1. Changed current population.
                2. Updated external archive.
                3. Updated number of individuals in the archive.
                4. New global best solution.
                5. New global best solutions fitness/objective value.

        """
        return pop, arc, arc_ind_cnt, xb, fxb

    def init_population(self, task):
        r"""Initialize starting population of optimization algorithm.

        Args:
            task (Task): Optimization task.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray, Dict[str, Any]]:
                1. New population.
                2. New population fitness values.
                3. Additional arguments:
                    * h_mem_cr (numpy.ndarray[float]): Historical values of crossover probability.
                    * h_mem_f (numpy.ndarray[float]): Historical values of scale factor.
                    * k (int): Historical memory current index.
                    * archive (numpy.ndarray): External archive.
                    * arc_ind_cnt (int): Number of individuals in the archive.

        See Also:
            * :func:`niapy.algorithms.Algorithm.init_population`

        """
        pop, fitness, _ = DifferentialEvolution.init_population(self, task)  # pop vectors are initialized randomly
        h_mem_cr = np.full(self.hist_mem_size, 0.5)
        h_mem_f = np.full(self.hist_mem_size, 0.5)
        # all values in the historical memory for parameters f and cr are initialized to 0.5
        k = 0  # the starting memory position is set to 1
        arc_size = np.int_(np.around(self.population_size * self.extern_arc_rate))
        archive = np.zeros((arc_size, task.dimension))
        # the external archive of max size pop_size * arc_rate is initialized
        arc_ind_cnt = 0  # the number of archive elements is set to 0

        return pop, fitness, {'h_mem_cr': h_mem_cr, 'h_mem_f': h_mem_f, 'k': k,
                              'archive': archive, 'arc_ind_cnt': arc_ind_cnt}

    def run_iteration(self, task, population, population_fitness, best_x, best_fitness, **params):
        r"""Core function of Success-history based adaptive differential evolution algorithm.

        Args:
            task (Task): Optimization task.
            population (numpy.ndarray): Current population.
            population_fitness (numpy.ndarray[float]): Current population function/fitness values.
            best_x (numpy.ndarray): Global best individual.
            best_fitness (float): Global best individual fitness/function value.
            **params (Dict[str, Any]): Additional arguments.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray[float], Dict[str, Any]]:
                1. New population.
                2. New population fitness/function values.
                3. Additional arguments:
                    * h_mem_cr (numpy.ndarray[float]): Historical values of crossover probability.
                    * h_mem_f (numpy.ndarray[float]): Historical values of scale factor.
                    * k (int): Historical memory current index.
                    * archive (numpy.ndarray): External archive.
                    * arc_ind_cnt (int): Number of individuals in the archive.

        """
        h_mem_cr = params.pop('h_mem_cr')
        h_mem_f = params.pop('h_mem_f')
        k = params.pop('k')
        archive = params.pop('archive')
        arc_ind_cnt = params.pop('arc_ind_cnt')

        indexes = np.argsort(population_fitness)
        sorted_pop = population[indexes]  # sorted population

        new_population = self.evolve(sorted_pop, h_mem_cr, h_mem_f, archive, arc_ind_cnt, task)
        # mutant vectors are created
        population, s_f, s_cr, fit_diff, archive, arc_ind_cnt, best_x, best_fitness = self.selection(
            sorted_pop, new_population, archive, arc_ind_cnt, best_x, best_fitness, task=task)
        # best individuals are selected for the next population

        num_of_success_params = len(s_f)
        if num_of_success_params > 0:
            # if children better than their parents were created the historical memory is updated
            m_sf_k = 0
            m_cr_k = 0
            sum_sf = 0
            sum_cr = 0
            diff_sum = np.sum(fit_diff)
            for i in range(num_of_success_params):
                weight = fit_diff[i] / diff_sum

                m_sf_k += weight * s_f[i] * s_f[i]
                sum_sf += weight * s_f[i]

                m_cr_k += weight * s_cr[i] * s_cr[i]
                sum_cr += weight * s_cr[i]

            h_mem_f[k] = m_sf_k / sum_sf
            # f and cr that are stored into the historic memory are calculated with the use of weighted Lehmer mean
            h_mem_cr[k] = -1 if sum_cr == 0 or h_mem_cr[k] == -1 else m_cr_k / sum_cr
            # the value of cr is updated if the previous value isn't terminal and max(s_cr)!=0

            k = 0 if k + 1 >= self.hist_mem_size else k + 1
            # historical memory position is increased, if over the size of the memory its set back to 1

        population, archive, arc_ind_cnt, best_x, best_fitness = self.post_selection(population, archive, arc_ind_cnt,
                                                                                     task, best_x, best_fitness)
        population_fitness = np.asarray([x.f for x in population])
        best_x, best_fitness = self.get_best(population, population_fitness, best_x, best_fitness)

        return population, population_fitness, best_x, best_fitness, {'h_mem_cr': h_mem_cr, 'h_mem_f': h_mem_f, 'k': k,
                                                                      'archive': archive, 'arc_ind_cnt': arc_ind_cnt}


class LpsrSuccessHistoryAdaptiveDifferentialEvolution(SuccessHistoryAdaptiveDifferentialEvolution):
    r"""Implementation of Success-history based adaptive differential evolution algorithm with Linear population size reduction.

    Algorithm:
        Success-history based adaptive differential evolution algorithm with Linear population size reduction

    Date:
        2022

    Author:
        Aleš Gartner

    License:
        MIT

    Reference paper:
        Ryoji Tanabe and Alex Fukunaga: Improving the Search Performance of SHADE Using Linear Population Size Reduction,  Proc. IEEE Congress on Evolutionary Computation (CEC-2014), Beijing, July, 2014.

    Attributes:
        Name (List[str]): List of strings representing algorithm name

    See Also:
        * :class:`niapy.algorithms.modified.SuccessHistoryAdaptiveDifferentialEvolution`

    """

    Name = ['LpsrSuccessHistoryAdaptiveDifferentialEvolution', 'L-SHADE']

    def post_selection(self, pop, arc, arc_ind_cnt, task, xb, fxb, **kwargs):
        r"""Post selection operator.

        In this algorithm the post selection operator linearly reduces the population size. The size of external archive is also updated.

        Args:
            pop (numpy.ndarray): Current population.
            arc (numpy.ndarray): External archive.
            arc_ind_cnt (int): Number of individuals in the archive.
            task (Task): Optimization task.
            xb (numpy.ndarray): Global best solution.
            fxb (float): Global best fitness.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray, int, numpy.ndarray, float]:
                1. Changed current population.
                2. Updated external archive.
                3. Updated number of individuals in the archive.
                4. New global best solution.
                5. New global best solutions fitness/objective value.

        """
        pop_size = len(pop)
        max_nfe = task.max_evals
        nfe = task.evals

        next_pop_size = np.int_(np.around((((4.0 - self.population_size) / np.float_(max_nfe)) * nfe) + self.population_size))

        if next_pop_size < 4:
            next_pop_size = 4
        # the size of the next population is calculated
        # if the size of the new population is smaller than the current,
        # the worst pop_size - new_pop_size individuals are deleted
        if next_pop_size < pop_size:
            reduction = pop_size - next_pop_size
            for i in range(reduction):
                worst = 0
                for j, e in enumerate(pop):
                    worst = j if e.f > pop[worst].f else worst
                pop = np.delete(pop, worst)

            next_arc_size = np.int_(next_pop_size * self.extern_arc_rate)  # the size of the new archive
            if arc_ind_cnt > next_arc_size:
                arc_ind_cnt = next_arc_size

        return pop, arc, arc_ind_cnt, xb, fxb
