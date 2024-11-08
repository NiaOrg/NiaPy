# encoding=utf8
import logging
import numpy as np

from niapy.algorithms.algorithm import Individual
from niapy.algorithms.basic.de import DifferentialEvolution
from niapy.algorithms.modified.shade import SuccessHistoryAdaptiveDifferentialEvolution
from niapy.algorithms.modified.shade import cross_curr2pbest1
from niapy.util import objects_to_array

logging.basicConfig()
logger = logging.getLogger('niapy.algorithms.modified')
logger.setLevel('INFO')

__all__ = [
    'ImprovedLpsrSuccessHistoryAdaptiveDifferentialEvolution',
]

class SolutionILSHADE(Individual):
    r"""Individual for iL-SHADE algorithm.

    Attributes:
        differential_weight (float): Scale factor.
        crossover_probability (float): Crossover probability.

    See Also:
        :class:`niapy.algorithms.Individual`

    """

    def __init__(self, differential_weight=0.5, crossover_probability=0.8, **kwargs):
        r"""Initialize SolutionILSHADE.

        Attributes:
            differential_weight (float): Scale factor.
            crossover_probability (float): Crossover probability.

        See Also:
            :func:`niapy.algorithm.Individual.__init__`

        """
        super().__init__(**kwargs)
        self.differential_weight = differential_weight
        self.crossover_probability = crossover_probability


class ImprovedLpsrSuccessHistoryAdaptiveDifferentialEvolution(SuccessHistoryAdaptiveDifferentialEvolution):
    r"""Implementation of Improved Success-history based adaptive differential evolution algorithm with Linear population size reduction.

    Algorithm:
        Improved Success-history based adaptive differential evolution algorithm with Linear population size reduction

    Date:
        2024

    Author:
        Grega Rubin

    License:
        MIT

    Reference paper:
        Janez Brest, Mirjam Sepesy Maućec, Borko Bošković: iL-SHADE: Improved L-SHADE Algorithm for single Objective Real-Parameter Optimization,  Proc. IEEE Congress on Evolutionary Computation (CEC-2016), Vancouver, July, 2016

    Attributes:
        Name (List[str]): List of strings representing algorithm name
        extern_arc_rate (float): External archive size factor.
        pbest_factor (float): Greediness factor for current-to-pbest/1 mutation.
        hist_mem_size (int): Size of historical memory.

    See Also:
        * :class:`niapy.algorithms.basic.DifferentialEvolution`

    """
    
    Name = ['ImprovedLsprSuccessHistoryAdaptiveDifferentialEvolution', 'ILSHADE']

    @staticmethod
    def info():
        r"""Get algorithm information.

        Returns:
            str: Algorithm information.

        See Also:
            * :func:`niapy.algorithms.Algorithm.info`

        """
        return r""""""

    def __init__(self, population_size=360, extern_arc_rate=2.6, pbest_start=0.2, pbest_end=0.1, hist_mem_size=6, *args, **kwargs):
        """Initialize iL-SHADE.

        Args:
            population_size (Optional[int]): Population size.
            extern_arc_rate (Optional[float]): External archive size factor.
            pbest_start (Optional[float]): Starting value for greediness factor for current-to-pbest/1 mutation.
            pbest_end (Optional[float]): End value for greediness factor for current-to-pbest/1 mutation.
            hist_mem_size (Optional[int]): Size of historical memory.

        See Also:
            * :func:`niapy.algorithms.basic.DifferentialEvolution.__init__`

        """
        super(SuccessHistoryAdaptiveDifferentialEvolution, self).__init__(population_size, individual_type=kwargs.pop('individual_type', SolutionILSHADE), *args, **kwargs)
        self.extern_arc_rate = extern_arc_rate
        self.pbest_start = pbest_start
        self.pbest_end = pbest_end
        self.hist_mem_size = hist_mem_size
    
    def set_parameters(self, population_size=360, extern_arc_rate=2.6, pbest_start=0.2, pbest_end=0.1, hist_mem_size=6, **kwargs):
        r"""Set the parameters of an algorithm.

        Args:
            population_size (Optional[int]): Population size.
            extern_arc_rate (Optional[float]): External archive size factor.
            pbest_start (Optional[float]): Start value for greediness factor for current-to-pbest/1 mutation.
            pbest_end (Optional[float]): End value for greediness factor for current-to-pbest/1 mutation.
            hist_mem_size (Optional[int]): Size of historical memory.

        See Also:
            * :func:`niapy.algorithms.basic.DifferentialEvolution.set_parameters`

        """
        super(SuccessHistoryAdaptiveDifferentialEvolution, self).set_parameters(population_size=population_size,
                               individual_type=kwargs.pop('individual_type', SolutionILSHADE), **kwargs)
        self.extern_arc_rate = extern_arc_rate
        self.pbest_start = pbest_start
        self.pbest_end = pbest_end
        self.hist_mem_size = hist_mem_size

    def get_parameters(self):
        r"""Get algorithm parameters.

        Returns:
            Dict[str, Any]: Algorithm parameters.

        """
        d = super(SuccessHistoryAdaptiveDifferentialEvolution, self).get_parameters()
        d.update({
            'extern_arc_rate': self.extern_arc_rate,
            'pbest_start': self.pbest_start,
            'pbest_end': self.pbest_end,
            'hist_mem_size': self.hist_mem_size,
        })
        return d
    
    def gen_ind_params(self, nfe, max_nfe, x, hist_cr, hist_f):
        r"""Generate new individual with new scale factor and crossover probability.

        Args:
            nfe (int): Current number of fitness function calls.
            max_nfe (int): Maximum number of fitness function calls.
            x (IndividualSHADE): Individual to apply function on.
            hist_cr (numpy.ndarray[float]): Historic values of crossover probability.
            hist_f (numpy.ndarray[float]): Historic values of scale factor.

        Returns:
            Individual: New individual with new parameters

        """
        mi = self.integers(self.hist_mem_size)  # a random pair of f cr is selected form historical memory
        #ilshade
        if(mi == self.hist_mem_size - 1):
            m_cr = 0.9
            m_f = 0.9
        else:
            m_cr = hist_cr[mi]
            m_f = hist_f[mi]
        
        #ilshade
        cr = self.normal(m_cr, 0.1) if m_cr >= 0 else 0 

        cr = np.clip(cr, 0, 1)
        #ilshade
        if((nfe < 0.25*max_nfe) and cr < 0.5): cr = 0.5
        elif((nfe < 0.5*max_nfe) and cr < 0.25): cr = 0.25
        

        f = self.cauchy(m_f, 0.1)
        f = np.clip(f, 0, 1)
        #ilshade
        if((nfe < 0.25*max_nfe) and f > 0.7): f = 0.7
        elif((nfe < 0.5*max_nfe) and f > 0.8): f = 0.8
        elif((nfe < 0.75*max_nfe) and f > 0.9): f = 0.9
        
        return self.individual_type(x=x.x, differential_weight=f, crossover_probability=cr, e=False)

    def evolve(self, pop, hist_cr, hist_f, archive, arc_ind_cnt, pbest_factor, task, **_kwargs):
        r"""Evolve current population.

        Args:
            pop (numpy.ndarray[IndividualSHADE]): Current population.
            hist_cr (numpy.ndarray[float]): Historic values of crossover probability.
            hist_f (numpy.ndarray[float]): Historic values of scale factor.
            archive (numpy.ndarray): External archive.
            arc_ind_cnt (int): Number of individuals in the archive.
            pbest_factor (float): Greediness factor for current-to-pbest/1 mutation
            task (Task): Optimization task.

        Returns:
            numpy.ndarray: New population.

        """
        max_nfe = task.max_evals
        nfe = task.evals
        new_pop = objects_to_array([self.gen_ind_params(nfe, max_nfe, xi, hist_cr, hist_f) for xi in pop])
        p_num = np.int_(np.around(len(pop) * pbest_factor))
        if p_num < 2:
            p_num = 2
        # cr and f for mutation are computed
        for i, xi in enumerate(new_pop):
            new_pop[i].x = cross_curr2pbest1(pop, i, xi.differential_weight, xi.crossover_probability, self.rng,
                                             p_num, archive, arc_ind_cnt, task)  # trial vectors are created
        for xi in new_pop:
            xi.evaluate(task, rng=self.random)

        return new_pop

    def post_selection(self, pop, arc, arc_ind_cnt, task, xb, fxb, pbest_factor, **kwargs):
        r"""Post selection operator.

        Args:
            pop (numpy.ndarray): Current population.
            arc (numpy.ndarray): External archive.
            arc_ind_cnt (int): Number of individuals in the archive.
            task (Task): Optimization task.
            xb (numpy.ndarray): Global best solution.
            fxb (float): Global best fitness.
            pbest_factor (float): Greediness factor for current-to-pbest/1 mutation 

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray, int, numpy.ndarray, float, float]:
                1. Changed current population.
                2. Updated external archive.
                3. Updated number of individuals in the archive.
                4. New global best solution.
                5. New global best solutions fitness/objective value.
                6. Updated greediness factor

        """

        pop_size = len(pop)
        max_nfe = task.max_evals
        nfe = task.evals

        next_pop_size = np.int_(np.around((((4.0 - self.population_size) / np.float64(max_nfe)) * nfe) + self.population_size))

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

            #ilshade
            pbest_factor = ((self.pbest_end - self.pbest_start) / max_nfe) * nfe + self.pbest_start

        return pop, arc, arc_ind_cnt, xb, fxb, pbest_factor
    
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
                    * pbest_factor (float): Current greediness factor value

        See Also:
            * :func:`niapy.algorithms.Algorithm.init_population`

        """
        pop, fitness, _ = DifferentialEvolution.init_population(self, task)  # pop vectors are initialized randomly
        h_mem_cr = np.full(self.hist_mem_size, 0.8) #ilshade
        h_mem_f = np.full(self.hist_mem_size, 0.5)
        # all values in the historical memory for parameters f and cr are initialized to 0.5
        k = 0  # the starting memory position is set to 1
        arc_size = np.int_(np.around(self.population_size * self.extern_arc_rate))
        archive = np.zeros((arc_size, task.dimension))
        # the external archive of max size pop_size * arc_rate is initialized
        arc_ind_cnt = 0  # the number of archive elements is set to 0
        pbest_factor = self.pbest_start #the greediness factor is set to the starting value

        return pop, fitness, {'h_mem_cr': h_mem_cr, 'h_mem_f': h_mem_f, 'k': k,
                              'archive': archive, 'arc_ind_cnt': arc_ind_cnt, 
                              'pbest_factor': pbest_factor}

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
                    * pbest_factor (float): Current greediness factor value

        """
        h_mem_cr = params.pop('h_mem_cr')
        h_mem_f = params.pop('h_mem_f')
        k = params.pop('k')
        archive = params.pop('archive')
        arc_ind_cnt = params.pop('arc_ind_cnt')
        #ilshade
        pbest_factor = params.pop('pbest_factor')

        indexes = np.argsort(population_fitness)
        sorted_pop = population[indexes]  # sorted population
        #ilshade
        new_population = self.evolve(sorted_pop, h_mem_cr, h_mem_f, archive, arc_ind_cnt, pbest_factor, task)
        # mutant vectors are created
        population, s_f, s_cr, fit_diff, archive, arc_ind_cnt, best_x, best_fitness = self.selection(
            sorted_pop, new_population, archive, arc_ind_cnt, best_x, best_fitness, task=task)
        # best individuals are selected for the next population

        num_of_success_params = len(s_f)
        if num_of_success_params > 0:
            #ilshade
            old_f = h_mem_f[k]
            old_cr = h_mem_cr[k]
            
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

            #ilshade
            h_mem_f[k] = (h_mem_f[k] + old_f) / 2
            h_mem_cr[k] = (h_mem_cr[k] + old_cr) / 2
            
            k = 0 if k + 1 >= self.hist_mem_size else k + 1
            # historical memory position is increased, if over the size of the memory its set back to 1

        population, archive, arc_ind_cnt, best_x, best_fitness, pbest_factor = self.post_selection(population, archive, arc_ind_cnt,
                                                                                     task, best_x, best_fitness, pbest_factor)
        population_fitness = np.asarray([x.f for x in population])
        best_x, best_fitness = self.get_best(population, population_fitness, best_x, best_fitness)

        return population, population_fitness, best_x, best_fitness, {'h_mem_cr': h_mem_cr, 'h_mem_f': h_mem_f, 'k': k,
                                                                      'archive': archive, 'arc_ind_cnt': arc_ind_cnt,
                                                                      'pbest_factor': pbest_factor}
