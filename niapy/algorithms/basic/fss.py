# encoding=utf8
import numpy as np

from niapy.algorithms.algorithm import Algorithm, Individual
from niapy.util import objects_to_array


class Fish(Individual):
    r"""Fish individual class.

    Attributes:
        weight (float): Weight of fish.
        delta_pos (float): Displacement due to individual movement.
        delta_cost (float): Cost at `delta_apos`.
        has_improved (bool): True if the fish has improved.

    See Also:
        * :class:`niapy.algorithms.algorithm.Individual`

    """

    def __init__(self, weight, **kwargs):
        r"""Initialize fish individual.

        Args:
            weight (float): Weight of fish.

        See Also:
            * :func:`niapy.algorithms.algorithm.Individual`

        """
        super().__init__(**kwargs)
        self.weight = weight
        self.delta_pos = np.nan
        self.delta_cost = np.nan
        self.has_improved = False


class FishSchoolSearch(Algorithm):
    r"""Implementation of Fish School Search algorithm.

    Algorithm:
        Fish School Search algorithm

    Date:
        2019

    Authors:
        Clodomir Santana Jr, Elliackin Figueredo, Mariana Maceds, Pedro Santos.
        Ported to niapy with small changes by Kristian Järvenpää (2018).
        Ported to niapy 2.0 by Klemen Berkovič (2019).

    License:
        MIT

    Reference paper:
        Bastos Filho, Lima Neto, Lins, D. O. Nascimento and P. Lima,
        “A novel search algorithm based on fish school behavior,”
        in 2008 IEEE International Conference on Systems, Man and Cybernetics, Oct 2008, pp. 2646–2651.

    Attributes:
        Name (List[str]): List of strings representing algorithm name.
        step_individual_init (float): Length of initial individual step.
        step_individual_final (float): Length of final individual step.
        step_volitive_init (float): Length of initial volatile step.
        step_volitive_final (float): Length of final volatile step.
        min_w (float): Minimum weight of a fish.
        w_scale (float): Maximum weight of a fish.

    See Also:
        * :class:`niapy.algorithms.algorithm.Algorithm`

    """

    Name = ['FSS', 'FishSchoolSearch']

    @staticmethod
    def info():
        r"""Get default information of algorithm.

        Returns:
            str: Basic information.

        See Also:
            * :func:`niapy.algorithms.Algorithm.info`

        """
        return r"""Bastos Filho, Lima Neto, Lins, D. O. Nascimento and P. Lima,
        “A novel search algorithm based on fish school behavior,”
        in 2008 IEEE International Conference on Systems, Man and Cybernetics, Oct 2008, pp. 2646–2651."""

    def __init__(self, population_size=30, step_individual_init=0.1, step_individual_final=0.0001,
                 step_volitive_init=0.01, step_volitive_final=0.001, min_w=1.0, w_scale=500.0, *args, **kwargs):
        """Initialize FishSchoolSearch.

        Args:
            population_size (Optional[int]): Number of fishes in school.
            step_individual_init (Optional[float]): Length of initial individual step.
            step_individual_final (Optional[float]): Length of final individual step.
            step_volitive_init (Optional[float]): Length of initial volatile step.
            step_volitive_final (Optional[float]): Length of final volatile step.
            min_w (Optional[float]): Minimum weight of a fish.
            w_scale (Optional[float]): Maximum weight of a fish. Recommended value: max_iterations / 2

        See Also:
            * :func:`niapy.algorithms.Algorithm.__init__`

        """
        super().__init__(population_size, *args, **kwargs)
        self.step_individual_init = step_individual_init
        self.step_individual_final = step_individual_final
        self.step_volitive_init = step_volitive_init
        self.step_volitive_final = step_volitive_final
        self.min_w = min_w
        self.w_scale = w_scale

    def set_parameters(self, population_size=30, step_individual_init=0.1, step_individual_final=0.0001,
                       step_volitive_init=0.01, step_volitive_final=0.001, min_w=1.0, w_scale=5000.0, **kwargs):
        r"""Set core arguments of FishSchoolSearch algorithm.

        Args:
            population_size (Optional[int]): Number of fishes in school.
            step_individual_init (Optional[float]): Length of initial individual step.
            step_individual_final (Optional[float]): Length of final individual step.
            step_volitive_init (Optional[float]): Length of initial volatile step.
            step_volitive_final (Optional[float]): Length of final volatile step.
            min_w (Optional[float]): Minimum weight of a fish.
            w_scale (Optional[float]): Maximum weight of a fish. Recommended value: max_iterations / 2

        See Also:
            * :func:`niapy.algorithms.Algorithm.set_parameters`

        """
        super().set_parameters(population_size=population_size, **kwargs)
        self.step_individual_init = step_individual_init
        self.step_individual_final = step_individual_final
        self.step_volitive_init = step_volitive_init
        self.step_volitive_final = step_volitive_final
        self.min_w = min_w
        self.w_scale = w_scale

    def get_parameters(self):
        r"""Get algorithm parameters.

        Returns:
            Dict[str, Any]: Algorithm parameters.

        See Also:
            * :func:`niapy.algorithms.Algorithm.set_parameters`

        """
        d = super().get_parameters()
        d.update({
            'step_individual_init': self.step_individual_init,
            'step_individual_final': self.step_individual_final,
            'step_volitive_init': self.step_volitive_init,
            'step_volitive_final': self.step_volitive_final,
            'min_w': self.min_w,
            'w_scale': self.w_scale
        })
        return d

    def init_school(self, task):
        """Initialize fish school with uniform distribution."""
        step_individual = self.step_individual_init * task.range
        step_volitive = self.step_volitive_init * task.range
        school = [Fish(weight=self.w_scale / 2.0, task=task, e=True, rng=self.rng) for _ in range(self.population_size)]
        school_weight = self.population_size * self.w_scale / 2.0
        return step_individual, step_volitive, school_weight, objects_to_array(school)

    def update_steps(self, task):
        r"""Update step length for individual and volatile steps.

        Args:
            task (Task): Optimization task

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray]:
                1. New individual step.
                2. New volitive step.

        """
        step_individual = np.full(task.dimension, self.step_individual_init - (task.iters + 1) * (
                    self.step_individual_init - self.step_individual_final) / task.max_iters)
        step_volitive = np.full(task.dimension, self.step_volitive_init - (task.iters + 1) * (
                    self.step_volitive_init - self.step_volitive_final) / task.max_iters)
        return step_individual, step_volitive

    def feeding(self, school):
        r"""Feed all fishes.

        Args:
            school (numpy.ndarray): Current school fish population.

        Returns:
            numpy.ndarray: New school fish population.

        """
        max_delta_cost = max(fish.delta_cost for fish in school)
        for fish in school:
            if max_delta_cost:
                fish.weight = fish.weight + (fish.delta_cost / max_delta_cost)
            fish.weight = np.clip(fish.weight, self.min_w, self.w_scale)
        return school

    def individual_movement(self, school, step_individual, xb, fxb, task):
        r"""Perform individual movement for each fish.

        Args:
            school (numpy.ndarray): School fish population.
            step_individual (numpy.ndarray): Current individual step.
            xb (numpy.ndarray): Global best solution.
            fxb (float): Global best solutions fitness/objective value.
            task (Task): Optimization task.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray, float]:
                1. New school of fishes.
                2. New global best position.
                3. New global best fitness.

        """
        for fish in school:
            new_pos = task.repair(fish.x + (step_individual * self.uniform(-1, 1, task.dimension)), rng=self.rng)
            cost = task.eval(new_pos)
            if cost < fish.f:
                xb, fxb = self.get_best(new_pos, cost, xb, fxb)
                fish.delta_cost = abs(cost - fish.f)
                fish.f = cost
                fish.delta_pos = new_pos - fish.x
                fish.x = new_pos
            else:
                fish.delta_pos = np.zeros(task.dimension)
                fish.delta_cost = 0
        return school, xb, fxb

    def collective_instinctive_movement(self, school, task):
        r"""Perform collective instinctive movement.

        Args:
            school (numpy.ndarray): Current population.
            task (Task): Optimization task.

        Returns:
            numpy.ndarray: New population

        """
        cost_eval_enhanced = sum((fish.delta_cost * fish.delta_pos for fish in school), start=np.zeros(task.dimension))
        density = sum(f.delta_cost for f in school)
        if density != 0:
            cost_eval_enhanced /= density
        for fish in school:
            fish.x = task.repair(fish.x + cost_eval_enhanced, rng=self.rng)
        return school

    def collective_volitive_movement(self, school, step_volitive, school_weight, xb, fxb, task):
        r"""Perform collective volitive movement.

        Args:
            school (numpy.ndarray):
            step_volitive :
            school_weight:
            xb (numpy.ndarray): Global best solution.
            fxb (float): Global best solutions fitness/objective value.
            task (Task): Optimization task.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray, float]:
                1. New population.
                2. New global best individual.
                3. New global best fitness.

        """
        prev_weight_school = school_weight
        school_weight = sum(fish.weight for fish in school)

        barycenter = sum((fish.x * fish.weight for fish in school), start=np.zeros(task.dimension))
        barycenter /= sum(fish.weight for fish in school)
        for fish in school:
            if school_weight > prev_weight_school:
                fish.x -= (fish.x - barycenter) * step_volitive * self.uniform(0, 1, task.dimension)
            else:
                fish.x += (fish.x - barycenter) * step_volitive * self.uniform(0, 1, task.dimension)
            fish.evaluate(task, rng=self.rng)
            xb, fxb = self.get_best(fish.x, fish.f, xb, fxb)
        return school, xb, fxb

    def init_population(self, task):
        r"""Initialize the school.

        Args:
            task (Task): Optimization task.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray, dict]:
                1. Population.
                2. Population fitness.
                3. Additional arguments:
                    * step_individual (float): Current individual step.
                    * step_volitive (float): Current volitive step.
                    * school_weight (float): Current school weight.

        """
        step_individual, step_volitive, school_weight, school = self.init_school(task)
        return school, np.asarray([f.f for f in school]), {'step_individual': step_individual,
                                                           'step_volitive': step_volitive,
                                                           'school_weight': school_weight}

    def run_iteration(self, task, population, population_fitness, best_x, best_fitness, **params):
        r"""Core function of algorithm.

        Args:
            task (Task): Optimization task.
            population (numpy.ndarray): Current population.
            population_fitness (numpy.ndarray): Current population fitness.
            best_x (numpy.ndarray): Current global best individual.
            best_fitness (float): Current global best fitness.
            **params: Additional parameters.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, float, dict]:
                1. New Population.
                2. New Population fitness.
                3. New global best individual.
                4. New global best fitness.
                5. Additional parameters:
                    * step_individual (float): Current individual step.
                    * step_volitive (float): Current volitive step.
                    * school_weight (float): Current school weight.

        """
        step_individual = params.pop('step_individual')
        step_volitive = params.pop('step_volitive')
        school_weight = params.pop('school_weight')

        population, best_x, best_fitness = self.individual_movement(population, step_individual, best_x, best_fitness, task)
        population = self.feeding(population)
        population = self.collective_instinctive_movement(population, task)
        population, best_x, best_fitness = self.collective_volitive_movement(population, step_volitive, school_weight,
                                                                             best_x, best_fitness, task)
        step_individual, step_volitive = self.update_steps(task)
        return population, np.asarray([f.f for f in population]), best_x, best_fitness, {'step_individual': step_individual,
                                                                                         'step_volitive': step_volitive,
                                                                                         'school_weight': school_weight
                                                                                         }

# vim: tabstop=3 noexpandtab shiftwidth=3 softtabstop=3
