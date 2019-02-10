# encoding=utf8
# pylint: disable=mixed-indentation, trailing-whitespace, multiple-statements, attribute-defined-outside-init, logging-not-lazy, arguments-differ, line-too-long, unused-argument, no-self-use
import copy
import numpy as np

from NiaPy.algorithms.algorithm import Algorithm, Individual

class Fish(Individual):
    def __init__(self, weight, **kwargs):
        Individual.__init__(self, **kwargs)
        self.weight = weight
        self.delta_pos = np.nan
        self.delta_cost = np.nan
        self.has_improved = False

class FishSchoolSearch(Algorithm):
    r"""Implementation of Fish School Search algorithm.

    **Algorithm:** Fish School Search algorithm

    **Date:** 2019

    **Authors:**
        Clodomir Santana Jr, Elliackin Figueredo, Mariana Maceds,
        Pedro Santos. Ported to NiaPy with small changes by Kristian Järvenpää and Klemen Berkovič

    **License:** MIT

    **Reference paper:**
        Bastos Filho, Lima Neto, Lins, D. O. Nascimento and P. Lima, “A novel search
        algorithm based on fish school behavior,” in 2008 IEEE International
        Conference on Systems, Man and Cybernetics, Oct 2008, pp. 2646–2651.
    """

    Name = ['FSS', 'FishSchoolSearch']

    @staticmethod
    def typeParameters():
        # TODO
        return {
            'school_size': lambda x: False,
            'SI_final': lambda x: False
        }

    def setParameters(self, NP=25, SI_init=3, SI_final=10, SV_init=3, SV_final=13, min_w=0.3, w_scale=0.7, **ukwargs):
        r"""**__init__(self, n_iter, D, school_size, SI_init, SI_final, SV_init, SV_final, min_w, w_scale, benchmark)**.

		  Arguments:
				school_size {integer} -- number of fishes in school

				SI_init {integer} -- length of initial individual step

				SI_final {integer} -- length of final individual step

				SV_init {integer} -- length of initial volatile step

				SV_final {integer} -- length of final volatile step

				min_w {float} -- minimum weight of a fish

				w_scale {float} -- maximum weight of a fish

				benchmark {object} -- benchmark implementation object

		 Raises:
				TypeError -- Raised when given benchmark function which does not exists.
		  """
        self.school_size = NP
        self.step_individual_init = SI_init
        self.step_individual_final = SI_final
        self.step_volitive_init = SV_init
        self.step_volitive_final = SV_final
        self.min_w = min_w
        self.w_scale = w_scale

    def generate_uniform_coordinates(self, task):
        """Return Numpy array with uniform distribution."""
        x = np.zeros((self.school_size, task.D))
        for i in range(self.school_size):
            x[i] = self.uniform(task.Lower[i], task.Upper[i], task.D)
        return x

    def gen_weight(self):
        """Get initial weight for fish."""
        return self.w_scale / 2.0

    def init_fish(self, pos, task):
        """Create a new fish to a given position."""
        return Fish(x=pos, weight=self.gen_weight(), task=task, e=True)

    def init_school(self, task):
        """Initialize fish school with uniform distribution."""
        curr_step_individual = self.step_individual_init * (task.Upper - task.Lower)
        curr_step_volitive = self.step_volitive_init * (task.Upper - task.Lower)
        best_fish = None
        curr_weight_school = 0.0
        prev_weight_school = 0.0
        school = []
        positions = self.generate_uniform_coordinates(task)
        for idx in range(self.school_size):
            fish = self.init_fish(positions[idx], task)
            school.append(fish)
            curr_weight_school += fish.weight
        prev_weight_school = curr_weight_school
        best_fish = self.update_best_fish(school, best_fish)
        return curr_step_individual, curr_step_volitive, best_fish, curr_weight_school, prev_weight_school, school

    def max_delta_cost(self, school):
        """Find maximum delta cost - return 0 if none of the fishes moved."""
        max_ = 0
        for fish in school:
            if max_ < fish.delta_cost: max_ = fish.delta_cost
        return max_

    def total_school_weight(self, school, prev_weight_school, curr_weight_school):
        """Calculate and update current weight of fish school."""
        prev_weight_school = curr_weight_school
        curr_weight_school = sum(fish.weight for fish in school)
        return prev_weight_school, curr_weight_school

    def calculate_barycenter(self, school, task):
        """Calculate barycenter of fish school."""
        barycenter = np.zeros((task.D,), dtype=np.float)
        density = 0.0
        for fish in school:
            density += fish.weight
            for dim in range(task.D): barycenter[dim] += (fish.x[dim] * fish.weight)
        for dim in range(task.D): barycenter[dim] = barycenter[dim] / density
        return barycenter

    def update_steps(self, task):
        """Update step length for individual and volatile steps."""
        curr_step_individual = np.full(task.D, self.step_individual_init - task.Iters * float(self.step_individual_init - self.step_individual_final) / task.nGEN)
        curr_step_volitive = np.full(task.D, self.step_volitive_init - task.Iters * float(self.step_volitive_init - self.step_volitive_final) / task.nGEN)
        return curr_step_individual, curr_step_volitive

    def update_best_fish(self, school, best_fish):
        """Find and update current best fish."""
        if best_fish is None: best_fish = copy.copy(school[0])
        for fish in school:
            if best_fish.f > fish.f: best_fish = copy.copy(fish)
        return best_fish

    def feeding(self, school):
        """Feed all fishes."""
        for fish in school:
            if self.max_delta_cost(school): fish.weight = fish.weight + (fish.delta_cost / self.max_delta_cost(school))
            if fish.weight > self.w_scale: fish.weight = self.w_scale
            elif fish.weight < self.min_w: fish.weight = self.min_w
        return school

    def individual_movement(self, school, curr_step_individual, task):
        """Perform individual movement for each fish."""
        for fish in school:
            new_pos = np.zeros((task.D,), dtype=np.float)
            for dim in range(task.D):
                new_pos[dim] = fish.x[dim] + (curr_step_individual[dim] * self.uniform(-1, 1))
                if new_pos[dim] < task.Lower[dim]: new_pos[dim] = task.Lower[dim]
                elif new_pos[dim] > task.Upper[dim]: new_pos[dim] = task.Upper[dim]
            cost = task.eval(new_pos)
            if cost < fish.f:
                fish.delta_cost = abs(cost - fish.cost)
                fish.cost = cost
                delta_pos = np.zeros((task.D,), dtype=np.float)
                for idx in range(task.D): delta_pos[idx] = new_pos[idx] - fish.x[idx]
                fish.delta_pos = delta_pos
                fish.x = new_pos
            else:
                fish.delta_pos = np.zeros((task.D,), dtype=np.float)
                fish.delta_cost = 0
        return school

    def collective_instinctive_movement(self, school, task):
        """Perform collective instictive movement."""
        cost_eval_enhanced = np.zeros((task.D,), dtype=np.float)
        density = 0.0
        for fish in school:
            density += fish.delta_cost
            for dim in range(task.D): cost_eval_enhanced[dim] += (fish.delta_pos[dim] * fish.delta_cost)
        for dim in range(task.D):
            if density != 0: cost_eval_enhanced[dim] = cost_eval_enhanced[dim] / density
        for fish in school:
            new_pos = np.zeros((task.D,), dtype=np.float)
            for dim in range(task.D):
                new_pos[dim] = fish.x[dim] + cost_eval_enhanced[dim]
                if new_pos[dim] < task.Lower[dim]: new_pos[dim] = task.Lower[dim]
                elif new_pos[dim] > task.Upper[dim]: new_pos[dim] = task.Upper[dim]
            fish.x = new_pos
        return school

    def collective_volitive_movement(self, school, curr_step_volitive, prev_weight_school, curr_weight_school, task):
        """Perform collective volitive movement."""
        prev_weight_school, curr_weight_school = self.total_school_weight(school=school, prev_weight_school=prev_weight_school, curr_weight_school=curr_weight_school)
        barycenter = self.calculate_barycenter(school, task)
        for fish in school:
            new_pos = np.zeros((task.D,), dtype=np.float)
            for dim in range(task.D):
                if curr_weight_school > prev_weight_school:
                    new_pos[dim] = fish.x[dim] - ((fish.x[dim] - barycenter[dim]) * curr_step_volitive * self.uniform(0, 1))
                else:
                    new_pos[dim] = fish.x[dim] + ((fish.x[dim] - barycenter[dim]) * curr_step_volitive[dim] * self.uniform(0, 1))
                if new_pos[dim] < task.Lower[dim]:
                    new_pos[dim] = task.Lower[dim]
                elif new_pos[dim] > task.Upper[dim]:
                    new_pos[dim] = task.Upper[dim]
            cost = task.eval(new_pos)
            fish.cost = cost
            fish.x = new_pos
        return school

    def runTask(self, task):
        """Run the algorithm."""
        curr_step_individual, curr_step_volitive, best_fish, curr_weight_school, prev_weight_school, school = self.init_school(task)
        while not task.stopCondI():
            school = self.individual_movement(school, curr_step_individual, task)
            best_fish = self.update_best_fish(school, best_fish)
            school = self.feeding(school)
            school = self.collective_instinctive_movement(school, task)
            school = self.collective_volitive_movement(school=school, curr_step_volitive=curr_step_volitive, prev_weight_school=prev_weight_school, curr_weight_school=curr_weight_school, task=task)
            curr_step_individual, curr_step_volitive = self.update_steps(task)
            best_fish = self.update_best_fish(school, best_fish)
        return (best_fish.x, best_fish.cost)
