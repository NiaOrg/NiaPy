# encoding=utf8
# pylint: disable=mixed-indentation, trailing-whitespace, multiple-statements, attribute-defined-outside-init, logging-not-lazy, arguments-differ, line-too-long, unused-argument
import copy
import numpy as np

class Fish(object):
    def __init__(self, D):
        nan = float('nan')
        self.pos = [nan for _ in range(D)]
        self.delta_pos = np.nan
        self.delta_cost = np.nan
        self.weight = np.nan
        self.cost = np.nan
        self.has_improved = False

class FishSchoolSearch(object):
    r"""Implementation of Fish School Search algorithm.

    **Algorithm:** Fish School Search algorithm

    **Date:** 2019

    **Authors:**
        Clodomir Santana Jr, Elliackin Figueredo, Mariana Maceds,
        Pedro Santos. Ported to NiaPy with small changes by Kristian Järvenpää

    **License:** MIT

    **Reference paper:**
        Bastos Filho, Lima Neto, Lins, D. O. Nascimento and P. Lima, “A novel search
        algorithm based on fish school behavior,” in 2008 IEEE International
        Conference on Systems, Man and Cybernetics, Oct 2008, pp. 2646–2651.
    """

    def __init__(self, n_iter, D, school_size, SI_init, SI_final, SV_init, SV_final, min_w, w_scale, benchmark):
        r"""**__init__(self, n_iter, D, school_size, SI_init, SI_final, SV_init, SV_final, min_w, w_scale, benchmark)**.

        Arguments:
            D {integer} -- dimension of problem

            n_iter {integer} -- number of iterations

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

        self.benchmark = benchmark
        self.Lower = self.benchmark.Lower
        self.Upper = self.benchmark.Upper
        self.D = D
        self.n_iter = n_iter

        self.school_size = school_size
        self.step_individual_init = SI_init
        self.step_individual_final = SI_final
        self.step_volitive_init = SV_init
        self.step_volitive_final = SV_final

        self.curr_step_individual = self.step_individual_init * (self.Upper - self.Lower)
        self.curr_step_volitive = self.step_volitive_init * (self.Upper - self.Lower)
        self.min_w = min_w
        self.w_scale = w_scale

        self.prev_weight_school = 0.0
        self.curr_weight_school = 0.0
        self.best_fish = None

        self.Fun = self.benchmark.function()

    def generate_uniform_coordinates(self):
        """Return Numpy array with uniform distribution."""
        x = np.zeros((self.school_size, self.D))
        for i in range(self.school_size):
            x[i] = np.random.uniform(self.Lower, self.Upper, self.D)
        return x

    def gen_weight(self):
        """Get initial weight for fish."""
        return self.w_scale / 2.0

    def init_fish(self, pos):
        """Create a new fish to a given position."""
        fish = Fish(self.D)
        fish.pos = pos
        fish.weight = self.gen_weight()
        fish.cost = self.Fun(self.D, fish.pos)
        return fish

    def init_school(self):
        """Initialize fish school with uniform distribution."""
        self.best_fish = Fish(self.D)
        self.best_fish.cost = np.inf
        self.curr_weight_school = 0.0
        self.prev_weight_school = 0.0
        self.school = []

        positions = self.generate_uniform_coordinates()

        for idx in range(self.school_size):
            fish = self.init_fish(positions[idx])
            self.school.append(fish)
            self.curr_weight_school += fish.weight
        self.prev_weight_school = self.curr_weight_school
        self.update_best_fish()

    def max_delta_cost(self):
        """Find maximum delta cost - return 0 if none of the fishes moved."""
        max_ = 0
        for fish in self.school:
            if max_ < fish.delta_cost:
                max_ = fish.delta_cost
        return max_

    def total_school_weight(self):
        """Calculate and update current weight of fish school."""
        self.prev_weight_school = self.curr_weight_school
        self.curr_weight_school = sum(fish.weight for fish in self.school)

    def calculate_barycenter(self):
        """Calculate barycenter of fish school."""
        barycenter = np.zeros((self.D,), dtype=np.float)
        density = 0.0

        for fish in self.school:
            density += fish.weight
            for dim in range(self.D):
                barycenter[dim] += (fish.pos[dim] * fish.weight)
        for dim in range(self.D):
            barycenter[dim] = barycenter[dim] / density

        return barycenter

    def update_steps(self, curr_iter):
        """Update step length for individual and volatile steps."""
        self.curr_step_individual = self.step_individual_init - curr_iter * float(
            self.step_individual_init - self.step_individual_final) / self.n_iter

        self.curr_step_volitive = self.step_volitive_init - curr_iter * float(
            self.step_volitive_init - self.step_volitive_final) / self.n_iter

    def update_best_fish(self):
        """Find and update current best fish."""
        for fish in self.school:
            if self.best_fish.cost > fish.cost:
                self.best_fish = copy.copy(fish)

    def feeding(self):
        """Feed all fishes."""
        for fish in self.school:
            if self.max_delta_cost():
                fish.weight = fish.weight + (fish.delta_cost / self.max_delta_cost())
            if fish.weight > self.w_scale:
                fish.weight = self.w_scale
            elif fish.weight < self.min_w:
                fish.weight = self.min_w

    def individual_movement(self):
        """Perform individual movement for each fish."""
        for fish in self.school:
            new_pos = np.zeros((self.D,), dtype=np.float)
            for dim in range(self.D):
                new_pos[dim] = fish.pos[dim] + (self.curr_step_individual * np.random.uniform(-1, 1))
                if new_pos[dim] < self.Lower:
                    new_pos[dim] = self.Lower
                elif new_pos[dim] > self.Upper:
                    new_pos[dim] = self.Upper
            cost = self.Fun(self.D, new_pos)
            if cost < fish.cost:
                fish.delta_cost = abs(cost - fish.cost)
                fish.cost = cost
                delta_pos = np.zeros((self.D,), dtype=np.float)
                for idx in range(self.D):
                    delta_pos[idx] = new_pos[idx] - fish.pos[idx]
                fish.delta_pos = delta_pos
                fish.pos = new_pos
            else:
                fish.delta_pos = np.zeros((self.D,), dtype=np.float)
                fish.delta_cost = 0

    def collective_instinctive_movement(self):
        """Perform collective instictive movement."""
        cost_eval_enhanced = np.zeros((self.D,), dtype=np.float)
        density = 0.0
        for fish in self.school:
            density += fish.delta_cost
            for dim in range(self.D):
                cost_eval_enhanced[dim] += (fish.delta_pos[dim] * fish.delta_cost)
        for dim in range(self.D):
            if density != 0:
                cost_eval_enhanced[dim] = cost_eval_enhanced[dim] / density
        for fish in self.school:
            new_pos = np.zeros((self.D,), dtype=np.float)
            for dim in range(self.D):
                new_pos[dim] = fish.pos[dim] + cost_eval_enhanced[dim]
                if new_pos[dim] < self.Lower:
                    new_pos[dim] = self.Lower
                elif new_pos[dim] > self.Upper:
                    new_pos[dim] = self.Upper

            fish.pos = new_pos

    def collective_volitive_movement(self):
        """Perform collective volitive movement."""
        self.total_school_weight()
        barycenter = self.calculate_barycenter()
        for fish in self.school:
            new_pos = np.zeros((self.D,), dtype=np.float)
            for dim in range(self.D):
                if self.curr_weight_school > self.prev_weight_school:
                    new_pos[dim] = fish.pos[dim] - ((fish.pos[dim] - barycenter[dim]) * self.curr_step_volitive *
                                                    np.random.uniform(0, 1))
                else:
                    new_pos[dim] = fish.pos[dim] + ((fish.pos[dim] - barycenter[dim]) * self.curr_step_volitive *
                                                    np.random.uniform(0, 1))
                if new_pos[dim] < self.Lower:
                    new_pos[dim] = self.Lower
                elif new_pos[dim] > self.Upper:
                    new_pos[dim] = self.Upper

            cost = self.Fun(self.D, new_pos)
            fish.cost = cost
            fish.pos = new_pos

    def run(self):
        """Run the algorithm."""
        self.init_school()
        for i in range(self.n_iter):
            self.individual_movement()
            self.update_best_fish()
            self.feeding()
            self.collective_instinctive_movement()
            self.collective_volitive_movement()
            self.update_steps(i)
            self.update_best_fish()
        return (self.best_fish.pos, self.best_fish.cost)
