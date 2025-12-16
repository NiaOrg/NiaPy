import copy
import logging

import numpy as np

from niapy.algorithms.algorithm import Algorithm, Individual, default_individual_init
from niapy.util import objects_to_array

logging.basicConfig()
logger = logging.getLogger("niapy.algorithms.basic")
logger.setLevel("INFO")

__all__ = ["LionOptimizationAlgorithm"]


class Lion(Individual):
    r"""Implementation of population individual for Lion Optimization Algorithm.

    Algorithm:
        Lion Optimization Algorithm

    Date:
        2021

    Authors:
        Aljoša Mesarec

    License:
        MIT

    Attributes:
        gender (string): Lion gender.
        has_pride (bool): Lion has a pride.
        pride (int): Lion's pride id.
        hunting_group (int): Lion's hunting group.
        current_x (numpy.ndarray): Lion's current position
        current_f (float): Lion's current fitness
        previous_iter_best_f (float): Lion's fitness at end of previous iteration.
        has_improved (bool): Lion has improved fitness since last iteration.

    See Also:
        * :class:`niapy.algorithms.Individual`

    """

    def __init__(
        self,
        gender="m",
        has_pride=False,
        pride=-1,
        hunting_group=0,
        has_improved=True,
        **kwargs,
    ):
        r"""Initialize the Lion.

        Args:
            gender (Optional[string]): Lion's gender.
            has_pride (Optional[bool]): Lion has a pride.
            pride (Optional[int]): Lion's pride id.
            hunting_group (Optional[int]): Lion's hunting group id.
            has_improved (Optional[bool]): Lion has improved fitness since last
             iteration.

        See Also:
            * :func:`niapy.algorithms.Individual.__init__`

        """
        super().__init__(**kwargs)
        self.gender = gender
        self.has_pride = has_pride
        self.pride = pride
        self.hunting_group = hunting_group
        self.current_x = np.copy(self.x)
        self.current_f = self.f
        self.previous_iter_best_f = self.f
        self.has_improved = has_improved


class LionOptimizationAlgorithm(Algorithm):
    r"""Implementation of lion optimization algorithm.

    Algorithm:
        Lion Optimization algorithm

    Date:
        2021

    Authors:
        Aljoša Mesarec

    License:
        MIT

    Reference URL:
        https://doi.org/10.1016/j.jcde.2015.06.003

    Reference paper:
        Yazdani, Maziar, Jolai, Fariborz. Lion Optimization Algorithm (LOA):
        A nature-inspired metaheuristic algorithm. Journal of Computational Design and
        Engineering, Volume 3, Issue 1, Pages 24-36. 2016.

    Attributes:
        Name (List[str]): List of strings representing name of the algorithm.
        population_size (Optional[int]): Population size :math:`\in [1, \infty)`.
        nomad_ratio (Optional[float]): Ratio of nomad lions :math:`\in [0, 1]`.
        num_of_prides (Optional[int]): Number of prides :math:`\in [1, \infty)`.
        female_ratio (Optional[float]): Ratio of female lions in prides
         :math:`\in [0, 1]`.
        roaming_factor (Optional[float]): Roaming factor :math:`\in [0, 1]`.
        mating_factor (Optional[float]): Mating factor :math:`\in [0, 1]`.
        mutation_factor (Optional[float]): Mutation factor :math:`\in [0, 1]`.
        immigration_factor (Optional[float]): Immigration factor :math:`\in [0, 1]`.

    See Also:
        * :class:`niapy.algorithms.Algorithm`

    """

    Name = ["LionOptimizationAlgorithm", "LOA"]

    @staticmethod
    def info():
        r"""Get information about algorithm.

        Returns:
            str: Algorithm information

        See Also:
            * :func:`niapy.algorithms.Algorithm.info`

        """
        return (
            r"Yazdani, Maziar, Jolai, Fariborz. Lion Optimization Algorithm (LOA):"
            r" A nature-inspired metaheuristic algorithm. Journal of Computational "
            r"Design and Engineering, Volume 3, Issue 1, Pages 24-36. 2016."
        )

    def __init__(
        self,
        population_size=50,
        nomad_ratio=0.2,
        num_of_prides=5,
        female_ratio=0.8,
        roaming_factor=0.2,
        mating_factor=0.3,
        mutation_factor=0.2,
        immigration_factor=0.4,
        *args,
        **kwargs,
    ):
        r"""Initialize LionOptimizationAlgorithm.

        Args:
            population_size (Optional[int]): Population size :math:`\in [1, \infty)`.
            nomad_ratio (Optional[float]): Ratio of nomad lions :math:`\in [0, 1]`.
            num_of_prides (Optional[int]): Number of prides :math:`\in [1, \infty)`.
            female_ratio (Optional[float]): Ratio of female lions in prides
             :math:`\in [0, 1]`.
            roaming_factor (Optional[float]): Roaming factor :math:`\in [0, 1]`.
            mating_factor (Optional[float]): Mating factor :math:`\in [0, 1]`.
            mutation_factor (Optional[float]): Mutation factor :math:`\in [0, 1]`.
            immigration_factor (Optional[float]): Immigration factor :math:`\in [0, 1]`.


        See Also:
            * :func:`niapy.algorithms.Algorithm.__init__`

        """
        super().__init__(
            population_size,
            *args,
            individual_type=kwargs.pop("individual_type", Lion),
            initialization_function=kwargs.pop(
                "initialization_function", default_individual_init
            ),
            **kwargs,
        )
        self.nomad_ratio = nomad_ratio
        self.num_of_prides = num_of_prides
        self.female_ratio = female_ratio
        self.roaming_factor = roaming_factor
        self.mating_factor = mating_factor
        self.mutation_factor = mutation_factor
        self.immigration_factor = immigration_factor

    def set_parameters(
        self,
        population_size=50,
        nomad_ratio=0.2,
        num_of_prides=5,
        female_ratio=0.8,
        roaming_factor=0.2,
        mating_factor=0.3,
        mutation_factor=0.2,
        immigration_factor=0.4,
        *args,
        **kwargs,
    ):
        r"""Set the arguments of an algorithm.

        Args:
            population_size (Optional[int]): Population size :math:`\in [1, \infty)`.
            nomad_ratio (Optional[float]): Ratio of nomad lions :math:`\in [0, 1]`.
            num_of_prides (Optional[int]): Number of prides :math:`\in [1, \infty)`.
            female_ratio (Optional[float]): Ratio of female lions in prides
             :math:`\in [0, 1]`.
            roaming_factor (Optional[float]): Roaming factor :math:`\in [0, 1]`.
            mating_factor (Optional[float]): Mating factor :math:`\in [0, 1]`.
            mutation_factor (Optional[float]): Mutation factor :math:`\in [0, 1]`.
            immigration_factor (Optional[float]): Immigration factor :math:`\in [0, 1]`.

        See Also:
            * :func:`niapy.algorithms.Algorithm.set_parameters`

        """
        super().set_parameters(
            population_size,
            *args,
            individual_type=Lion,
            initialization_function=kwargs.pop(
                "initialization_function", default_individual_init
            ),
            **kwargs,
        )
        self.nomad_ratio = nomad_ratio
        self.num_of_prides = num_of_prides
        self.female_ratio = female_ratio
        self.roaming_factor = roaming_factor
        self.mating_factor = mating_factor
        self.mutation_factor = mutation_factor
        self.immigration_factor = immigration_factor

    def get_parameters(self):
        r"""Get parameters of the algorithm.

        Returns:
            Dict[str, Any]: Algorithm Parameters.

        """
        d = super().get_parameters()
        d.update(
            {
                "nomad_ratio": self.nomad_ratio,
                "num_of_prides": self.num_of_prides,
                "female_ratio": self.female_ratio,
                "roaming_factor": self.roaming_factor,
                "mating_factor": self.mating_factor,
                "mutation_factor": self.mutation_factor,
                "immigration_factor": self.immigration_factor,
            }
        )
        return d

    def init_population(self, task):
        r"""Initialize starting population.

        Args:
            task (Task): Optimization task.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray]:
                1. Initialized population of lions.
                2. Initialized populations function/fitness values.
                3. Additional arguments:
                    * pride_size (numpy.ndarray): Pride and nomad sizes.
                    * gender_distribution (numpy.ndarray): Pride and nomad gender
                     distributions.

        """
        pop, fpop, d = super().init_population(task)
        pop, d = self.init_population_data(pop, d)
        return pop, fpop, d

    def init_population_data(self, pop, d):
        r"""Initialize data of starting population.

        Args:
            pop (numpy.ndarray): Starting lion population
            d (Dict[str, Any]): Additional arguments

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray]:
                1. Initialized population of lions.
                2. Additional arguments:
                    * pride_size (numpy.ndarray): Pride and nomad sizes.
                    * gender_distribution (numpy.ndarray): Pride and nomad gender
                     distributions.

        """
        nomad_size = round(self.nomad_ratio * self.population_size)
        pride_size = np.full(
            self.num_of_prides + 1,
            (self.population_size - nomad_size) // self.num_of_prides,
            dtype=int,
        )
        pride_size[: (self.population_size - nomad_size) % self.num_of_prides] += 1
        pride_size[-1] = nomad_size
        gender_distribution = np.zeros((self.num_of_prides + 1, 2), dtype=int)

        start = 0
        for pride_id, size in enumerate(pride_size):
            end = start + size
            lions = pop[start:end]

            if pride_id < self.num_of_prides:
                num_females = round(self.female_ratio * size)
                for i, lion in enumerate(lions):
                    lion.has_pride = True
                    lion.pride = pride_id
                    lion.gender = "f" if i < num_females else "m"
            else:
                num_females = round((1 - self.female_ratio) * size)
                for i, lion in enumerate(lions):
                    lion.gender = "f" if i < num_females else "m"

            for lion in lions:
                if lion.gender == "f":
                    gender_distribution[pride_id, 0] += 1
                else:
                    gender_distribution[pride_id, 1] += 1

            start = end

        d.update({"pride_size": pride_size, "gender_distribution": gender_distribution})
        return pop, d

    def _initialize_hunt(self, pride, task):
        prey_x = np.zeros(task.dimension)
        group_fitness = np.zeros(4)
        hunters = []

        for lion in pride:
            if lion.gender == "f":
                lion.hunting_group = self.integers(0, 4)
                group_fitness[lion.hunting_group] += lion.current_f

            if lion.hunting_group != 0:
                prey_x += lion.current_x
                hunters.append(lion)

        if hunters:
            prey_x /= len(hunters)

        return task.repair(prey_x), group_fitness, hunters

    def _move_opposition(self, lion, prey_x, task):
        for i in range(task.dimension):
            opposite = 2 * prey_x[i] - lion.current_x[i]
            if opposite < prey_x[i]:
                lion.current_x[i] = self.uniform(opposite, prey_x[i])
            elif opposite > prey_x[i]:
                lion.current_x[i] = self.uniform(prey_x[i], opposite)

        lion.current_x = task.repair(lion.current_x)
        lion.current_f = task.eval(lion.current_x)

    def _move_center(self, lion, prey_x, task):
        for i in range(task.dimension):
            if lion.current_x[i] < prey_x[i]:
                lion.current_x[i] = self.uniform(lion.current_x[i], prey_x[i])
            elif lion.current_x[i] > prey_x[i]:
                lion.current_x[i] = self.uniform(prey_x[i], lion.current_x[i])

        lion.current_x = task.repair(lion.current_x)
        lion.current_f = task.eval(lion.current_x)

    def _move_hunter(self, lion, prey_x, left, right, center, task):
        g = lion.hunting_group

        if g == left or g == right:
            self._move_opposition(lion, prey_x, task)
        elif g == center:
            self._move_center(lion, prey_x, task)

    def _update_prey(self, lion, prey_x, task):
        if lion.current_f < lion.f:
            lion.x = np.copy(lion.current_x)
            lion.f = lion.current_f

            improvement = 1 - lion.f / lion.previous_iter_best_f
            prey_x = prey_x + self.random() * improvement * (prey_x - lion.current_x)
            return task.repair(prey_x)

        return prey_x

    def hunting(self, population, pride_size, task):
        r"""Pride female hunters go hunting.

        Args:
            population (numpy.ndarray): Lion population.
            pride_size (numpy.ndarray): Pride and nomad sizes.
            task (Task): Optimization task.

        Returns:
            numpy.ndarray: Lion population that finished with hunting.

        """
        num_of_prides = len(pride_size) - 1
        index = 0

        for i in range(num_of_prides):
            size = pride_size[i]
            pride = population[index : index + size]

            prey_x, group_fitness, hunters = self._initialize_hunt(pride, task)
            right, left, center = np.argsort(group_fitness[1:]) + 1

            for lion in hunters:
                self._move_hunter(lion, prey_x, left, right, center, task)
                prey_x = self._update_prey(lion, prey_x, task)

            index += size

        return population

    def move_to_safe_place(self, population, pride_size, task):
        r"""Female pride lions move towards position with good fitness.

        Args:
            population (numpy.ndarray): Lion population.
            pride_size (numpy.ndarray): Pride and nomad sizes.
            task (Task): Optimization task.

        Returns:
            numpy.ndarray: Lion population that finished with moving to safe place.

        """
        num_of_prides = len(pride_size) - 1
        index_counter = 0
        search_range = np.linalg.norm(task.upper[0] - task.lower[0])

        for pride_idx in range(num_of_prides):
            curr_pride_size = pride_size[pride_idx]
            pride = population[index_counter : index_counter + curr_pride_size]

            # Build pride territory
            territory = []
            num_of_improvements = 0
            for lion in pride:
                territory.append(lion)
                num_of_improvements += lion.has_improved

            if len(territory) > 1:
                tournament_size = max(2, int(np.ceil(num_of_improvements / 2)))
                indices = self.rng.choice(
                    len(territory), tournament_size, replace=False
                )
                selected = [territory[i] for i in indices]

                winner = min(selected, key=lambda selected_lion: selected_lion.f)
            else:
                winner = territory[0]
            winner_x = winner.x.copy()

            # Move female non-hunters
            for lion in pride:
                if lion.gender != "f" or lion.hunting_group != 0:
                    continue

                r_one = winner_x - lion.x
                r_one_norm = np.linalg.norm(r_one)

                if r_one_norm == 0:
                    r_two = np.zeros_like(r_one)
                    r_two[self.integers(0, len(r_one))] = 1
                else:
                    random_vec = self.standard_normal(len(r_one))
                    projection = (r_one @ random_vec) / (r_one @ r_one)
                    r_two = random_vec - projection * r_one

                d = r_one_norm / search_range
                rnd = self.random()
                rnd_u = self.uniform(-1, 1)
                angle = self.uniform(-np.pi / 6, np.pi / 6)

                lion.current_x = (
                    lion.current_x
                    + 2 * d * rnd * r_one
                    + rnd_u * np.tan(angle) * d * r_two
                )

                lion.current_x = task.repair(lion.current_x)
                lion.current_f = task.eval(lion.current_x)

                if lion.current_f < lion.f:
                    lion.x = lion.current_x.copy()
                    lion.f = lion.current_f

            index_counter += curr_pride_size

        return population

    def roaming(self, population, pride_size, task):
        r"""Male lions move towards new position.

        Args:
            population (numpy.ndarray): Lion population.
            pride_size (numpy.ndarray): Pride and nomad sizes.
            task (Task): Optimization task.

        Returns:
            numpy.ndarray: Lion population that finished with roaming.

        """
        num_of_prides = len(pride_size) - 1
        index_counter = 0
        search_range = np.linalg.norm(task.upper[0] - task.lower[0])

        # Pride lions roam
        for pride_idx in range(num_of_prides):
            curr_pride_size = pride_size[pride_idx]
            pride = population[index_counter : index_counter + curr_pride_size]
            pride_len = len(pride)

            for lion in pride:
                if lion.gender != "m":
                    continue

                num_selected = round(curr_pride_size * self.roaming_factor)
                if num_selected <= 0:
                    continue

                sel_idx = self.rng.choice(pride_len, size=num_selected, replace=False)
                selected_x = [pride[i].x.copy() for i in sel_idx]

                for sx in selected_x:
                    d = np.linalg.norm(sx - lion.x) / search_range
                    x = self.uniform(0, 2 * d)
                    angle = self.uniform(-np.pi / 6, np.pi / 6)

                    lion.current_x += x * d * np.tan(angle)
                    lion.current_x = task.repair(lion.current_x)
                    lion.current_f = task.eval(lion.current_x)

                    if lion.current_f < lion.f:
                        lion.x = lion.current_x.copy()
                        lion.f = lion.current_f

            index_counter += curr_pride_size

        # Nomad lions roam
        nomad_size = pride_size[-1]
        nomads = population[len(population) - nomad_size :]

        for lion in nomads:
            best_nomad_fitness = np.min([c_l.current_f for c_l in nomads])
            roaming_probability = 0.1 + np.minimum(
                0.5, (lion.current_f - best_nomad_fitness) / best_nomad_fitness
            )

            if self.random() <= roaming_probability:
                lion.current_x = self.uniform(task.lower, task.upper, task.dimension)
                lion.current_f = task.eval(lion.current_x)

            if lion.current_f < lion.f:
                lion.x = lion.current_x.copy()
                lion.f = lion.current_f

        return population

    def create_offspring(self, female, males, pride_idx, has_pride, task):
        offspring = []
        # Handle edge case when there are no males
        if len(males) == 0:
            return offspring
        # Handle edge case when there's only 1 male or for nomads
        if not has_pride or len(males) == 1:
            n_mating_males = 1
        else:
            n_mating_males = self.integers(1, len(males))
        indices = self.rng.choice(len(males), n_mating_males, replace=False)
        mating_males = [males[i] for i in indices]

        beta = self.normal(0.5, 0.1)
        males_mean_x = np.mean([m.x for m in mating_males], axis=0)

        offspring1 = beta * female.x + (1 - beta) * males_mean_x
        offspring2 = (1 - beta) * female.x + beta * males_mean_x

        for i in range(task.dimension):
            if self.random() < self.mutation_factor:
                offspring1[i] = self.uniform(task.lower[i], task.upper[i])
            if self.random() < self.mutation_factor:
                offspring2[i] = self.uniform(task.lower[i], task.upper[i])

        if self.random() < 0.5:
            g1 = "m"
            g2 = "f"
        else:
            g1 = "f"
            g2 = "m"

        cubs = [(offspring1, g1), (offspring2, g2)]

        for x, gender in cubs:
            cub = copy.copy(female)
            cub.x = x.copy()
            cub.has_pride = has_pride
            cub.pride = pride_idx if has_pride else -1
            cub.hunting_group = 0
            cub.has_improved = True
            cub.gender = gender

            cub.evaluate(task)
            cub.current_x = x.copy()
            cub.current_f = cub.f
            cub.previous_iter_best_f = cub.f + 1

            offspring.append(cub)

        return offspring

    def mating(self, population, pride_size, gender_distribution, task):
        r"""Female lions mate with male lions to produce offspring.

        Args:
            population (numpy.ndarray): Lion population.
            pride_size (numpy.ndarray): Pride and nomad sizes.
            gender_distribution (numpy.ndarray): Pride and nomad gender distribution.
            task (Task): Optimization task.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray]:
                1. Lion population that finished with mating.
                2. Pride and nomad excess gender quantities.

        """
        num_prides = len(pride_size) - 1
        new_population = []
        excess = np.zeros((num_prides + 1, 2), dtype=int)

        idx = 0
        for pride_i in range(num_prides):
            size = pride_size[pride_i]
            pride = population[idx : idx + size]
            idx += size
            num_males = gender_distribution[pride_i][0]
            if num_males == 0:
                continue

            males = [lion for lion in pride if lion.gender == "m"]

            females = [lion for lion in pride if lion.gender == "f"]

            pride_cubs = []
            for female in females:
                if self.random() < self.mating_factor:
                    cubs = self.create_offspring(female, males, pride_i, True, task)
                    pride_cubs.extend(cubs)

            if pride_cubs:
                n_f = sum(1 for c in pride_cubs if c.gender == "f")
                n_m = len(pride_cubs) - n_f
                excess[pride_i][0] += n_f
                excess[pride_i][1] += n_m
                pride_size[pride_i] += len(pride_cubs)
                gender_distribution[pride_i][0] += n_f
                gender_distribution[pride_i][1] += n_m

            new_population.extend(pride)
            new_population.extend(pride_cubs)

        nomad_size = pride_size[-1]
        nomads = population[-nomad_size:]

        nomad_males = [lion for lion in nomads if lion.gender == "m"]
        nomad_females = [lion for lion in nomads if lion.gender == "f"]

        nomad_cubs = []
        for female in nomad_females:
            if self.random() < self.mating_factor:
                cubs = self.create_offspring(female, nomad_males, -1, False, task)
                nomad_cubs.extend(cubs)

        if nomad_cubs:
            n_f = sum(1 for c in nomad_cubs if c.gender == "f")
            n_m = len(nomad_cubs) - n_f
            excess[-1][0] += n_f
            excess[-1][1] += n_m
            pride_size[-1] += len(nomad_cubs)
            gender_distribution[-1][0] += n_f
            gender_distribution[-1][1] += n_m

        new_population.extend(nomads)
        new_population.extend(nomad_cubs)

        return objects_to_array(new_population), excess

    def defense(self, population, pride_size, gender_distribution, excess):
        r"""Male lions attack other lions in pride.

        Args:
            population (numpy.ndarray): Lion population.
            pride_size (numpy.ndarray): Pride and nomad sizes.
            gender_distribution (numpy.ndarray): Pride and nomad gender distribution.
            excess (numpy.ndarray): Pride and nomad excess members.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray]:
                1. Lion population that finished with defending.
                2. Pride and nomad excess gender quantities.

        """
        num_of_prides = len(pride_size) - 1
        population = list(population)

        original_pride_lions = []
        new_nomads = []

        index = 0
        for pride_i in range(num_of_prides):
            curr_pride_size = pride_size[pride_i]
            excess_males = excess[pride_i][1]

            pride_lions = population[index : index + curr_pride_size]
            index += curr_pride_size

            males = [lion for lion in pride_lions if lion.gender == "m"]
            females = [lion for lion in pride_lions if lion.gender == "f"]

            males.sort(key=lambda lion: lion.current_f, reverse=True)

            original_pride_lions.extend(females)
            original_pride_lions.extend(males[excess_males:])
            new_nomads.extend(males[:excess_males])

        moved_population = []
        moved_population.extend(original_pride_lions)

        original_nomads_size = pride_size[-1]
        moved_population.extend(population[-original_nomads_size:])

        for lion in new_nomads:
            excess[lion.pride][1] -= 1
            gender_distribution[lion.pride][1] -= 1
            pride_size[lion.pride] -= 1

            lion.has_pride = False
            lion.pride = -1

            excess[-1][1] += 1
            gender_distribution[-1][1] += 1
            pride_size[-1] += 1

            moved_population.append(lion)

        nomads_size = pride_size[-1]
        nomads = moved_population[-nomads_size:]

        for nomad in nomads:
            attack_mask = self.random(size=num_of_prides) < 0.5
            index = 0

            for pride_i in range(num_of_prides):
                curr_pride_size = pride_size[pride_i]

                if not attack_mask[pride_i]:
                    index += curr_pride_size
                    continue

                for pos in range(index, index + curr_pride_size):
                    pride_lion = moved_population[pos]

                    if (
                        pride_lion.gender == "m"
                        and nomad.current_f < pride_lion.current_f
                    ):
                        break

                index += curr_pride_size

        return objects_to_array(moved_population), excess

    def _migrate_pride_females(
        self, population, pride_size, gender_distribution, excess
    ):
        new_nomads = []
        original_pride_lions = []

        num_of_prides = len(pride_size) - 1
        index = 0

        for pride_i in range(num_of_prides):
            curr_pride_size = pride_size[pride_i]

            num_females = gender_distribution[pride_i][0]
            excess_females = excess[pride_i][0]
            females_to_migrate = excess_females + round(
                (num_females - excess_females) * self.immigration_factor
            )

            females = []

            for lion in population[index : index + curr_pride_size]:
                if lion.gender == "m":
                    original_pride_lions.append(lion)
                else:
                    females.append(lion)

            migrate_mask = np.zeros(num_females, dtype=int)
            migrate_mask[:females_to_migrate] = 1
            self.rng.shuffle(migrate_mask)

            for i, lion in enumerate(females):
                if migrate_mask[i]:
                    new_nomads.append(lion)
                else:
                    original_pride_lions.append(lion)

            index += curr_pride_size

        return original_pride_lions, new_nomads

    @staticmethod
    def _build_population_after_pride_migration(
        population,
        original_pride_lions,
        new_nomads,
        pride_size,
        gender_distribution,
        excess,
    ):
        moved_population = []

        for lion in original_pride_lions:
            moved_population.append(lion)

        original_nomads_size = pride_size[-1]
        start = len(population) - original_nomads_size
        for lion in population[start:]:
            moved_population.append(lion)

        for lion in new_nomads:
            lion_copy = copy.copy(lion)

            excess[lion_copy.pride][0] -= 1
            gender_distribution[lion_copy.pride][0] -= 1
            pride_size[lion_copy.pride] -= 1

            lion_copy.has_pride = False
            lion_copy.pride = -1

            moved_population.append(lion_copy)

            excess[-1][0] += 1
            gender_distribution[-1][0] += 1
            pride_size[-1] += 1

        return moved_population

    def _select_nomad_females_for_prides(
        self, moved_population, original_population, pride_size, excess
    ):
        num_of_prides = len(pride_size) - 1

        total_empty_spots = sum(abs(excess[i][0]) for i in range(num_of_prides))

        nomad_females = []
        nomad_males = []

        original_nomads_size = pride_size[-1]
        start = len(original_population) - original_nomads_size

        for lion in moved_population[start:]:
            if lion.gender == "f":
                nomad_females.append(lion)
            else:
                nomad_males.append(lion)

        nomad_females.sort(key=lambda female: female.current_f)

        nomad_females_to_move = nomad_females[:total_empty_spots]
        nomad_females_to_keep = nomad_females[total_empty_spots:]

        self.rng.shuffle(nomad_females_to_move)

        return nomad_females_to_move, nomad_females_to_keep, nomad_males

    @staticmethod
    def _assemble_final_population(
        moved_population,
        nomad_females_to_move,
        nomad_females_to_keep,
        nomad_males,
        pride_size,
        gender_distribution,
        excess,
    ):
        final_population = []

        index_pride = 0
        index_females = 0
        num_of_prides = len(pride_size) - 1

        for pride_i in range(num_of_prides):
            curr_pride_size = pride_size[pride_i]

            for lion in moved_population[index_pride : index_pride + curr_pride_size]:
                final_population.append(lion)

            empty_spots = abs(excess[pride_i][0])

            for lion in nomad_females_to_move[
                index_females : index_females + empty_spots
            ]:
                lion_copy = copy.copy(lion)

                excess[lion_copy.pride][0] -= 1
                gender_distribution[lion_copy.pride][0] -= 1
                pride_size[lion_copy.pride] -= 1

                lion_copy.has_pride = True
                lion_copy.pride = pride_i

                final_population.append(lion_copy)

                excess[pride_i][0] += 1
                gender_distribution[pride_i][0] += 1
                pride_size[pride_i] += 1

            index_pride += curr_pride_size
            index_females += empty_spots

        final_population.extend(nomad_females_to_keep)
        final_population.extend(nomad_males)

        return final_population

    def migration(self, population, pride_size, gender_distribution, excess):
        r"""Female lions randomly become nomad.

        Args:
            population (numpy.ndarray): Lion population.
            pride_size (numpy.ndarray): Pride and nomad sizes.
            gender_distribution (numpy.ndarray): Pride and nomad gender distribution.
            excess (numpy.ndarray): Pride and nomad excess members.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray]:
                1. Lion population that finished with migration.
                2. Pride and nomad excess gender quantities.

        """
        original_pride_lions, new_nomads = self._migrate_pride_females(
            population, pride_size, gender_distribution, excess
        )

        moved_population = self._build_population_after_pride_migration(
            population,
            original_pride_lions,
            new_nomads,
            pride_size,
            gender_distribution,
            excess,
        )

        (nomad_females_to_move, nomad_females_to_keep, nomad_males) = (
            self._select_nomad_females_for_prides(
                moved_population, population, pride_size, excess
            )
        )

        final_population = self._assemble_final_population(
            moved_population,
            nomad_females_to_move,
            nomad_females_to_keep,
            nomad_males,
            pride_size,
            gender_distribution,
            excess,
        )

        return objects_to_array(final_population), excess

    @staticmethod
    def population_equilibrium(population, pride_size, gender_distribution, excess):
        r"""Remove extra nomad lions.

        Args:
            population (numpy.ndarray): Lion population.
            pride_size (numpy.ndarray): Pride and nomad sizes.
            gender_distribution (numpy.ndarray): Pride and nomad gender distribution.
            excess (numpy.ndarray): Pride and nomad excess members.

        Returns:
            numpy.ndarray: Lion population with removed extra nomads.

        """
        kept_nomads = []
        nomad_size = pride_size[-1]
        # Number of males and females that need to be removed.
        remove_females, remove_males = excess[-1]
        nomads = population[-nomad_size:]
        nomad_males = [nomad for nomad in nomads if nomad.gender == "m"]
        nomad_females = [nomad for nomad in nomads if nomad.gender == "f"]

        # Sort lists descendingly.
        nomad_males = sorted(nomad_males, key=lambda male: male.current_f, reverse=True)
        nomad_females = sorted(
            nomad_females, key=lambda female: female.current_f, reverse=True
        )

        # Remove extra lions that have bad fitness, keep the rest.
        kept_nomads.extend(nomad_males[remove_males:])
        kept_nomads.extend(nomad_females[remove_females:])

        final_population = population[: len(population) - nomad_size].tolist()
        final_population.extend(kept_nomads)

        pride_size[-1] -= remove_females + remove_males
        gender_distribution[-1][0] -= remove_females
        gender_distribution[-1][1] -= remove_males

        return objects_to_array(final_population)

    @staticmethod
    def data_correction(population):
        r"""Update lion's data if his position has improved since last iteration.

        Args:
            population (numpy.ndarray): Lion population.

        Returns:
            numpy.ndarray: Lion population with corrected data.

        """
        for lion in population:
            if lion.f < lion.previous_iter_best_f:
                lion.has_improved = True
                lion.previous_iter_best_f = lion.f
            else:
                lion.has_improved = False
        return population

    def run_iteration(
        self, task, population, population_fitness, best_x, best_fitness, **params
    ):
        pride_size = params.pop("pride_size")
        gender_distribution = params.pop("gender_distribution")

        lions = self.hunting(population, pride_size, task)
        lions = self.move_to_safe_place(lions, pride_size, task)
        lions = self.roaming(lions, pride_size, task)
        lions, excess = self.mating(lions, pride_size, gender_distribution, task)
        lions, excess = self.defense(lions, pride_size, gender_distribution, excess)
        lions, excess = self.migration(lions, pride_size, gender_distribution, excess)
        lions = self.population_equilibrium(
            lions, pride_size, gender_distribution, excess
        )
        lions = self.data_correction(lions)

        lions_fitness = np.asarray([lion.f for lion in lions])
        best_x, best_fitness = self.get_best(lions, lions_fitness, best_x, best_fitness)
        return (
            lions,
            lions_fitness,
            best_x,
            best_fitness,
            {"pride_size": pride_size, "gender_distribution": gender_distribution},
        )
