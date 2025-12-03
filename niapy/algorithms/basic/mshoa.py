# encoding=utf8

"""Mantis Search Algorithm module."""

import numpy as np

from niapy.algorithms.algorithm import Algorithm

__all__ = [
    'MantisSearchAlgorithm',
    'MShOA'
]


class MantisSearchAlgorithm(Algorithm):
    r"""Implementation of Mantis Shrimp Optimization Algorithm (MShOA).

    Algorithm:
        Mantis Shrimp Optimization Algorithm

    Date:
        2025

    Authors:
        José Alfonso Sánchez Cortez, Hernán Peraza Vázquez, Adrián Fermin Peña Delgado

    License:
        MIT

    Reference paper:
        Sánchez Cortez, J. A., Peraza Vázquez, H., & Peña Delgado, A. F. (2025).
        "A Novel Bio-Inspired Optimization Algorithm Based on Mantis Shrimp Survival Tactics".
        Mathematics, 13(9), 1500. https://doi.org/10.3390/math13091500

    Attributes:
        Name (List[str]): List of strings representing algorithm names.
        polarization_rate (float): Controls switching between navigation and strike.
        strike_factor (float): Controls the speed of convergence.
        defense_factor (float): Controls the diversity around the best solution.
        defense_probability (float): Probability of applying defense mechanism.

    See Also:
        * :class:`niapy.algorithms.Algorithm`

    """

    Name = ['MantisSearchAlgorithm', 'MShOA']

    @staticmethod
    def info():
        r"""Get basic information of algorithm.

        Returns:
            str: Basic information of algorithm.

        See Also:
            * :func:`niapy.algorithms.Algorithm.info`

        """
        return r"""Sánchez Cortez, J. A., Peraza Vázquez, H., & Peña Delgado, A. F. (2025). "A Novel Bio-Inspired Optimization Algorithm Based on Mantis Shrimp Survival Tactics". Mathematics, 13(9), 1500. https://doi.org/10.3390/math13091500"""

    def __init__(self, population_size=50, polarization_rate=0.5, strike_factor=2.0,
                 defense_factor=0.1, defense_probability=0.1, *args, **kwargs):
        """Initialize MantisSearchAlgorithm.

        Args:
            population_size (int): Population size.
            polarization_rate (float): Controls switching between navigation and strike, default = 0.5.
            strike_factor (float): Controls the speed of convergence, default = 2.0.
            defense_factor (float): Controls the diversity around the best solution, default = 0.1.
            defense_probability (float): Probability of applying defense mechanism, default = 0.1.

        See Also:
            * :func:`niapy.algorithms.Algorithm.__init__`

        """
        super().__init__(population_size, *args, **kwargs)
        self.polarization_rate = polarization_rate
        self.strike_factor = strike_factor
        self.defense_factor = defense_factor
        self.defense_probability = defense_probability

    def set_parameters(self, population_size=50, polarization_rate=0.5, strike_factor=2.0,
                       defense_factor=0.1, defense_probability=0.1,*args ,**kwargs):
        r"""Set Mantis Search Algorithm main parameters.

        Args:
            population_size (int): Population size.
            polarization_rate (float): Controls switching between navigation and strike.
            strike_factor (float): Controls the speed of convergence.
            defense_factor (float): Controls the diversity around the best solution.
            defense_probability (float): Probability of applying defense mechanism.

        See Also:
            * :func:`niapy.algorithms.Algorithm.set_parameters`

        """
        super().set_parameters(population_size=population_size,*args, **kwargs)
        self.polarization_rate = polarization_rate
        self.strike_factor = strike_factor
        self.defense_factor = defense_factor
        self.defense_probability = defense_probability

    def get_parameters(self):
        r"""Get value of parameters for this instance of algorithm.

        Returns:
            Dict[str, Union[int, float]]: Dictionary which has parameters mapped to values.

        See Also:
            * :func:`niapy.algorithms.Algorithm.get_parameters`

        """
        d = super().get_parameters()
        d.update({
            'polarization_rate': self.polarization_rate,
            'strike_factor': self.strike_factor,
            'defense_factor': self.defense_factor,
            'defense_probability': self.defense_probability
        })
        return d

    def run_iteration(self, task, pop, fpop, xb, fxb, **params):
        r"""Core function of Mantis Shrimp Optimization Algorithm.

        Args:
            task (Task): Optimization task.
            pop (numpy.ndarray): Current population.
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
                5. Additional arguments (empty dict).

        See Also:
            * :class:`niapy.algorithms.algorithm.Algorithm.run_iteration`

        """
        # Generate all random numbers in bulk for vectorization
        r1 = self.random(self.population_size)  # Shape: (pop_size,)
        r2 = self.random(self.population_size)  # Shape: (pop_size,)
        r3 = self.random(self.population_size)  # Shape: (pop_size,)

        # Generate random indices for navigation phase
        random_indices = self.integers(0, self.population_size, self.population_size)  # Shape: (pop_size,)

        # Create boolean masks for vectorized conditional logic
        mask_navigation = r1 < self.polarization_rate  # Shape: (pop_size,)
        mask_defense = r3 < self.defense_probability  # Shape: (pop_size,)

        # Expand masks to match dimensions: (pop_size, 1) -> (pop_size, n_dims)
        mask_nav_expanded = mask_navigation[:, np.newaxis]  # Shape: (pop_size, 1)
        mask_def_expanded = mask_defense[:, np.newaxis]  # Shape: (pop_size, 1)

        # Expand r2 to match dimensions: (pop_size,) -> (pop_size, n_dims)
        r2_expanded = r2[:, np.newaxis]  # Shape: (pop_size, 1)

        # Phase 1: Navigation (Exploration) - Equation 1
        # pos_new = current + r2 * (random - current)
        random_pop_pos = pop[random_indices]  # Shape: (pop_size, n_dims)
        navigation_pos = pop + r2_expanded * (random_pop_pos - pop)  # Shape: (pop_size, n_dims)

        # Phase 2: Raptorial Strike (Exploitation) - Equation 2
        # pos_new = current + strike_factor * (g_best - current)
        strike_pos = pop + self.strike_factor * (xb - pop)  # Shape: (pop_size, n_dims)

        # Combine Phase 1 and Phase 2 using np.where
        # If mask_navigation is True, use navigation_pos; else use strike_pos
        pos_new = np.where(mask_nav_expanded, navigation_pos, strike_pos)  # Shape: (pop_size, n_dims)

        # Phase 3: Defense Mechanism (Stagnation Avoidance) - Scale-aware
        # pos_new = g_best + defense_factor * (ub - lb) * uniform(-1, 1)
        scale = task.upper - task.lower  # Shape: (n_dims,)
        defense_random = self.uniform(-1.0, 1.0, size=(self.population_size, task.dimension))  # Shape: (pop_size, n_dims)
        defense_pos = xb + self.defense_factor * scale * defense_random  # Shape: (pop_size, n_dims)

        # Apply defense mechanism where mask_defense is True
        pos_new = np.where(mask_def_expanded, defense_pos, pos_new)  # Shape: (pop_size, n_dims)

        # Store old fitness values for greedy selection
        fpop_old = fpop.copy()

        # Update population positions and evaluate fitness
        for i in range(self.population_size):
            # Boundary check: repair solution
            pos_corrected = task.repair(pos_new[i], rng=self.rng)
            # Evaluate fitness
            fpop_new = task.eval(pos_corrected)
            # Greedy selection: keep better solution
            if fpop_new < fpop_old[i]:
                pop[i] = pos_corrected
                fpop[i] = fpop_new
            # Update global best if better solution found
            if fpop[i] < fxb:
                xb, fxb = pop[i].copy(), fpop[i]

        return pop, fpop, xb, fxb, {}


# Alias for backward compatibility
MShOA = MantisSearchAlgorithm

