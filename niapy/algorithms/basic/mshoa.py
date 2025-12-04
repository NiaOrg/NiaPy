# encoding=utf8

"""Mantis Search Algorithm module."""

import logging
import numpy as np

from niapy.algorithms.algorithm import Algorithm

__all__ = [
    'MantisSearchAlgorithm',
    'MShOA'
]

logger = logging.getLogger(__name__)


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
        k_value (float): Defense/shelter strategy parameter (default: 0.3).
        use_reflection (bool): Use reflection-based boundary handling (default: True).
        normalize_space (bool): Use normalized search space [-1, 1] (default: False).
        debug_pti (bool): Enable PTI distribution debugging (default: False).

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

    def __init__(self, population_size=50, k_value=0.3, use_reflection=True,
                 normalize_space=False, debug_pti=False, *args, **kwargs):
        """Initialize MantisSearchAlgorithm.

        Args:
            population_size (int): Population size.
            k_value (float): Defense/shelter strategy parameter, default = 0.3.
            use_reflection (bool): Use reflection-based boundary handling, default = True.
            normalize_space (bool): Use normalized search space [-1, 1], default = False.
            debug_pti (bool): Enable PTI distribution debugging, default = False.

        See Also:
            * :func:`niapy.algorithms.Algorithm.__init__`

        """
        super().__init__(population_size, *args, **kwargs)
        self.k_value = k_value
        self.use_reflection = use_reflection
        self.normalize_space = normalize_space
        self.debug_pti = debug_pti

    def set_parameters(self, population_size=50, k_value=0.3, use_reflection=True,
                      normalize_space=False, debug_pti=False, *args, **kwargs):
        r"""Set Mantis Search Algorithm main parameters.

        Args:
            population_size (int): Population size.
            k_value (float): Defense/shelter strategy parameter.
            use_reflection (bool): Use reflection-based boundary handling.
            normalize_space (bool): Use normalized search space [-1, 1].
            debug_pti (bool): Enable PTI distribution debugging.

        See Also:
            * :func:`niapy.algorithms.Algorithm.set_parameters`

        """
        super().set_parameters(population_size=population_size, *args, **kwargs)
        self.k_value = k_value
        self.use_reflection = use_reflection
        self.normalize_space = normalize_space
        self.debug_pti = debug_pti

    def get_parameters(self):
        r"""Get value of parameters for this instance of algorithm.

        Returns:
            Dict[str, Union[int, float, bool]]: Dictionary which has parameters mapped to values.

        See Also:
            * :func:`niapy.algorithms.Algorithm.get_parameters`

        """
        d = super().get_parameters()
        d.update({
            'k_value': self.k_value,
            'use_reflection': self.use_reflection,
            'normalize_space': self.normalize_space,
            'debug_pti': self.debug_pti
        })
        return d

    def _reflect_repair(self, x, lower, upper):
        r"""Apply reflection-based boundary handling.

        Reflection rule:
        - if x < lb: x = lb + (lb - x)
        - if x > ub: x = ub - (x - ub)
        Repeated reflections are applied until all components are inside [lower, upper].

        Args:
            x (numpy.ndarray): Solution(s) to repair. Can be 1D or 2D array.
            lower (numpy.ndarray): Lower bounds.
            upper (numpy.ndarray): Upper bounds.

        Returns:
            numpy.ndarray: Repaired solution(s).

        """
        x = np.asarray(x)
        is_1d = (x.ndim == 1)
        if is_1d:
            x = x[np.newaxis, :]
        # Apply reflection iteratively until all components are in bounds
        max_iterations = 10  # Safety limit
        for _ in range(max_iterations):
            # Check violations
            below_lower = x < lower
            above_upper = x > upper
            
            if not (np.any(below_lower) or np.any(above_upper)):
                break

            # Apply reflection
            x = np.where(below_lower, lower + (lower - x), x)
            x = np.where(above_upper, upper - (x - upper), x)

        # Final clamp to ensure bounds (safety fallback)
        x = np.clip(x, lower, upper)

        if is_1d:
            x = x[0]
        return x

    def _normalize_to_real(self, z_normalized, lower, upper):
        r"""Convert normalized space [-1, 1] to real search space [lower, upper].

        Args:
            z_normalized (numpy.ndarray): Position in normalized space.
            lower (numpy.ndarray): Lower bounds.
            upper (numpy.ndarray): Upper bounds.

        Returns:
            numpy.ndarray: Position in real search space.

        """
        return lower + (z_normalized + 1.0) * 0.5 * (upper - lower)

    def _normalize_from_real(self, x_real, lower, upper):
        r"""Convert real search space [lower, upper] to normalized space [-1, 1].

        Args:
            x_real (numpy.ndarray): Position in real search space.
            lower (numpy.ndarray): Lower bounds.
            upper (numpy.ndarray): Upper bounds.

        Returns:
            numpy.ndarray: Position in normalized space.

        """
        return 2.0 * (x_real - lower) / (upper - lower) - 1.0

    def _determine_polarization_type_vectorized(self, angles):
        r"""Determine polarization type based on angles (Eq. 5 in paper) - Vectorized.

        Exact piecewise function from Eq. (5):
        - Type 1: 3π/8 ≤ θ ≤ 5π/8
        - Type 2: (0 ≤ θ ≤ π/8) or (7π/8 ≤ θ ≤ π)
        - Type 3: (π/8 < θ < 3π/8) or (5π/8 < θ < 7π/8)

        Args:
            angles (numpy.ndarray): Array of angles in radians, 0 ≤ θ ≤ π.

        Returns:
            numpy.ndarray: Array of polarization types (1, 2, or 3).

        """
        pi_8 = np.pi / 8
        pi = np.pi

        # Initialize result array with default value 2
        result = np.full(angles.shape, 2, dtype=int)

        # Type 1: 3π/8 ≤ θ ≤ 5π/8
        mask_type1 = (3 * pi_8 <= angles) & (angles <= 5 * pi_8)
        result[mask_type1] = 1

        # Type 2: (0 ≤ θ ≤ π/8) or (7π/8 ≤ θ ≤ π)
        mask_type2 = ((0 <= angles) & (angles <= pi_8)) | ((7 * pi_8 <= angles) & (angles <= pi))
        result[mask_type2] = 2

        # Type 3: (π/8 < θ < 3π/8) or (5π/8 < θ < 7π/8)
        mask_type3 = ((pi_8 < angles) & (angles < 3 * pi_8)) | ((5 * pi_8 < angles) & (angles < 7 * pi_8))
        result[mask_type3] = 3

        return result

    def _calculate_angular_difference_vectorized(self, angles, polarization_types):
        r"""Calculate angular differences (LAD or RAD) according to Eq. 6 in paper - Vectorized.

        Eq. 6 defines the angular difference as the minimum absolute difference
        between the angle and the reference angles for the given polarization type:
        - Type 1 → reference angles: π/2
        - Type 2 → reference angles: 0 or π
        - Type 3 → reference angles: π/4 or 3π/4

        Args:
            angles (numpy.ndarray): Array of angles (LPA or RPA).
            polarization_types (numpy.ndarray): Array of polarization types (1, 2, or 3).

        Returns:
            numpy.ndarray: Array of minimum angular differences (LAD or RAD).

        """
        pi = np.pi
        result = np.zeros_like(angles)
        
        # Type 1: reference angle is π/2
        mask_type1 = (polarization_types == 1)
        result[mask_type1] = np.abs(angles[mask_type1] - pi / 2)
        
        # Type 2: reference angles are 0 or π
        mask_type2 = (polarization_types == 2)
        if np.any(mask_type2):
            angles_type2 = angles[mask_type2]
            diff_0 = np.abs(angles_type2 - 0.0)
            diff_pi = np.abs(angles_type2 - pi)
            result[mask_type2] = np.minimum(diff_0, diff_pi)
        
        # Type 3: reference angles are π/4 or 3π/4
        mask_type3 = (polarization_types == 3)
        if np.any(mask_type3):
            angles_type3 = angles[mask_type3]
            diff_pi4 = np.abs(angles_type3 - pi / 4)
            diff_3pi4 = np.abs(angles_type3 - 3 * pi / 4)
            result[mask_type3] = np.minimum(diff_pi4, diff_3pi4)
        
        return result

    def init_population(self, task):
        r"""Initialize population and PTI (Polarization Type Indicator) vector.

        Args:
            task (Task): Optimization task.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray, dict]:
                1. Initial population.
                2. Initial population fitness/function values.
                3. Additional arguments:
                    * pti (numpy.ndarray): Polarization Type Indicator vector (values 1, 2, or 3).

        See Also:
            * :func:`niapy.algorithms.Algorithm.init_population`

        """
        pop, fpop, d = super().init_population(task)

        # Optional normalization: convert to normalized space if enabled
        if self.normalize_space:
            pop = self._normalize_from_real(pop, task.lower, task.upper)
            # Convert back to real space for initial evaluation
            pop_real = self._normalize_to_real(pop, task.lower, task.upper)
            fpop = np.apply_along_axis(task.eval, 1, pop_real)
        else:
            fpop = np.apply_along_axis(task.eval, 1, pop)

        # Initialize PTI vector: round(1 + 2 * rand) gives values {1, 2, 3}
        pti = np.round(1 + 2 * self.random(self.population_size)).astype(int)
        d['pti'] = pti
        return pop, fpop, d

    def run_iteration(self, task, pop, fpop, xb, fxb, **params):
        r"""Core function of Mantis Shrimp Optimization Algorithm - Fully Vectorized.

        Implements Algorithm 1 (PTI Update) and Algorithm 2 (Position Update)
        exactly as described in the paper, using vectorized NumPy operations.

        Args:
            task (Task): Optimization task.
            pop (numpy.ndarray): Current population.
            fpop (numpy.ndarray): Current population fitness/function values.
            xb (numpy.ndarray): Current best particle.
            fxb (float): Current best particle fitness/function value.
            params (dict): Additional function keyword arguments:
                * pti (numpy.ndarray): Polarization Type Indicator vector.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, float, dict]:
                1. New population.
                2. New population fitness/function values.
                3. New global best position.
                4. New global best positions function/fitness value.
                5. Additional arguments:
                    * pti (numpy.ndarray): Updated Polarization Type Indicator vector.

        See Also:
            * :class:`niapy.algorithms.algorithm.Algorithm.run_iteration`

        """
        pti = params.pop('pti')

        # Handle normalization: convert to/from normalized space if enabled
        if self.normalize_space:
            # Convert current population and best to normalized space
            pop_normalized = self._normalize_from_real(pop, task.lower, task.upper)
            xb_normalized = self._normalize_from_real(xb, task.lower, task.upper)
            # Normalized bounds are [-1, 1]
            lower_norm = np.full(task.dimension, -1.0)
            upper_norm = np.full(task.dimension, 1.0)
            pop_old = pop_normalized.copy()
            xb_work = xb_normalized.copy()
            lower_work = lower_norm
            upper_work = upper_norm
        else:
            pop_old = pop.copy()
            xb_work = xb.copy()
            lower_work = task.lower
            upper_work = task.upper

        # Create boolean masks for each PTI type
        mask_foraging = (pti == 1)
        mask_attack = (pti == 2)
        mask_defense = (pti == 3)
        
        # Initialize new population with old positions
        pop_new = pop_old.copy()

        # Algorithm 2: Update positions based on PTI (Vectorized)

        # PTI == 1: Foraging - Eq. 12 (Langevin Equation)
        # x_new = x_best - (x_i - x_best) + D * (x_r - x_i)
        if np.any(mask_foraging):
            n_foraging = np.sum(mask_foraging)
            # Generate random D values for foraging particles
            D = self.uniform(-1.0, 1.0, size=n_foraging)
            D = D[:, np.newaxis]  # Shape: (n_foraging, 1) for broadcasting
            
            # Generate random indices for x_r (distinct from current index)
            foraging_indices = np.where(mask_foraging)[0]
            r_indices = self.integers(0, self.population_size, size=n_foraging)
            # Ensure r_indices != foraging_indices (vectorized fix)
            if self.population_size > 1:
                equal_mask = (r_indices == foraging_indices)
                r_indices[equal_mask] = (r_indices[equal_mask] + 1) % self.population_size

            x_r = pop_old[r_indices]
            x_i = pop_old[mask_foraging]
            x_best_expanded = np.tile(xb_work, (n_foraging, 1))

            # Apply Eq. 12 vectorized
            pop_new[mask_foraging] = x_best_expanded - (x_i - x_best_expanded) + D * (x_r - x_i)

        # PTI == 2: Attack - Eq. 14
        # x_new = x_best * cos(theta)
        if np.any(mask_attack):
            n_attack = np.sum(mask_attack)
            # Generate random theta values in [pi, 2*pi]
            theta = self.uniform(np.pi, 2 * np.pi, size=n_attack)
            theta = theta[:, np.newaxis]

            x_best_expanded = np.tile(xb_work, (n_attack, 1))
            # Apply Eq. 14 vectorized
            pop_new[mask_attack] = x_best_expanded * np.cos(theta)

        # PTI == 3: Burrow/Defense/Shelter - Eq. 15
        # x_new = x_best ± k * x_best (with k = k_value, default 0.3)
        if np.any(mask_defense):
            n_defense = np.sum(mask_defense)
            # Use fixed k_value (default 0.3) instead of random
            k = np.full((n_defense, 1), self.k_value, dtype=float)

            # Randomly decide between Defense (+) or Shelter (-)
            sign = np.where(self.random(n_defense) < 0.5, 1.0, -1.0)
            sign = sign[:, np.newaxis]

            x_best_expanded = np.tile(xb_work, (n_defense, 1))
            # Apply Eq. 15 vectorized
            pop_new[mask_defense] = x_best_expanded + sign * k * x_best_expanded

        # Boundary check and repair
        if self.use_reflection:
            # Reflection-based boundary handling
            for i in range(self.population_size):
                pop_new[i] = self._reflect_repair(pop_new[i], lower_work, upper_work)
        else:
            # Standard repair
            pop_new = np.apply_along_axis(task.repair, 1, pop_new, rng=self.rng)

        # Convert to real space for evaluation if normalization is enabled
        if self.normalize_space:
            pop_real = np.apply_along_axis(
                lambda z: self._normalize_to_real(z, task.lower, task.upper),
                1, pop_new
            )
            fpop = np.apply_along_axis(task.eval, 1, pop_real)
        else:
            fpop = np.apply_along_axis(task.eval, 1, pop_new)

        # Update global best if better solution found
        best_idx = np.argmin(fpop)
        if fpop[best_idx] < fxb:
            if self.normalize_space:
                # Update normalized best
                xb_normalized = pop_new[best_idx].copy()
                # Also update real-space best for return
                xb = self._normalize_to_real(xb_normalized, task.lower, task.upper)
                xb_work = xb_normalized
            else:
                xb = pop_new[best_idx].copy()
                xb_work = xb.copy()
            fxb = fpop[best_idx]
        
        # Use working population for PTI calculation
        pop_work = pop_new

        # Algorithm 1: Update PTI vector using "Eye Vision" logic (Vectorized)
        # Step 1: Compute LPA and RPA for all particles

        # Calculate dot products: X_old · X_new for all particles
        dot_products = np.sum(pop_old * pop_work, axis=1)

        # Calculate norms: ||X_old|| and ||X_new|| for all particles
        norms_old = np.linalg.norm(pop_old, axis=1)
        norms_new = np.linalg.norm(pop_work, axis=1)

        # Avoid division by zero
        denominator = norms_old * norms_new
        mask_zero = (denominator == 0)

        # Calculate cosine angles
        cos_angles = np.clip(dot_products / np.where(denominator != 0, denominator, 1.0), -1.0, 1.0)

        # Calculate LPA (Left Polarization Angle)
        lpa = np.arccos(cos_angles)
        lpa = np.clip(lpa, 0.0, np.pi)
        lpa[mask_zero] = 0.0
        
        # Calculate RPA (Right Polarization Angle) = rand * pi for all particles
        rpa = self.random(self.population_size) * np.pi

        # Step 2: Determine LPT and RPT based on angle ranges (Eq. 5) - Vectorized
        lpt = self._determine_polarization_type_vectorized(lpa)
        rpt = self._determine_polarization_type_vectorized(rpa)

        # Step 3: Calculate LAD and RAD (Angular differences) according to Eq. 6 - Vectorized
        lad = self._calculate_angular_difference_vectorized(lpa, lpt)
        rad = self._calculate_angular_difference_vectorized(rpa, rpt)

        # Step 4: Update PTI: if LAD < RAD, PTI = LPT, else PTI = RPT - Vectorized
        pti = np.where(lad < rad, lpt, rpt)

        # PTI distribution debugging (optional)
        if self.debug_pti:
            unique, counts = np.unique(pti, return_counts=True)
            pti_dist = dict(zip(unique.tolist(), counts.tolist()))
            logger.debug(f"PTI distribution: {pti_dist}")

        # Return population in the correct space
        # If normalization is enabled, convert back to real space for return
        if self.normalize_space:
            pop_return = np.apply_along_axis(
                lambda z: self._normalize_to_real(z, task.lower, task.upper),
                1, pop_new
            )
        else:
            pop_return = pop_new

        return pop_return, fpop, xb, fxb, {'pti': pti}


# Alias for backward compatibility
MShOA = MantisSearchAlgorithm
