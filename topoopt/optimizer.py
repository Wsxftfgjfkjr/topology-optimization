"""SIMP minimum-compliance topology optimization with an optimality-criteria update.

The problem solved is::

    minimize    c(x) = U^T K(x) U
    subject to  K(x) U = F
                sum(x_e) / n_elements <= volfrac
                x_min <= x_e <= 1

where the stiffness of element ``e`` follows the SIMP interpolation::

    E_e(x_e) = E_min + x_e^penal * (E_0 - E_min)

Element volumes are one on this mesh, so the volume constraint bounds the mean
density.  The compliance derivative with respect to the density is analytical
(for a fixed load the problem is self-adjoint)::

    dc/dx_e = -penal * x_e^(penal - 1) * (E_0 - E_min) * u_e^T k_0 u_e

and the design update is the corresponding optimality-criteria fixed point.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from . import fem
from .filter import build_filter_kernel, filter_sensitivities

# Dimensionless material defaults, as used by the standard cantilever benchmark.
DEFAULT_YOUNGS_MODULUS = 1.0
DEFAULT_POISSON_RATIO = 0.3
DEFAULT_VOID_MODULUS_RATIO = 1e-9
DEFAULT_MIN_DENSITY = 1e-3
DEFAULT_MOVE_LIMIT = 0.2
DEFAULT_MAX_ITERATIONS = 200
DEFAULT_CONVERGENCE_TOLERANCE = 1e-2

# Relative tolerance on the volume-constraint multiplier found by bisection.
# Small enough that the volume constraint holds to roughly the same relative
# accuracy, and cheap to reach.
BISECTION_TOLERANCE = 1e-6


@dataclass
class IterationRecord:
    """Result of a single optimization iteration.

    ``compliance`` and ``volume_fraction`` both describe the design the finite
    element analysis was run on, i.e. the state *before* the design update of
    this iteration.
    """

    iteration: int
    compliance: float
    volume_fraction: float
    change: float
    time_s: float


@dataclass
class OptimizationResult:
    """Outcome of :meth:`TopologyOptimizer.run`."""

    density: np.ndarray
    history: list[IterationRecord]
    converged: bool
    iterations: int


def _optimality_criteria_trial(sensitivities, density, multiplier, move, min_density):
    """Density field produced by one multiplier value, before clipping."""
    ratio = np.maximum(-sensitivities / multiplier, 0.0)
    updated = density * np.sqrt(ratio)
    return np.clip(np.clip(updated, density - move, density + move), min_density, 1.0)


def optimality_criteria_update(
    sensitivities,
    density,
    volfrac,
    move=DEFAULT_MOVE_LIMIT,
    min_density=DEFAULT_MIN_DENSITY,
    tolerance=BISECTION_TOLERANCE,
):
    """One optimality-criteria design update, enforcing the volume constraint.

    Stationarity of the Lagrangian of the minimum-compliance problem gives the
    element-wise optimality condition ``-dc_e / (lambda * dv_e) = 1``, so the
    update is ``x_e <- x_e * sqrt(-dc_e / lambda)`` with ``lambda`` the Lagrange
    multiplier of the volume constraint.  The move limit and the density bounds
    are applied afterwards, and ``lambda`` is found by bisecting the (monotone)
    map from ``lambda`` to the resulting volume until the constraint is met.

    Parameters
    ----------
    sensitivities : ndarray
        Compliance derivatives ``dc/dx_e``; non-positive for this problem.
    density : ndarray
        Current densities, same shape as ``sensitivities``.
    volfrac : float
        Requested volume fraction of the design domain.

    Returns
    -------
    ndarray
        Updated densities, same shape as the input.
    """
    x = np.asarray(density, dtype=float)
    dc = np.asarray(sensitivities, dtype=float)
    if x.shape != dc.shape:
        raise ValueError("density and sensitivities must have the same shape")

    # The multiplier is bounded below by zero and shrinks the design as it
    # grows; the initial bracket spans any reasonable scaling.
    lower, upper = 1e-9, 1e9
    target = volfrac * x.size

    while (upper - lower) / (lower + upper) > tolerance:
        multiplier = 0.5 * (lower + upper)
        trial = _optimality_criteria_trial(dc, x, multiplier, move, min_density)
        if trial.sum() > target:
            lower = multiplier
        else:
            upper = multiplier

    # Recompute at the midpoint of the final bracket so that the returned field
    # is guaranteed to correspond to a multiplier inside the tolerance.
    return _optimality_criteria_trial(dc, x, 0.5 * (lower + upper), move, min_density)


class TopologyOptimizer:
    """Minimum-compliance topology optimization on a structured Q4 mesh.

    The mesh, the restrained degrees of freedom and the load vector fully define
    the problem; everything else is an optimization or material parameter.

    Parameters
    ----------
    mesh : topoopt.fem.Mesh
    fixed_dofs : array_like of int
        Restrained degrees of freedom, held at zero displacement.
    loads : ndarray, shape (n_dofs,)
        Nodal load vector.
    volfrac : float
        Maximum volume fraction of the design domain.
    penal : float
        SIMP penalty exponent.
    rmin : float
        Sensitivity filter radius in element units.
    youngs_modulus, poisson_ratio : float
        Solid material properties.
    void_modulus_ratio : float
        ``E_min / E_0``, keeping the stiffness matrix non-singular.
    min_density : float
        Lower bound on the densities.
    move : float
        Maximum density change per iteration.
    max_iterations : int
    tolerance : float
        Convergence tolerance on the maximum density change.
    verbose : bool
        Print one line per iteration.
    """

    def __init__(
        self,
        mesh: fem.Mesh,
        fixed_dofs,
        loads,
        volfrac: float = 0.4,
        penal: float = 3.0,
        rmin: float = 1.5,
        youngs_modulus: float = DEFAULT_YOUNGS_MODULUS,
        poisson_ratio: float = DEFAULT_POISSON_RATIO,
        void_modulus_ratio: float = DEFAULT_VOID_MODULUS_RATIO,
        min_density: float = DEFAULT_MIN_DENSITY,
        move: float = DEFAULT_MOVE_LIMIT,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        tolerance: float = DEFAULT_CONVERGENCE_TOLERANCE,
        verbose: bool = True,
    ):
        if not 0.0 < volfrac <= 1.0:
            raise ValueError("volfrac must lie in (0, 1]")

        self.mesh = mesh
        self.fixed_dofs = np.asarray(fixed_dofs, dtype=np.intp)
        self.loads = np.asarray(loads, dtype=float)
        if self.loads.shape != (mesh.n_dofs,):
            raise ValueError(f"loads must have shape ({mesh.n_dofs},)")

        self.volfrac = float(volfrac)
        self.penal = float(penal)
        self.rmin = float(rmin)
        self.youngs_modulus = float(youngs_modulus)
        self.void_modulus = float(void_modulus_ratio) * float(youngs_modulus)
        self.poisson_ratio = float(poisson_ratio)
        self.min_density = float(min_density)
        self.move = float(move)
        self.max_iterations = int(max_iterations)
        self.tolerance = float(tolerance)
        self.verbose = bool(verbose)

        # Everything below is constant across iterations and is built once.
        self.element_stiffness = fem.element_stiffness_matrix(
            self.youngs_modulus, self.poisson_ratio
        )
        self.filter_kernel, self.filter_row_sums = build_filter_kernel(
            mesh.nelx, mesh.nely, self.rmin
        )
        self.free_dofs = fem.free_degrees_of_freedom(mesh.n_dofs, self.fixed_dofs)
        self.density = np.full((mesh.nely, mesh.nelx), self.volfrac)

    def element_moduli(self):
        """SIMP interpolation of the element Young's moduli."""
        density = self.density.reshape(-1)
        return self.void_modulus + density**self.penal * (
            self.youngs_modulus - self.void_modulus
        )

    def analyze(self):
        """Assemble, solve, and evaluate the compliance and element energies.

        Returns
        -------
        displacements : ndarray, shape (n_dofs,)
        compliance : float
            ``U^T K U``, equal to the sum of the element strain energies.
        element_energy : ndarray, shape (n_elements,)
            ``u_e^T k_0 u_e`` per element, with ``k_0`` the unit-modulus
            element stiffness matrix.  This is the sensitivity kernel.
        """
        stiffness = fem.assemble_stiffness_matrix(
            self.mesh, self.element_stiffness, self.element_moduli()
        )
        displacements = fem.solve_displacements(stiffness, self.loads, self.free_dofs)
        compliance = float(displacements @ (stiffness @ displacements))

        element_displacements = displacements[self.mesh.element_dofs]
        element_energy = np.einsum(
            "ei,ij,ej->e",
            element_displacements,
            self.element_stiffness,
            element_displacements,
        )
        return displacements, compliance, element_energy

    def compliance_sensitivity(self, element_energy):
        """Analytical compliance derivative with respect to the density."""
        density = self.density.reshape(-1)
        derivative = (
            -self.penal
            * density ** (self.penal - 1.0)
            * (self.youngs_modulus - self.void_modulus)
            * element_energy
        )
        return derivative.reshape(self.density.shape)

    def apply_sensitivity_filter(self, sensitivities):
        """Smooth the sensitivities over the cone neighbourhood."""
        return filter_sensitivities(
            sensitivities,
            self.density,
            self.filter_kernel,
            self.filter_row_sums,
            self.min_density,
        )

    def run(self) -> OptimizationResult:
        """Run the optimization loop until convergence or the iteration cap."""
        history: list[IterationRecord] = []
        converged = False
        iteration = 0

        if self.verbose:
            print(f"{'iter':>5} {'compliance':>15} {'volfrac':>9} {'change':>10} {'time [s]':>9}")
            print("-" * 52)

        for iteration in range(1, self.max_iterations + 1):
            start = time.perf_counter()

            _, compliance, element_energy = self.analyze()
            sensitivities = self.compliance_sensitivity(element_energy)
            sensitivities = self.apply_sensitivity_filter(sensitivities)
            updated = optimality_criteria_update(
                sensitivities, self.density, self.volfrac, self.move, self.min_density
            )

            change = float(np.max(np.abs(updated - self.density)))
            record = IterationRecord(
                iteration=iteration,
                compliance=compliance,
                volume_fraction=float(self.density.mean()),
                change=change,
                time_s=time.perf_counter() - start,
            )
            history.append(record)
            self.density = updated

            if self.verbose:
                print(
                    f"{record.iteration:5d} {record.compliance:15.6e} "
                    f"{record.volume_fraction:9.4f} {record.change:10.3e} "
                    f"{record.time_s:9.3f}"
                )

            if change < self.tolerance:
                converged = True
                break

        if self.verbose:
            status = "converged" if converged else "stopped at the iteration cap"
            print("-" * 52)
            print(
                f"{status} after {iteration} iterations: "
                f"compliance = {history[-1].compliance:.6e}, "
                f"volume fraction = {self.density.mean():.4f}"
            )

        return OptimizationResult(
            density=self.density,
            history=history,
            converged=converged,
            iterations=iteration,
        )
