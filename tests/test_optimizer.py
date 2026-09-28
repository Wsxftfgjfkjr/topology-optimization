"""Tests for the SIMP optimizer and the optimality-criteria update."""

import numpy as np
import pytest

from topoopt import fem
from topoopt.optimizer import TopologyOptimizer, optimality_criteria_update

VOLFRAC = 0.4


def _cantilever(nelx, nely, load=1.0):
    """Left edge fully fixed, downward point load mid-way up the right edge."""
    mesh = fem.Mesh(nelx, nely)
    left_edge = mesh.node_index(0, np.arange(nely + 1))
    fixed_dofs = np.concatenate([2 * left_edge, 2 * left_edge + 1]).astype(np.intp)

    loads = np.zeros(mesh.n_dofs)
    loads[2 * mesh.node_index(nelx, nely // 2) + 1] = -load
    return mesh, fixed_dofs, loads


def _optimizer(mesh, fixed_dofs, loads, **overrides):
    settings = dict(volfrac=VOLFRAC, penal=3.0, rmin=1.5, verbose=False)
    settings.update(overrides)
    return TopologyOptimizer(mesh, fixed_dofs, loads, **settings)


# --------------------------------------------------------------------------
# Optimality criteria update
# --------------------------------------------------------------------------


def test_oc_update_respects_the_density_bounds_and_the_move_limit():
    rng = np.random.default_rng(0)
    density = np.full((4, 6), 0.5)
    sensitivities = -1.0 - 0.1 * rng.random((4, 6))

    updated = optimality_criteria_update(sensitivities, density, VOLFRAC, move=0.2)

    assert np.all(updated >= 1e-3)
    assert np.all(updated <= 1.0)
    assert np.all(np.abs(updated - density) <= 0.2 + 1e-12)


def test_oc_update_satisfies_the_volume_constraint():
    rng = np.random.default_rng(1)
    density = np.full((4, 6), 0.5)
    sensitivities = -1.0 - 0.1 * rng.random((4, 6))

    updated = optimality_criteria_update(sensitivities, density, VOLFRAC, move=0.2)

    assert np.isclose(updated.mean(), VOLFRAC, atol=1e-4)


def test_oc_update_leaves_a_uniform_optimal_field_unchanged():
    """A uniform field at the target volume fraction is already stationary."""
    density = np.full((3, 4), VOLFRAC)
    sensitivities = -np.ones((3, 4))

    updated = optimality_criteria_update(sensitivities, density, VOLFRAC, move=0.2)

    assert np.allclose(updated, density, atol=1e-3)


def test_oc_update_moves_density_towards_the_most_sensitive_elements():
    """Elements with a larger compliance derivative must gain material.

    The sensitivities are spread narrowly enough that no element reaches the
    move limit, so the ordering is strict.
    """
    density = np.full((1, 4), 0.5)
    sensitivities = np.array([[-1.5, -1.2, -1.0, -0.9]])

    updated = optimality_criteria_update(sensitivities, density, VOLFRAC, move=0.2)

    assert np.all(np.diff(updated[0]) < 0.0)
    assert np.isclose(updated.mean(), VOLFRAC, atol=1e-4)


# --------------------------------------------------------------------------
# Analysis and sensitivities
# --------------------------------------------------------------------------


def test_analysis_returns_finite_positive_compliance():
    mesh, fixed_dofs, loads = _cantilever(10, 4)
    optimizer = _optimizer(mesh, fixed_dofs, loads)

    displacements, compliance, element_energy = optimizer.analyze()

    assert np.all(np.isfinite(displacements))
    assert np.isfinite(compliance)
    assert compliance > 0.0
    assert element_energy.shape == (mesh.n_elements,)
    assert np.all(element_energy >= 0.0)


def test_solid_cantilever_compliance_matches_slender_beam_theory():
    """Compare the assembled system against an independent physical theory.

    Timoshenko beam theory gives the tip deflection of a slender cantilever as

        delta = P L^3 / (3 E I) + P L / (k G A)

    with shear correction factor k = 5/6 for a rectangular section.  A
    displacement-based finite element model is stiffer than the exact solution,
    so the computed compliance must sit just below this estimate.  This is a
    sanity check on the whole pipeline, not a benchmark against other software.
    """
    nelx, nely = 48, 8
    mesh, fixed_dofs, loads = _cantilever(nelx, nely)
    optimizer = _optimizer(mesh, fixed_dofs, loads)
    optimizer.density[:] = 1.0  # solid material, so the moduli equal E_0

    _, compliance, _ = optimizer.analyze()

    shear_modulus = 1.0 / (2.0 * (1.0 + 0.3))
    second_moment = nely**3 / 12.0
    bending = nelx**3 / (3.0 * second_moment)
    shear = nelx / (5.0 / 6.0 * shear_modulus * nely)
    expected = bending + shear

    assert compliance < expected
    assert compliance > 0.97 * expected


def test_compliance_equals_the_sum_of_the_element_strain_energies():
    """U^T K U must equal sum_e E_e * u_e^T k_0 u_e, the sensitivity kernel."""
    mesh, fixed_dofs, loads = _cantilever(10, 4)
    optimizer = _optimizer(mesh, fixed_dofs, loads)
    optimizer.density[:] = 0.37

    _, compliance, element_energy = optimizer.analyze()
    total = float(np.sum(optimizer.element_moduli() * element_energy))

    assert np.isclose(compliance, total, rtol=1e-9)


def test_compliance_sensitivity_is_non_positive():
    """Adding material can never increase compliance for a fixed load."""
    mesh, fixed_dofs, loads = _cantilever(10, 4)
    optimizer = _optimizer(mesh, fixed_dofs, loads)
    optimizer.density[:] = 0.5

    _, _, element_energy = optimizer.analyze()
    sensitivities = optimizer.compliance_sensitivity(element_energy)
    filtered = optimizer.apply_sensitivity_filter(sensitivities)

    assert np.all(sensitivities <= 0.0)
    assert np.any(sensitivities < 0.0)
    assert np.all(filtered <= 0.0)


# --------------------------------------------------------------------------
# End-to-end smoke test
# --------------------------------------------------------------------------


def test_small_cantilever_optimization_returns_a_finite_improved_design():
    mesh, fixed_dofs, loads = _cantilever(12, 4)
    optimizer = _optimizer(mesh, fixed_dofs, loads, max_iterations=30, tolerance=1e-3)
    initial_density = optimizer.density.copy()
    initial_compliance = optimizer.analyze()[1]

    result = optimizer.run()

    assert result.iterations == len(result.history) > 0
    assert np.isfinite(result.history[-1].compliance)

    # The design must have changed, and the reported history must be monotone
    # in the sense that the final design is stiffer than the uniform start.
    assert not np.allclose(result.density, initial_density)
    assert result.history[-1].compliance < initial_compliance

    # Densities stay in range and the volume constraint is respected.
    assert np.all(np.isfinite(result.density))
    assert np.all(result.density > 0.0)
    assert np.all(result.density <= 1.0)
    assert np.isclose(result.density.mean(), VOLFRAC, atol=0.01)

    # The converged density field solves cleanly.
    displacements, compliance, _ = optimizer.analyze()
    assert np.all(np.isfinite(displacements))
    assert np.isfinite(compliance) and compliance > 0.0


def test_optimizer_rejects_a_volume_fraction_outside_the_unit_interval():
    mesh, fixed_dofs, loads = _cantilever(4, 2)

    with pytest.raises(ValueError):
        _optimizer(mesh, fixed_dofs, loads, volfrac=1.5)
