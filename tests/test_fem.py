"""Tests for the finite element building blocks."""

import numpy as np
import pytest
import scipy.sparse as sparse

from topoopt import fem

YOUNGS = 1.0
POISSON = 0.3


def test_plane_stress_matrix_is_symmetric_positive_definite():
    D = fem.plane_stress_matrix(YOUNGS, POISSON)

    assert D.shape == (3, 3)
    assert np.allclose(D, D.T)
    assert np.all(np.linalg.eigvalsh(D) > 0.0)


def test_plane_stress_matrix_matches_uniaxial_stress():
    """sigma_xx = E / (1 - nu^2) * eps_xx when eps_yy = gamma_xy = 0."""
    D = fem.plane_stress_matrix(YOUNGS, POISSON)
    stress = D @ np.array([1.0, 0.0, 0.0])
    scale = YOUNGS / (1.0 - POISSON**2)

    assert np.allclose(stress, [scale, POISSON * scale, 0.0])


def test_element_stiffness_matrix_shape_and_symmetry():
    stiffness = fem.element_stiffness_matrix(YOUNGS, POISSON)

    assert stiffness.shape == (8, 8)
    assert np.allclose(stiffness, stiffness.T)


@pytest.mark.parametrize("direction", [0, 1])
def test_element_stiffness_matrix_has_no_rigid_translation_force(direction):
    """A rigid translation of the element must produce zero nodal forces."""
    stiffness = fem.element_stiffness_matrix(YOUNGS, POISSON)
    translation = np.zeros(8)
    translation[direction::2] = 1.0

    assert np.allclose(stiffness @ translation, 0.0, atol=1e-13)


def test_element_stiffness_matrix_is_positive_semidefinite_with_three_null_modes():
    """Two translations and one rotation, and nothing else, cost no energy."""
    stiffness = fem.element_stiffness_matrix(YOUNGS, POISSON)
    eigenvalues = np.linalg.eigvalsh(stiffness)

    assert eigenvalues.min() > -1e-12
    assert np.count_nonzero(np.abs(eigenvalues) < 1e-12) == 3


@pytest.mark.parametrize("hx,hy", [(1.0, 1.0), (2.0, 1.0), (1.0, 3.0)])
def test_element_stiffness_matrix_reproduces_uniaxial_strain_energy(hx, hy):
    """For u_x = eps * x the strain energy is 0.5 * E / (1 - nu^2) * eps^2 * V."""
    stiffness = fem.element_stiffness_matrix(YOUNGS, POISSON, hx=hx, hy=hy)

    # Node order is lower left, lower right, upper right, upper left, so the
    # x-displacement degrees of freedom sit at indices 0, 2, 4 and 6.  With
    # unit strain the right-hand nodes sit at x = hx.
    displacement = np.zeros(8)
    displacement[2] = hx  # lower right
    displacement[4] = hx  # upper right

    energy = displacement @ stiffness @ displacement
    assert np.isclose(energy, YOUNGS * hx * hy / (1.0 - POISSON**2))


@pytest.mark.parametrize("hx,hy", [(1.0, 1.0), (2.0, 1.0), (1.0, 3.0)])
def test_element_stiffness_matrix_reproduces_shear_strain_energy(hx, hy):
    """For u_x = gamma * y the strain energy is 0.5 * G * gamma^2 * V."""
    stiffness = fem.element_stiffness_matrix(YOUNGS, POISSON, hx=hx, hy=hy)

    # With unit shear strain the top nodes sit at y = hy.
    displacement = np.zeros(8)
    displacement[4] = hy  # upper right
    displacement[6] = hy  # upper left

    energy = displacement @ stiffness @ displacement
    shear_modulus = YOUNGS / (2.0 * (1.0 + POISSON))
    assert np.isclose(energy, shear_modulus * hx * hy)


def test_square_element_stiffness_is_independent_of_the_element_size():
    """B scales like 1/h and the volume like h^2, so a square element's
    stiffness does not depend on its edge length."""
    reference = fem.element_stiffness_matrix(YOUNGS, POISSON, hx=1.0, hy=1.0)

    for size in (0.25, 2.0, 7.5):
        scaled = fem.element_stiffness_matrix(YOUNGS, POISSON, hx=size, hy=size)
        assert np.allclose(scaled, reference)


@pytest.mark.parametrize("nelx,nely", [(0, 3), (3, 0), (-1, 2)])
def test_mesh_rejects_degenerate_sizes(nelx, nely):
    with pytest.raises(ValueError):
        fem.Mesh(nelx, nely)


def test_mesh_sizes():
    mesh = fem.Mesh(3, 2)

    assert (mesh.nelx, mesh.nely) == (3, 2)
    assert mesh.n_elements == 6
    assert mesh.n_nodes == 12
    assert mesh.n_dofs == 24
    assert mesh.element_dofs.shape == (6, 8)
    assert mesh.node_coordinates.shape == (12, 2)


def test_mesh_node_numbering_is_row_major_with_x_fastest():
    mesh = fem.Mesh(2, 1)

    assert mesh.node_index(0, 0) == 0
    assert mesh.node_index(1, 0) == 1
    assert mesh.node_index(2, 0) == 2
    assert mesh.node_index(0, 1) == 3
    assert mesh.node_index(2, 1) == 5


def test_mesh_element_dofs_follow_the_documented_node_order():
    """Element DOFs are [u, v] per node, counter-clockwise from lower left."""
    mesh = fem.Mesh(2, 1)

    # Left element: nodes (0,0), (1,0), (1,1), (0,1) -> global 0, 1, 4, 3.
    assert mesh.element_dofs[0].tolist() == [0, 1, 2, 3, 8, 9, 6, 7]
    # Right element: nodes (1,0), (2,0), (2,1), (1,1) -> global 1, 2, 5, 4.
    assert mesh.element_dofs[1].tolist() == [2, 3, 4, 5, 10, 11, 8, 9]


def test_mesh_element_dofs_are_distinct_and_cover_every_dof():
    mesh = fem.Mesh(4, 3)

    for row in mesh.element_dofs:
        assert len(set(row.tolist())) == 8
    assert np.array_equal(np.unique(mesh.element_dofs), np.arange(mesh.n_dofs))


def test_global_stiffness_is_symmetric_and_annihilates_rigid_translation():
    mesh = fem.Mesh(3, 2)
    element_stiffness = fem.element_stiffness_matrix(YOUNGS, POISSON)
    stiffness = fem.assemble_stiffness_matrix(
        mesh, element_stiffness, np.ones(mesh.n_elements)
    )

    assert abs(stiffness - stiffness.T).max() < 1e-14

    for direction in (0, 1):
        translation = np.zeros(mesh.n_dofs)
        translation[direction::2] = 1.0
        assert np.allclose(stiffness @ translation, 0.0, atol=1e-13)


def test_global_stiffness_scales_with_the_element_moduli():
    mesh = fem.Mesh(3, 2)
    element_stiffness = fem.element_stiffness_matrix(YOUNGS, POISSON)

    unit = fem.assemble_stiffness_matrix(
        mesh, element_stiffness, np.ones(mesh.n_elements)
    )
    doubled = fem.assemble_stiffness_matrix(
        mesh, element_stiffness, 2.0 * np.ones(mesh.n_elements)
    )

    assert np.allclose((doubled - 2.0 * unit).toarray(), 0.0)


# --------------------------------------------------------------------------
# Cached assembly structure
# --------------------------------------------------------------------------


def _reference_assembly(mesh, element_stiffness, element_moduli):
    """The reference COO scatter, spelled out independently of the cached plan.

    This is ``v0.1-reference``'s ``assemble_stiffness_matrix`` verbatim: build all
    64 * n_elements (row, column) pairs and let ``coo_matrix`` sum the duplicates
    while converting to CSC.  Keeping it here means the cached path is checked
    against a restatement of the reference rather than against itself.
    """
    element_dofs = mesh.element_dofs
    rows = np.repeat(element_dofs, 8, axis=1).reshape(-1)
    columns = np.tile(element_dofs, (1, 8)).reshape(-1)
    values = (
        element_stiffness.reshape(-1)[None, :] * element_moduli[:, None]
    ).reshape(-1)

    return sparse.coo_matrix(
        (values, (rows, columns)), shape=(mesh.n_dofs, mesh.n_dofs)
    ).tocsc()


def _moduli(mesh):
    """A SIMP-shaped but otherwise arbitrary positive field."""
    return 1e-9 + np.linspace(0.01, 1.0, mesh.n_elements) ** 3 * (1.0 - 1e-9)


@pytest.mark.parametrize("nelx,nely", [(3, 2), (6, 5), (10, 4)])
def test_cached_assembly_reproduces_the_coo_reference_bit_for_bit(nelx, nely):
    """The cached structure must not change the assembled numbers at all.

    Only the summation *order* is at stake: the cached path accumulates entries
    into their CSC slot in COO order, which is the order SciPy's COO -> CSC
    conversion sums them in, so even the last bit must survive.
    """
    mesh = fem.Mesh(nelx, nely)
    element_stiffness = fem.element_stiffness_matrix(YOUNGS, POISSON)

    assembled = fem.assemble_stiffness_matrix(mesh, element_stiffness, _moduli(mesh))
    reference = _reference_assembly(mesh, element_stiffness, _moduli(mesh))

    assert assembled.format == "csc"
    assert np.array_equal(assembled.indptr, reference.indptr)
    assert np.array_equal(assembled.indices, reference.indices)
    assert np.array_equal(assembled.data, reference.data)


@pytest.mark.parametrize("nelx,nely", [(6, 5), (12, 4)])
def test_cached_assembly_is_bit_exact_across_unrelated_moduli(nelx, nely):
    """The same cached plan must be valid for every value field, not just one."""
    mesh = fem.Mesh(nelx, nely)
    element_stiffness = fem.element_stiffness_matrix(YOUNGS, POISSON)
    rng = np.random.default_rng(7)

    fields = [
        np.ones(mesh.n_elements),
        _moduli(mesh),
        rng.random(mesh.n_elements) + 1e-3,
    ]

    for moduli in fields:
        assembled = fem.assemble_stiffness_matrix(mesh, element_stiffness, moduli)
        reference = _reference_assembly(mesh, element_stiffness, moduli)
        assert np.array_equal(assembled.data, reference.data)


def test_assembly_plan_is_cached_per_mesh():
    mesh = fem.Mesh(4, 3)

    plan = mesh.assembly_plan

    assert mesh.assembly_plan is plan
    # Each mesh owns its own plan: the structure is derived from its element_dofs.
    assert fem.Mesh(4, 3).assembly_plan is not plan


def test_assembly_plan_scatter_map_covers_every_entry_exactly_once():
    mesh = fem.Mesh(4, 3)
    plan = mesh.assembly_plan

    assert plan.slot.shape == (64 * mesh.n_elements,)
    assert plan.indptr.shape == (mesh.n_dofs + 1,)
    assert plan.indptr[0] == 0
    assert plan.indptr[-1] == plan.nnz
    assert plan.indices.shape == (plan.nnz,)

    # Every COO entry lands in a slot, and every slot receives at least one
    # entry -- an unreachable slot would be a structural entry that is always
    # zero, which the assembled matrix does not have.
    assert plan.slot.min() == 0
    assert plan.slot.max() == plan.nnz - 1
    assert np.bincount(plan.slot, minlength=plan.nnz).min() > 0


def test_assembly_plan_structure_is_independent_of_the_element_stiffness():
    """A different constitutive matrix changes values, never the pattern."""
    mesh = fem.Mesh(4, 3)
    plan = mesh.assembly_plan
    before = (plan.indptr.copy(), plan.indices.copy(), plan.slot.copy())

    other_stiffness = fem.element_stiffness_matrix(2.5, 0.15)
    assembled = fem.assemble_stiffness_matrix(mesh, other_stiffness, _moduli(mesh))

    assert mesh.assembly_plan is plan
    assert np.array_equal(plan.indptr, before[0])
    assert np.array_equal(plan.indices, before[1])
    assert np.array_equal(plan.slot, before[2])
    assert assembled.nnz == plan.nnz


def test_free_degrees_of_freedom_complement_the_restrained_ones():
    free = fem.free_degrees_of_freedom(6, np.array([0, 1, 5]))

    assert free.tolist() == [2, 3, 4]


def _uniform_stress_loads(mesh, sigma_xx, sigma_yy):
    """Consistent nodal loads for a uniform stress state.

    A uniform traction on a unit-length element edge is carried half by each of
    the edge's two nodes, so a node shared by two segments collects one full
    traction.  Loads are applied on the right, top and bottom edges; the left
    edge is restrained by displacement boundary conditions instead.
    """
    loads = np.zeros(mesh.n_dofs)

    # Right edge, outward normal (1, 0): traction (sigma_xx, 0).
    segments = np.arange(mesh.nely)
    nodes = np.concatenate(
        [mesh.node_index(mesh.nelx, segments), mesh.node_index(mesh.nelx, segments + 1)]
    )
    np.add.at(loads, 2 * nodes, 0.5 * sigma_xx)

    # Top edge, outward normal (0, 1): traction (0, sigma_yy).
    segments = np.arange(mesh.nelx)
    nodes = np.concatenate(
        [mesh.node_index(segments, mesh.nely), mesh.node_index(segments + 1, mesh.nely)]
    )
    np.add.at(loads, 2 * nodes + 1, 0.5 * sigma_yy)

    # Bottom edge, outward normal (0, -1): traction (0, -sigma_yy).
    nodes = np.concatenate(
        [mesh.node_index(segments, 0), mesh.node_index(segments + 1, 0)]
    )
    np.add.at(loads, 2 * nodes + 1, -0.5 * sigma_yy)

    return loads


def test_constant_strain_patch_test():
    """A linear displacement field is reproduced exactly on a distorted-free patch.

    Uniform tractions are imposed on three edges and the left edge is fixed.
    The exact solution is u_x = eps * x, u_y = 0, which lies in the finite
    element space, so the computed nodal displacements must match it to machine
    precision.  This exercises the constitutive matrix, the strain-displacement
    matrix, the assembly and the solve together.
    """
    mesh = fem.Mesh(4, 3)
    element_stiffness = fem.element_stiffness_matrix(YOUNGS, POISSON)
    stiffness = fem.assemble_stiffness_matrix(
        mesh, element_stiffness, np.ones(mesh.n_elements)
    )

    strain_xx = 1e-3
    scale = YOUNGS / (1.0 - POISSON**2)
    loads = _uniform_stress_loads(mesh, strain_xx * scale, POISSON * strain_xx * scale)

    left_edge = mesh.node_index(0, np.arange(mesh.nely + 1))
    fixed_dofs = np.concatenate([2 * left_edge, 2 * left_edge + 1])
    free_dofs = fem.free_degrees_of_freedom(mesh.n_dofs, fixed_dofs)

    displacements = fem.solve_displacements(stiffness, loads, free_dofs)

    expected = np.zeros(mesh.n_dofs)
    expected[0::2] = strain_xx * mesh.node_coordinates[:, 0]
    assert np.allclose(displacements, expected, atol=1e-12)
