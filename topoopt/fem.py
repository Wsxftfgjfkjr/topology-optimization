"""Finite element analysis on a structured mesh of plane-stress Q4 elements.

The mesh is a regular grid of ``nelx`` x ``nely`` four-node bilinear
quadrilaterals of unit edge length, so the design domain is the rectangle
``[0, nelx] x [0, nely]``.  Every node carries two degrees of freedom: the
horizontal displacement ``u`` and the vertical displacement ``v``.

Numbering convention used throughout the package
------------------------------------------------
Nodes are numbered row-major with the x index varying fastest::

    node(i, j) = j * (nelx + 1) + i        i in [0, nelx], j in [0, nely]

Elements use the same layout, so the element array reshapes to ``(nely, nelx)``
and is indexed as ``[j, i]``::

    element(i, j) = j * nelx + i           i in [0, nelx), j in [0, nely)

The four nodes of an element are ordered counter-clockwise starting from the
lower left corner, which matches the reference-square coordinates
``(-1,-1), (1,-1), (1,1), (-1,1)``.  The element degree-of-freedom vector is
therefore::

    [u1, v1, u2, v2, u3, v3, u4, v4]

so even positions are horizontal and odd positions vertical displacements.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as sparse_linalg

# 2 x 2 Gauss-Legendre quadrature on the reference square [-1, 1]^2.  The rule
# is exact here: for a rectangular element the stiffness integrand is at most
# quadratic in each natural coordinate, and the rule integrates degree three.
GAUSS_POINTS = np.array([-1.0, 1.0]) / np.sqrt(3.0)
GAUSS_WEIGHTS = np.array([1.0, 1.0])


def plane_stress_matrix(youngs_modulus, poisson_ratio):
    """Isotropic plane-stress constitutive matrix ``D`` (3 x 3).

    The strain vector is ordered ``(eps_xx, eps_yy, gamma_xy)``.
    """
    factor = youngs_modulus / (1.0 - poisson_ratio**2)
    return factor * np.array(
        [
            [1.0, poisson_ratio, 0.0],
            [poisson_ratio, 1.0, 0.0],
            [0.0, 0.0, 0.5 * (1.0 - poisson_ratio)],
        ]
    )


def shape_function_derivatives(xi, eta):
    """Natural-coordinate derivatives of the four bilinear shape functions.

    Returns ``(dN_dxi, dN_deta)``, each an array of length four ordered like the
    element nodes (lower left, lower right, upper right, upper left).
    """
    dN_dxi = 0.25 * np.array([-(1.0 - eta), (1.0 - eta), (1.0 + eta), -(1.0 + eta)])
    dN_deta = 0.25 * np.array([-(1.0 - xi), -(1.0 + xi), (1.0 + xi), (1.0 - xi)])
    return dN_dxi, dN_deta


def strain_displacement_matrix(xi, eta, hx=1.0, hy=1.0):
    """Strain-displacement matrix ``B`` (3 x 8) at a quadrature point.

    The element is a rectangle of size ``hx`` x ``hy``, so the Jacobian of the
    isoparametric map is diagonal and constant and the chain rule reduces to
    ``d/dx = (2 / hx) d/dxi`` and ``d/dy = (2 / hy) d/deta``.
    """
    dN_dxi, dN_deta = shape_function_derivatives(xi, eta)
    dN_dx = dN_dxi * (2.0 / hx)
    dN_dy = dN_deta * (2.0 / hy)

    B = np.zeros((3, 8))
    B[0, 0::2] = dN_dx  # eps_xx     = du/dx
    B[1, 1::2] = dN_dy  # eps_yy     = dv/dy
    B[2, 0::2] = dN_dy  # gamma_xy   = du/dy + dv/dx
    B[2, 1::2] = dN_dx
    return B


def element_stiffness_matrix(youngs_modulus, poisson_ratio, hx=1.0, hy=1.0):
    """Q4 plane-stress element stiffness matrix (8 x 8).

    Computes ``integral of B^T D B`` over the element with 2 x 2 Gauss
    quadrature.  The result is symmetric, positive semi-definite and has three
    zero eigenvalues (two translations and one rotation).
    """
    D = plane_stress_matrix(youngs_modulus, poisson_ratio)
    det_jacobian = 0.25 * hx * hy

    stiffness = np.zeros((8, 8))
    for xi, weight_xi in zip(GAUSS_POINTS, GAUSS_WEIGHTS):
        for eta, weight_eta in zip(GAUSS_POINTS, GAUSS_WEIGHTS):
            B = strain_displacement_matrix(xi, eta, hx, hy)
            stiffness += weight_xi * weight_eta * det_jacobian * (B.T @ D @ B)
    return stiffness


class Mesh:
    """Structured rectangular mesh of ``nelx`` x ``nely`` Q4 elements.

    Elements have unit edge length, so the domain is ``[0, nelx] x [0, nely]``
    and element volumes are one.  See the module docstring for the numbering
    convention shared by every array produced here.
    """

    def __init__(self, nelx: int, nely: int):
        if nelx < 1 or nely < 1:
            raise ValueError("nelx and nely must be positive integers")

        self.nelx = int(nelx)
        self.nely = int(nely)
        self.n_elements = self.nelx * self.nely
        self.n_nodes = (self.nelx + 1) * (self.nely + 1)
        self.n_dofs = 2 * self.n_nodes

        self.element_dofs = self._build_element_dofs()
        self.node_coordinates = self._build_node_coordinates()

        # Built on first use; see Mesh.assembly_plan.
        self._assembly_plan = None

    @property
    def assembly_plan(self) -> "StiffnessAssemblyPlan":
        """Sparsity structure for assembling this mesh's stiffness matrix.

        Lazily built and cached, because it is derived from ``element_dofs``
        alone -- which this class fixes at construction and never mutates -- and
        is therefore reusable for every assembly on this mesh.  Caching it on the
        mesh makes that validity explicit: the cache lives exactly as long as the
        connectivity it was derived from.
        """
        plan = self._assembly_plan
        if plan is None:
            plan = self._assembly_plan = StiffnessAssemblyPlan(self)
        return plan

    def __repr__(self) -> str:
        return f"Mesh(nelx={self.nelx}, nely={self.nely})"

    def node_index(self, i, j):
        """Global index of grid node ``(i, j)``; accepts scalars or arrays."""
        i = np.asarray(i, dtype=np.intp)
        j = np.asarray(j, dtype=np.intp)
        return j * (self.nelx + 1) + i

    def _build_element_dofs(self):
        # np.meshgrid defaults to "xy" indexing, so i_index and j_index have
        # shape (nely, nelx) and element (i, j) sits at [j, i].  Ravel order is
        # therefore j * nelx + i, matching the layout of the density array.
        i_index, j_index = np.meshgrid(np.arange(self.nelx), np.arange(self.nely))
        corners = np.stack(
            [
                self.node_index(i_index, j_index),  # lower left
                self.node_index(i_index + 1, j_index),  # lower right
                self.node_index(i_index + 1, j_index + 1),  # upper right
                self.node_index(i_index, j_index + 1),  # upper left
            ],
            axis=-1,
        )

        element_dofs = np.empty(corners.shape[:-1] + (8,), dtype=np.intp)
        element_dofs[..., 0::2] = 2 * corners
        element_dofs[..., 1::2] = 2 * corners + 1
        return element_dofs.reshape(self.n_elements, 8)

    def _build_node_coordinates(self):
        i_index, j_index = np.meshgrid(
            np.arange(self.nelx + 1), np.arange(self.nely + 1)
        )
        return np.stack([i_index.ravel(), j_index.ravel()], axis=-1).astype(float)


class StiffnessAssemblyPlan:
    """Precomputed sparsity structure for assembling a mesh's stiffness matrix.

    Why this is cached
    ------------------
    ``Mesh.element_dofs`` is built once in :meth:`Mesh.__init__` and never
    mutated, so the mesh alone fixes every structural quantity of the assembly:

    * which ``(row, column)`` degree-of-freedom pair each of the 64 entries of an
      element matrix is scattered to,
    * the CSC sparsity structure ``(indptr, indices)`` of the assembled matrix,
    * ``slot``, the CSC data slot that each of those entries accumulates into.

    In a topology-optimization loop the mesh is fixed and only the element moduli
    change, so rebuilding those three arrays every iteration recomputes the same
    answer.  Precomputing them turns each assembly into one scatter-add and
    removes the per-iteration COO index construction and SciPy's COO -> CSC sort
    from the hot path.

    Nothing here depends on ``element_stiffness``: it scales values, never the
    pattern, so the plan stays valid if the constitutive matrix changes too.

    Construction works on a few temporaries the size of the COO entry list
    (``64 * n_elements``).  They are released as soon as they are dead rather
    than left to accumulate: this is a one-off cost per mesh, but resident-set
    size is a high-water mark, so the peak it reaches is paid for by the rest of
    the run.

    Parameters
    ----------
    mesh : Mesh
        The mesh whose connectivity is captured.  The plan stays valid only while
        ``mesh.element_dofs`` and ``mesh.n_dofs`` are unchanged.
    """

    def __init__(self, mesh: Mesh):
        n_dofs = mesh.n_dofs
        element_dofs = mesh.element_dofs

        # The (row, column) pair of every one of the 64 entries of every element
        # matrix.  ``np.repeat`` walks the eight element DOFs slowly (row index)
        # and ``np.tile`` walks them quickly (column index), which lines up with
        # the C-order ravel of the element matrix.
        rows = np.repeat(element_dofs, 8, axis=1).reshape(-1)
        columns = np.tile(element_dofs, (1, 8)).reshape(-1)
        n_entries = rows.size

        # Packing each pair into one integer makes "sort by (column, row)" an
        # integer sort, which NumPy does with a linear-time radix pass, and lets
        # the column buffer double as the key: ``columns`` is freshly allocated by
        # ``np.tile`` above and is not read again.  Both indices are below
        # n_dofs, so the key stays under n_dofs**2 -- far below 2**63 for any
        # mesh that fits in memory.
        key = columns
        key *= n_dofs
        key += rows
        del rows

        # Sorting by (column, row) puts the entries into CSC order and brings the
        # ones that share a pair -- the ones COO would have summed -- next to
        # each other, so a single pass yields both the CSC structure and the slot
        # map below.
        order = np.argsort(key, kind="stable")
        sorted_key = key[order]
        del key

        # First entry of each run of identical (column, row) pairs.
        starts = np.empty(n_entries, dtype=bool)
        starts[0] = True
        np.not_equal(sorted_key[1:], sorted_key[:-1], out=starts[1:])

        # ``slot[i]`` is the CSC data slot that COO entry ``i`` accumulates into.
        # Entries sharing a slot are exactly those sharing a (row, column) pair,
        # so consecutive slots follow the sorted order.
        self.slot = np.empty(n_entries, dtype=np.intp)
        self.slot[order] = np.cumsum(starts) - 1
        del order

        # SciPy selects int32 indices whenever every index fits; matching its
        # choice here means csc_matrix() is handed arrays it can keep as they
        # are, instead of re-casting (and copying) them on every assembly.
        index_dtype = (
            np.int32 if max(n_dofs, n_entries) <= np.iinfo(np.int32).max
            else np.int64
        )

        unique_pairs = sorted_key[starts]
        del sorted_key
        self.indices = (unique_pairs % n_dofs).astype(index_dtype, copy=False)

        indptr = np.zeros(n_dofs + 1, dtype=np.intp)
        np.cumsum(
            np.bincount(unique_pairs // n_dofs, minlength=n_dofs), out=indptr[1:]
        )
        self.indptr = indptr.astype(index_dtype, copy=False)

        self.shape = (n_dofs, n_dofs)
        self.nnz = int(self.indptr[-1])

    def assemble(self, element_stiffness, element_moduli):
        """Scatter element matrices into this plan's fixed CSC structure.

        Parameters
        ----------
        element_stiffness : ndarray, shape (8, 8)
            Element stiffness matrix for unit Young's modulus.
        element_moduli : ndarray, shape (n_elements,)
            Young's modulus of each element, in the mesh's element order.

        Returns
        -------
        csc_matrix, shape (n_dofs, n_dofs)
            Symmetric positive definite once the rigid body modes are restrained.
        """
        values = (
            element_stiffness.reshape(-1)[None, :] * element_moduli[:, None]
        ).reshape(-1)

        # Each entry adds into the slot of its (row, column) pair.  ``bincount``
        # walks the entries in COO order and accumulates in place, which is the
        # same order SciPy's COO -> CSC conversion sums them in, so the result is
        # bit-for-bit the reference assembly (see tests/test_fem.py).
        data = np.bincount(self.slot, weights=values, minlength=self.nnz)
        return sparse.csc_matrix(
            (data, self.indices, self.indptr), shape=self.shape
        )


def assemble_stiffness_matrix(mesh, element_stiffness, element_moduli):
    """Assemble the global stiffness matrix as a sparse CSC matrix.

    The mesh caches the sparsity structure and the COO -> CSC scatter map (see
    :class:`StiffnessAssemblyPlan`), so repeated assemblies on an unchanged mesh
    only recompute values.

    Parameters
    ----------
    mesh : Mesh
    element_stiffness : ndarray, shape (8, 8)
        Element stiffness matrix for unit Young's modulus.
    element_moduli : ndarray, shape (n_elements,)
        Young's modulus of each element, in the mesh's element order.

    Returns
    -------
    csc_matrix, shape (n_dofs, n_dofs)
        Symmetric positive definite once the rigid body modes are restrained.
    """
    return mesh.assembly_plan.assemble(element_stiffness, element_moduli)


def free_degrees_of_freedom(n_dofs, fixed_dofs):
    """Indices of the degrees of freedom that are not restrained."""
    return np.setdiff1d(np.arange(n_dofs, dtype=np.intp), np.asarray(fixed_dofs, dtype=np.intp))


def solve_displacements(stiffness, loads, free_dofs):
    """Solve ``K U = F`` with the restrained degrees of freedom held at zero.

    The system is reduced to the free degrees of freedom and solved with
    SciPy's sparse LU solver; the restrained entries of the returned vector
    stay at zero.
    """
    displacements = np.zeros(stiffness.shape[0])
    reduced = stiffness.tocsr()[free_dofs, :][:, free_dofs].tocsc()
    displacements[free_dofs] = sparse_linalg.spsolve(reduced, loads[free_dofs])
    return displacements
