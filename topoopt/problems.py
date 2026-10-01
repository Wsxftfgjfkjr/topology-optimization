"""Canonical problem definitions.

The cantilever defined here is the problem the example CLI, the benchmark suite
and the HTTP API all solve.  It lives in the numerical core so that all three
share one definition and cannot drift apart, and so that importing it costs
nothing beyond NumPy and SciPy.

Only the problem -- mesh, restraints and loads -- belongs in this module.
Plotting, the command line and the optimization loop stay where they are.
"""

from __future__ import annotations

import numpy as np

from .fem import Mesh

# Magnitude of the single applied force, in the dimensionless units the solver
# works in.  The solver is linear, so this scales the compliance and leaves the
# optimized layout unchanged.
LOAD_MAGNITUDE = 1.0


def build_cantilever(nelx, nely, load=LOAD_MAGNITUDE):
    """Build the canonical cantilever: fixed left edge, point load on the right.

    Returns the mesh, the restrained degrees of freedom (both components of
    every node on the left edge) and the nodal load vector (a single downward
    force at the middle of the right edge).
    """
    mesh = Mesh(nelx, nely)

    left_edge = mesh.node_index(0, np.arange(nely + 1))
    fixed_dofs = np.concatenate([2 * left_edge, 2 * left_edge + 1]).astype(np.intp)

    loads = np.zeros(mesh.n_dofs)
    loads[2 * mesh.node_index(nelx, nely // 2) + 1] = -load

    return mesh, fixed_dofs, loads
