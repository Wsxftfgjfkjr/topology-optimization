"""Sensitivity filtering, used to suppress checkerboarding.

The filter is the mesh-independence filter of Sigmund and Petersson (1998) in
its sensitivity form: the raw element sensitivities are replaced by a cone
weighted average of the neighbouring sensitivities, scaled by the density
field so that the filtered quantity keeps the same physical dimension as the
compliance derivative.  Unlike the density filter it leaves the finite element
analysis untouched, which keeps the analysis and the design update decoupled.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sparse

# Densities are bounded below by the optimizer, so this floor only guards the
# division in the filter against a caller passing an exactly empty element.
DEFAULT_MIN_DENSITY = 1e-3


def build_filter_kernel(nelx, nely, rmin):
    """Cone ("hat") weighting over element centres.

    Element ``f`` influences element ``e`` with weight

        H[e, f] = max(0, rmin - ||c_e - c_f||)

    where the distance between element centres is measured in units of the
    unit element edge length, so it is simply the index distance.  The
    diagonal ``e == f`` carries weight ``rmin``.

    Parameters
    ----------
    nelx, nely : int
        Mesh size in elements.
    rmin : float
        Filter radius in element units.  Values of one or less leave the
        sensitivities untouched, since no neighbour then falls inside the cone.

    Returns
    -------
    kernel : csc_matrix, shape (nelx * nely, nelx * nely)
        Symmetric, with the mesh's element ordering on both axes.
    row_sums : ndarray, shape (nelx * nely,)
        Row sums of the kernel, i.e. the total weight of each element's
        neighbourhood.  Elements on the boundary have smaller row sums.
    """
    if rmin <= 0.0:
        raise ValueError("rmin must be positive")

    n_elements = nelx * nely
    reach = int(np.ceil(rmin))
    rows, columns, weights = [], [], []

    # One pass per (di, dj) offset: every element is paired with the neighbour
    # that many elements away, clipped to the mesh.  Offsets outside the cone
    # are skipped, which also drops the corners of the bounding box whenever
    # rmin is smaller than sqrt(2) * reach.
    for di in range(-reach, reach + 1):
        for dj in range(-reach, reach + 1):
            weight = rmin - float(np.hypot(di, dj))
            if weight <= 0.0:
                continue

            target_i = np.arange(max(0, -di), min(nelx, nelx - di))[:, None]
            target_j = np.arange(max(0, -dj), min(nely, nely - dj))[None, :]

            target = target_j * nelx + target_i
            source = (target_j + dj) * nelx + (target_i + di)

            rows.append(target.reshape(-1))
            columns.append(source.reshape(-1))
            weights.append(np.full(target.size, weight))

    kernel = sparse.coo_matrix(
        (np.concatenate(weights), (np.concatenate(rows), np.concatenate(columns))),
        shape=(n_elements, n_elements),
    ).tocsc()
    row_sums = np.asarray(kernel.sum(axis=1)).ravel()
    return kernel, row_sums


def filter_sensitivities(
    sensitivities,
    densities,
    kernel,
    row_sums,
    min_density=DEFAULT_MIN_DENSITY,
):
    """Apply the Sigmund-Petersson sensitivity filter.

    ``dc_e <- sum_f H[e, f] * x_f * dc_f / (x_e * sum_f H[e, f])``

    Both arrays are interpreted in the mesh's element order, so an array of
    shape ``(nely, nelx)`` is filtered row by row and returned with the same
    shape.  A uniform sensitivity field is reproduced exactly.
    """
    x = np.asarray(densities, dtype=float).reshape(-1)
    dc = np.asarray(sensitivities, dtype=float).reshape(-1)

    filtered = (kernel @ (x * dc)) / (row_sums * np.maximum(x, min_density))
    return filtered.reshape(np.shape(sensitivities))
