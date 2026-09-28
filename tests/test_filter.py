"""Tests for the sensitivity filter."""

import numpy as np
import pytest

from topoopt.filter import build_filter_kernel, filter_sensitivities


def test_filter_kernel_is_symmetric_with_the_expected_weights():
    kernel, _ = build_filter_kernel(3, 3, rmin=1.5)
    dense = kernel.toarray()

    assert kernel.shape == (9, 9)
    assert np.allclose(dense, dense.T)

    centre = 4  # element (i=1, j=1) on a 3 x 3 mesh is index j * nelx + i
    assert np.isclose(dense[centre, centre], 1.5)
    assert np.isclose(dense[centre, centre + 1], 0.5)
    assert np.isclose(dense[centre, centre - 4], 1.5 - np.sqrt(2.0))


def test_filter_kernel_row_sums_match_the_cone_weights():
    _, row_sums = build_filter_kernel(3, 3, rmin=1.5)

    # The centre sees four neighbours at distance one and four at sqrt(2).
    assert np.isclose(row_sums[4], 1.5 + 4 * 0.5 + 4 * (1.5 - np.sqrt(2.0)))

    # A corner sees two neighbours at distance one and one at sqrt(2).
    assert np.isclose(row_sums[0], 1.5 + 2 * 0.5 + (1.5 - np.sqrt(2.0)))


def test_filter_kernel_excludes_elements_beyond_the_radius():
    kernel, _ = build_filter_kernel(5, 5, rmin=1.5)
    dense = kernel.toarray()

    # Element (0, 0) and (2, 0) are two elements apart, further than rmin.
    assert dense[0, 2] == 0.0
    # Element (0, 0) and (4, 4) are far apart.
    assert dense[0, 24] == 0.0


def test_filter_kernel_rejects_a_non_positive_radius():
    with pytest.raises(ValueError):
        build_filter_kernel(4, 4, rmin=0.0)


def test_filter_preserves_a_uniform_sensitivity_field():
    """With uniform densities the filter is a weighted average that sums to one."""
    kernel, row_sums = build_filter_kernel(4, 3, rmin=1.5)
    density = np.full((3, 4), 0.4)
    sensitivities = -np.ones((3, 4))

    filtered = filter_sensitivities(sensitivities, density, kernel, row_sums)

    assert np.allclose(filtered, sensitivities)


def test_filter_spreads_a_point_sensitivity_over_the_kernel():
    kernel, row_sums = build_filter_kernel(3, 3, rmin=1.5)
    density = np.ones((3, 3))
    sensitivities = np.zeros((3, 3))
    sensitivities[1, 1] = -1.0

    filtered = filter_sensitivities(sensitivities, density, kernel, row_sums)

    # The single negative sensitivity is spread with the cone weights and kept
    # negative; with unit densities the density factors cancel out.
    assert np.isclose(filtered[1, 1], -1.5 / row_sums[4])
    assert np.isclose(filtered[1, 2], -0.5 / row_sums[5])
    assert np.isclose(filtered[0, 0], -(1.5 - np.sqrt(2.0)) / row_sums[0])


def test_filter_preserves_left_right_symmetry():
    kernel, row_sums = build_filter_kernel(6, 4, rmin=2.5)
    profile = np.array([0.1, 0.2, 0.3, 0.3, 0.2, 0.1])
    density = np.tile(profile, (4, 1))
    sensitivities = -np.tile(profile, (4, 1))

    filtered = filter_sensitivities(sensitivities, density, kernel, row_sums)

    assert np.allclose(filtered, filtered[:, ::-1])


def test_filter_stays_finite_at_the_density_lower_bound():
    kernel, row_sums = build_filter_kernel(3, 3, rmin=1.5)
    density = np.full((3, 3), 1e-3)
    sensitivities = -np.ones((3, 3))

    filtered = filter_sensitivities(sensitivities, density, kernel, row_sums)

    assert np.all(np.isfinite(filtered))


def test_filter_returns_the_shape_of_its_input():
    kernel, row_sums = build_filter_kernel(4, 3, rmin=1.5)

    filtered = filter_sensitivities(
        -np.ones((3, 4)), np.full((3, 4), 0.5), kernel, row_sums
    )

    assert filtered.shape == (3, 4)
