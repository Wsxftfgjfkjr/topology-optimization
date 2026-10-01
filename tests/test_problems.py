"""Tests for the shared canonical problem and the boundary around it.

``topoopt.problems`` defines the cantilever once for the example, the benchmark
suite and the API.  These tests pin the problem's actual content and pin the
import boundary that keeps the plotting stack out of the backend.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from topoopt import problems
from topoopt.problems import build_cantilever

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize("nelx,nely", [(6, 4), (60, 20)])
def test_cantilever_restrains_the_whole_left_edge(nelx, nely):
    mesh, fixed_dofs, _ = build_cantilever(nelx, nely)

    left_nodes = mesh.node_index(0, np.arange(nely + 1))
    expected = np.sort(np.concatenate([2 * left_nodes, 2 * left_nodes + 1]))

    assert np.array_equal(np.sort(fixed_dofs), expected)
    assert fixed_dofs.dtype == np.intp


@pytest.mark.parametrize("nelx,nely", [(6, 4), (60, 20)])
def test_cantilever_applies_one_downward_point_load_at_mid_height(nelx, nely):
    mesh, _, loads = build_cantilever(nelx, nely)

    loaded_node = mesh.node_index(nelx, nely // 2)
    expected = np.zeros(mesh.n_dofs)
    expected[2 * loaded_node + 1] = -problems.LOAD_MAGNITUDE

    assert np.array_equal(loads, expected)
    assert np.count_nonzero(loads) == 1


def test_load_magnitude_scales_the_load_linearly():
    _, _, unit = build_cantilever(6, 4)
    _, _, doubled = build_cantilever(6, 4, load=2.0 * problems.LOAD_MAGNITUDE)

    assert np.array_equal(doubled, 2.0 * unit)


# --------------------------------------------------------------------------
# the boundary
# --------------------------------------------------------------------------

def test_the_example_reuses_the_shared_builder_rather_than_defining_its_own():
    """There must be exactly one definition of the canonical problem."""
    import examples.cantilever as example

    assert example.build_cantilever is build_cantilever


def test_the_api_reuses_the_shared_builder():
    from api import service

    assert service.build_cantilever is build_cantilever


def _assert_no_matplotlib(import_statement):
    """Import something in a fresh interpreter and check the plotting stack stayed out.

    A subprocess is required: another test importing the example CLI leaves
    matplotlib in ``sys.modules`` for the rest of the session, which would hide
    a regression here.
    """
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; "
            f"{import_statement}; "
            "loaded = sorted(m for m in sys.modules if m.startswith('matplotlib')); "
            "assert not loaded, loaded",
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert completed.returncode == 0, completed.stderr


def test_importing_the_problem_module_does_not_import_matplotlib():
    """The shared problem definition must stay plotting-free."""
    _assert_no_matplotlib("import topoopt.problems")


def test_importing_the_api_does_not_import_matplotlib():
    """Neither must the backend that is built on it."""
    _assert_no_matplotlib("import api.main")
