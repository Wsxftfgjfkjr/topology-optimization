"""Tests for the HTTP API layer.

Routine endpoint tests use a small deterministic mesh so the suite stays fast;
the canonical 60x20 case appears once, as a regression check that the API agrees
with the numerical core.
"""

import inspect

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api import service
from api.main import app
from api.models import OptimizeRequest
from topoopt import optimizer as optimizer_module
from topoopt.optimizer import TopologyOptimizer
from topoopt.problems import build_cantilever

client = TestClient(app)

# Small and deterministic: 24 elements, converges or caps in a few iterations.
SMALL = {"nelx": 6, "nely": 4, "max_iterations": 8}


def post(payload):
    return client.post("/api/v1/optimize", json=payload)


# --------------------------------------------------------------------------
# health and schema
# --------------------------------------------------------------------------

def test_health_endpoint_reports_ok():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_openapi_schema_exposes_the_intended_endpoints():
    schema = client.get("/openapi.json").json()

    assert set(schema["paths"]) == {"/health", "/api/v1/optimize"}
    assert set(schema["paths"]["/api/v1/optimize"]) == {"post"}
    assert "OptimizeRequest" in schema["components"]["schemas"]
    assert "OptimizeResponse" in schema["components"]["schemas"]


# --------------------------------------------------------------------------
# a valid request
# --------------------------------------------------------------------------

def test_small_optimization_request_succeeds():
    response = post(SMALL)

    assert response.status_code == 200
    body = response.json()
    assert body["nelx"] == 6
    assert body["nely"] == 4
    assert body["n_elements"] == 24
    assert body["n_dofs"] == 2 * 5 * 7
    assert body["iterations"] >= 1
    assert isinstance(body["converged"], bool)


def test_response_density_has_the_core_layout_and_physical_values():
    body = post(SMALL).json()

    density = np.array(body["density"])
    assert density.shape == (body["nely"], body["nelx"])
    assert np.isfinite(density).all()
    # Densities live in [min_density, 1]; the core's floor is 1e-3.
    assert density.min() >= optimizer_module.DEFAULT_MIN_DENSITY
    assert density.max() <= 1.0


def test_response_history_and_final_scalars_agree():
    body = post(SMALL).json()

    history = body["history"]
    assert len(history) == body["iterations"]
    assert [point["iteration"] for point in history] == list(
        range(1, body["iterations"] + 1)
    )

    last = history[-1]
    assert body["final_compliance"] == last["compliance"]
    assert body["final_volume_fraction"] == last["volume_fraction"]
    assert body["final_change"] == last["change"]

    # The design the run converged on sits at the requested volume fraction.
    assert body["final_volume_fraction"] == pytest.approx(body["volfrac"], abs=1e-6)


def test_request_echoes_the_applied_parameters():
    payload = {**SMALL, "volfrac": 0.35, "penal": 2.5, "rmin": 2.0,
               "tolerance": 5e-3}
    body = post(payload).json()

    assert body["volfrac"] == 0.35
    assert body["penal"] == 2.5
    assert body["rmin"] == 2.0
    assert body["tolerance"] == 5e-3
    assert body["max_iterations"] == SMALL["max_iterations"]


def test_omitted_fields_use_the_documented_defaults():
    body = post({"nelx": 6, "nely": 4, "max_iterations": 3}).json()
    defaults = OptimizeRequest()

    assert body["volfrac"] == defaults.volfrac
    assert body["penal"] == defaults.penal
    assert body["rmin"] == defaults.rmin
    assert body["tolerance"] == defaults.tolerance


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------

@pytest.mark.parametrize("volfrac", [0.0, -0.1, 1.5])
def test_invalid_volume_fraction_is_rejected(volfrac):
    response = post({**SMALL, "volfrac": volfrac})

    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"][-1] == "volfrac"


@pytest.mark.parametrize("field,value", [("nelx", 0), ("nelx", -3), ("nely", 0)])
def test_invalid_mesh_dimensions_are_rejected(field, value):
    response = post({**SMALL, field: value})

    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"][-1] == field


@pytest.mark.parametrize("field,value", [
    ("penal", 0.0),
    ("penal", -1.0),
    ("rmin", 0.0),
    ("rmin", -1.5),
    ("max_iterations", 0),
    ("max_iterations", -1),
    ("tolerance", 0.0),
    ("tolerance", -1e-3),
])
def test_invalid_numerical_parameters_are_rejected(field, value):
    response = post({**SMALL, field: value})

    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"][-1] == field


def test_unknown_fields_are_rejected_rather_than_ignored():
    """A typo must not silently fall back to a default."""
    response = post({**SMALL, "maxiter": 50})

    assert response.status_code == 422
    assert response.json()["detail"][0]["type"] == "extra_forbidden"


def test_validation_errors_do_not_leak_internals():
    response = post({**SMALL, "volfrac": 7.0})

    assert response.status_code == 422
    for leak in ("Traceback", "/Users/", "site-packages"):
        assert leak not in response.text


# --------------------------------------------------------------------------
# equivalence with the numerical core
# --------------------------------------------------------------------------

def test_api_matches_a_direct_core_invocation():
    """The API must not be a second version of the cantilever problem."""
    payload = {"nelx": 8, "nely": 5, "volfrac": 0.45, "penal": 3.0, "rmin": 1.5,
               "max_iterations": 25, "tolerance": 1e-2}

    body = post(payload).json()

    mesh, fixed_dofs, loads = build_cantilever(8, 5)
    direct = TopologyOptimizer(
        mesh, fixed_dofs, loads, volfrac=0.45, penal=3.0, rmin=1.5,
        max_iterations=25, tolerance=1e-2, verbose=False,
    ).run()

    assert body["iterations"] == direct.iterations
    assert body["converged"] == direct.converged
    assert body["n_elements"] == mesh.n_elements
    assert body["n_dofs"] == mesh.n_dofs
    assert np.array_equal(np.array(body["density"]), direct.density)
    assert [point["compliance"] for point in body["history"]] == [
        record.compliance for record in direct.history
    ]


def test_canonical_request_reproduces_the_canonical_run():
    """The 60x20 request is the example's run, to within rounding."""
    body = post({"nelx": 60, "nely": 20}).json()

    assert body["iterations"] == 44
    assert body["converged"] is True
    assert body["final_compliance"] == pytest.approx(250.46, rel=1e-6)
    assert body["final_volume_fraction"] == pytest.approx(0.40, abs=1e-6)
    assert np.array(body["density"]).shape == (20, 60)


def test_request_defaults_track_the_numerical_core():
    """An omitted field must mean the same thing here as in the core."""
    signature = inspect.signature(TopologyOptimizer.__init__).parameters
    defaults = OptimizeRequest()

    assert defaults.volfrac == signature["volfrac"].default
    assert defaults.penal == signature["penal"].default
    assert defaults.rmin == signature["rmin"].default
    assert defaults.max_iterations == optimizer_module.DEFAULT_MAX_ITERATIONS
    assert defaults.tolerance == optimizer_module.DEFAULT_CONVERGENCE_TOLERANCE


# --------------------------------------------------------------------------
# failure handling
# --------------------------------------------------------------------------

def test_core_failure_becomes_a_controlled_500(monkeypatch):
    """A numerical failure must not become a traceback on the wire."""

    class ExplodingOptimizer:
        def __init__(self, *args, **kwargs):
            pass

        def run(self):
            raise MemoryError("allocation failed at 0xdeadbeef in /Users/private")

    monkeypatch.setattr(service, "TopologyOptimizer", ExplodingOptimizer)

    response = post(SMALL)

    assert response.status_code == 500
    detail = response.json()["detail"]
    assert detail
    for leak in ("0xdeadbeef", "/Users/", "Traceback", "MemoryError"):
        assert leak not in response.text


def test_programming_errors_are_not_swallowed(monkeypatch):
    """A bug in our own code must not be reported as a numerical failure."""

    class BuggyOptimizer:
        def __init__(self, *args, **kwargs):
            pass

        def run(self):
            raise AttributeError("a genuine bug")

    monkeypatch.setattr(service, "TopologyOptimizer", BuggyOptimizer)

    with pytest.raises(AttributeError):
        post(SMALL)
