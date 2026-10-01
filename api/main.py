"""FastAPI application exposing the topoopt core over HTTP.

Run it with::

    uvicorn api.main:app --reload

The generated schema and interactive documentation are at ``/docs`` (Swagger UI)
and ``/openapi.json``.

Execution model
---------------
Both endpoints are declared with ``def``, not ``async def``.  The optimization is
a CPU-bound call into SciPy that blocks for its whole duration, so it is written
as a plain blocking function and left to FastAPI, which runs it in a worker
thread and keeps the event loop free to answer ``/health`` and ``/docs``.
Declaring it ``async def`` would block that loop instead.  Each request still
occupies one thread until it finishes; genuine parallelism across concurrent
requests would need a process pool, which is a later phase.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from . import service
from .models import OptimizeRequest, OptimizeResponse

app = FastAPI(
    title="topoopt API",
    version="0.1.0",
    summary="Run the 2D cantilever topology-optimization core over HTTP.",
)


@app.exception_handler(service.OptimizationFailed)
async def _optimization_failed_handler(request, exc: service.OptimizationFailed):
    """Report a core failure as a 500 that reveals nothing about the server."""
    return JSONResponse(status_code=500, content={"detail": str(exc)})


@app.get("/health", summary="Liveness probe", tags=["meta"])
def health() -> dict[str, str]:
    """Return ``{"status": "ok"}`` once the application is serving."""
    return {"status": "ok"}


@app.post(
    "/api/v1/optimize",
    response_model=OptimizeResponse,
    summary="Optimize the canonical cantilever",
    tags=["optimization"],
)
def optimize(request: OptimizeRequest) -> OptimizeResponse:
    """Run the canonical cantilever problem and return the density field.

    The request carries only the problem parameters; the load case and boundary
    conditions are the fixed cantilever used by the example and the benchmark
    suite.  The response carries the optimized density field and the
    per-iteration convergence history.

    Blocking: the call returns when the optimization finishes, which for the
    default 60x20 mesh is around a fifth of a second, and grows quickly with
    mesh size.
    """
    return service.optimize(request)
