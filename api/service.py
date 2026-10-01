"""Adapter from validated requests to one run of the numerical core.

Nothing numerical is defined here.  The cantilever's mesh, boundary conditions
and load come from :func:`topoopt.problems.build_cantilever`, and the solve and
design loop come from :class:`topoopt.optimizer.TopologyOptimizer`, so the API
cannot drift away from the problem the example and the benchmark suite run.

The backend depends on ``topoopt`` alone.  In particular it never imports the
example CLI, which would drag Matplotlib into the service for no reason.
"""

from __future__ import annotations

from topoopt.optimizer import TopologyOptimizer
from topoopt.problems import build_cantilever

from .models import ConvergencePoint, OptimizeRequest, OptimizeResponse


class OptimizationFailed(RuntimeError):
    """The numerical core could not finish a well-formed request.

    Raised only for failures that are not the client's fault.  The message is
    written to be safe to return and says nothing about paths, environment or
    implementation internals; the original exception is chained onto it so a
    server-side traceback still carries the detail.
    """


def optimize(request: OptimizeRequest) -> OptimizeResponse:
    """Run the canonical cantilever problem for one validated request.

    The caller is responsible for validation -- ``request`` is expected to have
    passed :class:`OptimizeRequest`'s own checks already.
    """
    mesh, fixed_dofs, loads = build_cantilever(request.nelx, request.nely)

    optimizer = TopologyOptimizer(
        mesh,
        fixed_dofs,
        loads,
        volfrac=request.volfrac,
        penal=request.penal,
        rmin=request.rmin,
        max_iterations=request.max_iterations,
        tolerance=request.tolerance,
        # The API returns structured numbers, not console output.
        verbose=False,
    )

    try:
        result = optimizer.run()
    except MemoryError as exc:
        raise OptimizationFailed(
            "the request exhausted the memory available to the solver; "
            "retry with a smaller mesh"
        ) from exc
    except Exception as exc:
        # Everything the core raises for a *valid* request is a server-side
        # problem: the request model already rejects every input the core
        # itself validates.  Report it without echoing the original text,
        # which may name internals.  Programming errors are not caught here --
        # they surface as an unhandled exception and a plain 500.
        if not isinstance(exc, (ValueError, ArithmeticError)):
            raise
        raise OptimizationFailed(
            "the numerical core could not complete the optimization"
        ) from exc

    history = [
        ConvergencePoint(
            iteration=record.iteration,
            compliance=record.compliance,
            volume_fraction=record.volume_fraction,
            change=record.change,
        )
        for record in result.history
    ]

    return OptimizeResponse(
        nelx=mesh.nelx,
        nely=mesh.nely,
        n_elements=mesh.n_elements,
        n_dofs=mesh.n_dofs,
        volfrac=request.volfrac,
        penal=request.penal,
        rmin=request.rmin,
        max_iterations=request.max_iterations,
        tolerance=request.tolerance,
        iterations=result.iterations,
        converged=result.converged,
        final_compliance=history[-1].compliance,
        final_volume_fraction=history[-1].volume_fraction,
        final_change=history[-1].change,
        # tolist() gives nely rows of nelx JSON numbers, the core's own layout.
        density=result.density.tolist(),
        history=history,
    )
