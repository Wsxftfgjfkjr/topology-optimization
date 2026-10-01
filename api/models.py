"""Request and response models for the optimization API.

The bounds on :class:`OptimizeRequest` mirror the numerical core rather than
inventing new policy, and the defaults are the core's own.  Each field is
annotated with the check it corresponds to:

===================  ==========================================  ==================
field                bound                                       enforced by
===================  ==========================================  ==================
``nelx``, ``nely``   ``> 0``                                     ``fem.Mesh``
``volfrac``          ``0 < volfrac <= 1``                        ``TopologyOptimizer``
``penal``            ``> 0``                                     new -- see below
``rmin``             ``> 0``                                     ``filter.build_filter_kernel``
``max_iterations``   ``>= 1``                                    new -- see below
``tolerance``        ``> 0``                                     new -- see below
===================  ==========================================  ==================

The three "new" bounds cover parameters the core does not police but which are
degenerate or outright broken at the boundary: ``max_iterations = 0`` makes
``TopologyOptimizer.run`` raise ``IndexError`` on its empty history, ``penal = 0``
makes the SIMP interpolation constant so the design no longer affects the
compliance, and ``tolerance = 0`` can never be met so the run always hits the
iteration cap.  There is deliberately no upper bound on the mesh dimensions; see
the V1 limitations in the README.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

# The canonical cantilever the example and the benchmark suite both use.
CANONICAL_NELX = 60
CANONICAL_NELY = 20


class OptimizeRequest(BaseModel):
    """One minimum-compliance cantilever optimization."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "nelx": 60,
                    "nely": 20,
                    "volfrac": 0.4,
                    "penal": 3.0,
                    "rmin": 1.5,
                    "max_iterations": 200,
                    "tolerance": 1e-2,
                }
            ]
        },
    )

    nelx: int = Field(
        default=CANONICAL_NELX,
        gt=0,
        description="Elements along x. The left edge is fully restrained.",
    )
    nely: int = Field(
        default=CANONICAL_NELY,
        gt=0,
        description="Elements along y. A point load is applied mid-height on the right edge.",
    )
    volfrac: float = Field(
        default=0.4,
        gt=0.0,
        le=1.0,
        description="Upper bound on the mean density; element volumes are one.",
    )
    penal: float = Field(
        default=3.0,
        gt=0.0,
        description="SIMP penalty exponent. 1 is linear interpolation, 3 is the usual choice.",
    )
    rmin: float = Field(
        default=1.5,
        gt=0.0,
        description="Sensitivity filter radius in element units. Values at or below 1 do not filter.",
    )
    max_iterations: int = Field(
        default=200,
        ge=1,
        description="Iteration cap. The run stops earlier if it converges.",
    )
    tolerance: float = Field(
        default=1e-2,
        gt=0.0,
        description="Stop once the maximum density change between iterations falls below this.",
    )


class ConvergencePoint(BaseModel):
    """One iteration of the convergence history.

    ``compliance``, ``volume_fraction`` and ``change`` describe the design that
    was analysed at this iteration, which is the design *before* the update this
    iteration performed.
    """

    iteration: int = Field(description="1-based iteration index.")
    compliance: float = Field(description="Compliance c = u^T K u of the analysed design.")
    volume_fraction: float = Field(description="Mean density of the analysed design.")
    change: float = Field(description="Maximum density change this iteration produced.")


class OptimizeResponse(BaseModel):
    """Result of one optimization, shaped for a frontend to render."""

    nelx: int = Field(description="Elements along x.")
    nely: int = Field(description="Elements along y.")
    n_elements: int = Field(description="Total elements, nelx * nely.")
    n_dofs: int = Field(description="Total degrees of freedom before restraint.")

    volfrac: float = Field(description="Volume fraction bound that was applied.")
    penal: float = Field(description="SIMP penalty exponent that was applied.")
    rmin: float = Field(description="Filter radius that was applied.")
    max_iterations: int = Field(description="Iteration cap that was applied.")
    tolerance: float = Field(description="Convergence tolerance that was applied.")

    iterations: int = Field(description="Iterations actually run.")
    converged: bool = Field(description="Whether the density change fell below the tolerance.")
    final_compliance: float = Field(description="Compliance of the last analysed design, history[-1].compliance.")
    final_volume_fraction: float = Field(description="Volume fraction of the last analysed design.")
    final_change: float = Field(description="Density change of the final iteration.")

    density: list[list[float]] = Field(
        description=(
            "Optimized element densities as nely rows of nelx values, matching the "
            "core's (nely, nelx) layout. This is the design *after* the final "
            "update, one iteration later than the final_* scalars above."
        )
    )
    history: list[ConvergencePoint] = Field(
        default_factory=list,
        description="One entry per iteration; the last entry backs the final_* scalars.",
    )
