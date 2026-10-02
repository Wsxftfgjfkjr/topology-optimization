/**
 * TypeScript mirror of the FastAPI contract in `api/models.py`.
 *
 * Kept in one place so the wire format has a single description on this side.
 * The Pydantic models remain authoritative: the backend validates every request
 * and rejects unknown fields, so nothing here is a substitute for that.
 */

/** `api.models.OptimizeRequest` */
export interface OptimizeRequest {
  nelx: number
  nely: number
  volfrac: number
  penal: number
  rmin: number
  max_iterations: number
  tolerance: number
}

/** `api.models.ConvergencePoint` — one row of the per-iteration history. */
export interface ConvergencePoint {
  iteration: number
  compliance: number
  volume_fraction: number
  change: number
}

/** `api.models.OptimizeResponse` */
export interface OptimizeResponse {
  nelx: number
  nely: number
  n_elements: number
  n_dofs: number

  volfrac: number
  penal: number
  rmin: number
  max_iterations: number
  tolerance: number

  iterations: number
  converged: boolean
  final_compliance: number
  final_volume_fraction: number
  final_change: number

  /**
   * Element densities as `nely` rows of `nelx` values, matching the core's
   * `(nely, nelx)` layout. Row 0 is the bottom row, the orientation the
   * canonical `imshow(..., origin="lower")` plot uses.
   */
  density: number[][]
  history: ConvergencePoint[]
}

/** One entry of FastAPI's 422 `detail` array. */
export interface ValidationErrorItem {
  type: string
  loc: (string | number)[]
  msg: string
  input?: unknown
}

/** The body FastAPI returns for a 422. */
export interface ValidationErrorResponse {
  detail: ValidationErrorItem[]
}

/** The body the API returns for a handled 5xx. */
export interface ServerErrorResponse {
  detail: string
}
