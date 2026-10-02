import type { ConvergencePoint, OptimizeResponse } from '../types'

/** A small, fully determined response shaped exactly like the FastAPI one. */
export function makeResponse(
  overrides: Partial<OptimizeResponse> = {},
): OptimizeResponse {
  const nelx = overrides.nelx ?? 4
  const nely = overrides.nely ?? 2

  const density =
    overrides.density ??
    Array.from({ length: nely }, (_, j) =>
      Array.from({ length: nelx }, (_, i) => (i + 1) / (nelx + j)),
    )

  const history: ConvergencePoint[] =
    overrides.history ??
    Array.from({ length: 5 }, (_, index) => ({
      iteration: index + 1,
      compliance: 1000 / (index + 1),
      volume_fraction: 0.4,
      change: 0.2 / (index + 1),
    }))

  return {
    nelx,
    nely,
    n_elements: nelx * nely,
    n_dofs: 2 * (nelx + 1) * (nely + 1),
    volfrac: 0.4,
    penal: 3,
    rmin: 1.5,
    max_iterations: 200,
    tolerance: 0.01,
    iterations: history.length,
    converged: true,
    final_compliance: 250.46,
    final_volume_fraction: 0.4,
    final_change: 0.00901,
    density,
    history,
    ...overrides,
  }
}

export function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  })
}

/** The exact shape FastAPI returns for a rejected field. */
export function validationResponse(field: string, message: string): Response {
  return jsonResponse(
    {
      detail: [
        {
          type: 'less_than_equal',
          loc: ['body', field],
          msg: message,
          input: 1.5,
        },
      ],
    },
    422,
  )
}
