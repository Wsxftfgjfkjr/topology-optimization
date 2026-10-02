/**
 * HTTP boundary for the topoopt backend.
 *
 * The base path is configurable through `VITE_API_BASE` so a deployment can
 * point somewhere else without touching components. During development it is
 * left empty and the Vite dev server proxies the relative `/api` path to
 * uvicorn (see `vite.config.ts`).
 */

import type {
  OptimizeRequest,
  OptimizeResponse,
  ValidationErrorItem,
} from './types'

export const API_BASE: string = import.meta.env.VITE_API_BASE ?? ''

/** A field-level message extracted from a FastAPI 422 response. */
export interface FieldError {
  field: string
  message: string
}

/** Failure modes the UI treats differently. */
export type ApiErrorKind = 'validation' | 'server' | 'network'

export class ApiError extends Error {
  readonly kind: ApiErrorKind
  readonly fieldErrors: FieldError[]
  readonly status?: number

  constructor(
    kind: ApiErrorKind,
    message: string,
    options: { fieldErrors?: FieldError[]; status?: number } = {},
  ) {
    super(message)
    this.name = 'ApiError'
    this.kind = kind
    this.fieldErrors = options.fieldErrors ?? []
    this.status = options.status
  }
}

/**
 * Turn FastAPI's `loc` array into a human field name.
 *
 * `["body", "volfrac"]` and `["body", 3]` both mean a request field; the last
 * element is the one worth showing.
 */
function fieldFromLocation(loc: ValidationErrorItem['loc']): string {
  const last = loc[loc.length - 1]
  return last === undefined ? 'request' : String(last)
}

function parseValidationDetail(detail: unknown): FieldError[] {
  if (!Array.isArray(detail)) return []
  return detail.flatMap((item) => {
    if (typeof item !== 'object' || item === null) return []
    const entry = item as Partial<ValidationErrorItem>
    if (typeof entry.msg !== 'string') return []
    return [{ field: fieldFromLocation(entry.loc ?? []), message: entry.msg }]
  })
}

function summaryFrom(fieldErrors: FieldError[], fallback: string): string {
  if (fieldErrors.length === 0) return fallback
  if (fieldErrors.length === 1) {
    const [only] = fieldErrors
    return only ? `${only.field}: ${only.message}` : fallback
  }
  return `${fieldErrors.length} parameters were rejected`
}

async function readJson(response: Response): Promise<unknown> {
  try {
    return await response.json()
  } catch {
    return null
  }
}

/**
 * Run one optimization.
 *
 * Errors are normalised into `ApiError` so the UI never has to interpret a
 * `Response` itself, and never sees a stack trace.
 */
export async function optimize(
  request: OptimizeRequest,
  signal?: AbortSignal,
): Promise<OptimizeResponse> {
  let response: Response
  try {
    response = await fetch(`${API_BASE}/api/v1/optimize`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
      signal,
    })
  } catch (cause) {
    if (cause instanceof DOMException && cause.name === 'AbortError') throw cause
    throw new ApiError(
      'network',
      'Could not reach the optimization backend. Is the API running?',
    )
  }

  const body = await readJson(response)

  if (response.status === 422) {
    const fieldErrors = parseValidationDetail(
      (body as { detail?: unknown } | null)?.detail,
    )
    throw new ApiError(
      'validation',
      summaryFrom(fieldErrors, 'The request was rejected as invalid.'),
      { fieldErrors, status: response.status },
    )
  }

  if (!response.ok) {
    const detail = (body as { detail?: unknown } | null)?.detail
    // Prefer the backend's own message. When there is none the request most
    // likely never reached FastAPI at all -- the Vite dev proxy answers 500
    // with an empty body when its upstream is down -- so point at that rather
    // than reporting a bare status code.
    throw new ApiError(
      'server',
      typeof detail === 'string' && detail.length > 0
        ? detail
        : `The backend returned status ${response.status} without a message. ` +
          'Is the API running?',
      { status: response.status },
    )
  }

  return body as OptimizeResponse
}
