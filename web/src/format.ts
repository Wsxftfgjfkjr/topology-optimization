/**
 * Number formatting for display only.
 *
 * These helpers never feed values back into a request or into the canvas; the
 * numbers the solver produced stay untouched everywhere else.
 */

/** Compliance spans roughly 1e2..1e4; six significant digits is plenty. */
export function formatCompliance(value: number): string {
  return value.toPrecision(6)
}

/** The volume constraint is meaningful to about four decimals. */
export function formatVolumeFraction(value: number): string {
  return value.toFixed(4)
}

/** The density change reaches 1e-3 and below, so use scientific notation. */
export function formatChange(value: number): string {
  return value.toExponential(2)
}

/** Counts such as elements and DOFs. */
export function formatCount(value: number): string {
  return value.toLocaleString('en-US')
}

/** Axis ticks: compact but not misleading. */
export function formatTick(value: number): string {
  const magnitude = Math.abs(value)
  if (magnitude !== 0 && (magnitude < 1e-2 || magnitude >= 1e5)) {
    return value.toExponential(1)
  }
  if (magnitude >= 1000) return value.toFixed(0)
  if (magnitude >= 10) return value.toFixed(1)
  return value.toFixed(3)
}
