import { useEffect, useRef } from 'react'
import type { OptimizeResponse } from '../types'

/**
 * Density -> colour ramp.
 *
 * A single-hue light-to-dark blue, matching the ramp the canonical Matplotlib
 * figure uses so the browser and the committed documentation images read the
 * same way: light is void, dark is solid.
 */
const RAMP: readonly [number, number, number][] = [
  [205, 226, 251],
  [134, 182, 239],
  [57, 135, 229],
  [28, 92, 171],
  [13, 54, 107],
]

export function densityColor(t: number): string {
  const clamped = Math.min(1, Math.max(0, t))
  const span = RAMP.length - 1
  const scaled = clamped * span
  const index = Math.min(span - 1, Math.floor(scaled))
  const frac = scaled - index
  const low = RAMP[index] as [number, number, number]
  const high = RAMP[index + 1] as [number, number, number]
  const channel = (a: number, b: number) => Math.round(a + (b - a) * frac)
  return `rgb(${channel(low[0], high[0])}, ${channel(low[1], high[1])}, ${channel(low[2], high[2])})`
}

/**
 * Paint the element field onto a canvas.
 *
 * One rectangle per element, no smoothing or interpolation, so the discrete
 * grid stays visible at any zoom. Row 0 is drawn at the *bottom*, matching the
 * canonical plot's `imshow(density, origin="lower", extent=(0, nelx, 0, nely))`
 * and therefore `density[j][i]` = element `i` from the left, `j` from the bottom.
 */
function paint(canvas: HTMLCanvasElement, density: number[][]): void {
  const nely = density.length
  const nelx = density[0]?.length ?? 0
  const context = canvas.getContext('2d')
  if (context === null || nelx === 0 || nely === 0) return

  const ratio = window.devicePixelRatio || 1
  const cssWidth = canvas.clientWidth || nelx
  const cssHeight = (cssWidth * nely) / nelx

  canvas.width = Math.round(cssWidth * ratio)
  canvas.height = Math.round(cssHeight * ratio)
  canvas.style.height = `${cssHeight}px`

  context.setTransform(ratio, 0, 0, ratio, 0, 0)
  context.clearRect(0, 0, cssWidth, cssHeight)

  const cellWidth = cssWidth / nelx
  const cellHeight = cssHeight / nely

  for (let j = 0; j < nely; j += 1) {
    const row = density[j]
    if (row === undefined) continue
    // Row 0 sits at the bottom of the canvas.
    const y = (nely - 1 - j) * cellHeight
    for (let i = 0; i < nelx; i += 1) {
      const value = row[i]
      if (value === undefined) continue
      context.fillStyle = densityColor(value)
      // Overdraw by a fraction of a pixel so adjacent cells leave no seam.
      context.fillRect(i * cellWidth, y, cellWidth + 0.5, cellHeight + 0.5)
    }
  }
}

interface TopologyViewProps {
  result: OptimizeResponse | null
  running: boolean
}

export function TopologyView({ result, running }: TopologyViewProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (canvas === null || result === null) return

    const draw = () => paint(canvas, result.density)
    draw()

    // Repaint on resize so the field always fills the panel crisply.
    const observer = new ResizeObserver(draw)
    observer.observe(canvas)
    return () => observer.disconnect()
  }, [result])

  if (result === null) {
    return (
      <div className="panel-body placeholder" data-testid="topology-empty">
        <p>
          Set the parameters and choose <strong>Optimize</strong> to solve the
          canonical cantilever. The optimized density field appears here.
        </p>
      </div>
    )
  }

  const { nely, nelx, density } = result
  const label =
    `Optimized density field on a ${nelx} by ${nely} element grid. ` +
    `Density runs from ${minOf(density).toFixed(3)} (void) to ` +
    `${maxOf(density).toFixed(3)} (solid).`

  return (
    <div className="panel-body">
      <canvas
        ref={canvasRef}
        className="topology-canvas"
        data-testid="topology-canvas"
        data-nelx={nelx}
        data-nely={nely}
        role="img"
        aria-label={label}
        aria-busy={running}
      />
      <div className="legend">
        <span className="legend-label">void</span>
        <span className="legend-ramp" aria-hidden="true" />
        <span className="legend-label">solid</span>
        <span className="legend-note">
          element density, {nelx} × {nely}
        </span>
      </div>
    </div>
  )
}

function minOf(grid: number[][]): number {
  let value = Infinity
  for (const row of grid) for (const cell of row) if (cell < value) value = cell
  return Number.isFinite(value) ? value : 0
}

function maxOf(grid: number[][]): number {
  let value = -Infinity
  for (const row of grid) for (const cell of row) if (cell > value) value = cell
  return Number.isFinite(value) ? value : 0
}
