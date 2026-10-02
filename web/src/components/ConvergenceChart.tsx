import { useId, useState } from 'react'
import type { ConvergencePoint } from '../types'
import { formatTick } from '../format'

/**
 * Line charts drawn as inline SVG.
 *
 * The three series span very different ranges (compliance ~1e2, volume fraction
 * ~0.4, density change ~1e-3 down to 1e-2), so each gets its own labelled axis
 * and only one is shown at a time rather than sharing a scale.
 *
 * No charting dependency: these are single-series line plots with linear or
 * logarithmic y-axes, which is a small amount of geometry to get right and
 * keeps full control over labelling and accessible text.
 */

type MetricKey = 'compliance' | 'volume_fraction' | 'change'

interface Metric {
  key: MetricKey
  label: string
  axis: string
  log: boolean
}

const METRICS: readonly Metric[] = [
  { key: 'compliance', label: 'Compliance', axis: 'c = uᵀKu', log: false },
  {
    key: 'volume_fraction',
    label: 'Volume fraction',
    axis: 'mean density',
    log: false,
  },
  {
    key: 'change',
    label: 'Density change',
    axis: 'max |Δx| (log scale)',
    log: true,
  },
]

const WIDTH = 640
const HEIGHT = 260
const MARGIN = { top: 16, right: 16, bottom: 40, left: 68 }

function niceTicks(min: number, max: number, count = 5): number[] {
  if (!Number.isFinite(min) || !Number.isFinite(max)) return []
  if (min === max) return [min]
  const raw = (max - min) / (count - 1)
  const magnitude = 10 ** Math.floor(Math.log10(raw))
  const step = [1, 2, 2.5, 5, 10].map((m) => m * magnitude).find((s) => s >= raw) ?? magnitude
  const start = Math.ceil(min / step) * step
  const ticks: number[] = []
  for (let v = start; v <= max + step / 1000; v += step) ticks.push(v)
  return ticks
}

function logTicks(min: number, max: number): number[] {
  const lo = Math.floor(Math.log10(min))
  const hi = Math.ceil(Math.log10(max))
  const ticks: number[] = []
  for (let e = lo; e <= hi; e += 1) ticks.push(10 ** e)
  return ticks.filter((t) => t >= min * 0.999 && t <= max * 1.001)
}

interface Scale {
  toY: (value: number) => number
  ticks: number[]
}

function makeScale(values: number[], log: boolean): Scale | null {
  const finite = values.filter((v) => Number.isFinite(v))
  if (finite.length === 0) return null
  const plotHeight = HEIGHT - MARGIN.top - MARGIN.bottom

  if (log) {
    const positive = finite.filter((v) => v > 0)
    if (positive.length === 0) return null
    const lo = Math.log10(Math.min(...positive))
    const hi = Math.log10(Math.max(...positive))
    const span = hi - lo || 1
    return {
      toY: (value) =>
        MARGIN.top + plotHeight - ((Math.log10(value) - lo) / span) * plotHeight,
      ticks: logTicks(10 ** lo, 10 ** hi),
    }
  }

  const lo = Math.min(...finite)
  const hi = Math.max(...finite)
  const pad = (hi - lo) * 0.05 || Math.abs(hi) * 0.05 || 1
  const min = lo - pad
  const max = hi + pad
  const span = max - min || 1
  return {
    toY: (value) => MARGIN.top + plotHeight - ((value - min) / span) * plotHeight,
    ticks: niceTicks(min, max),
  }
}

interface ConvergenceChartProps {
  history: ConvergencePoint[]
}

export function ConvergenceChart({ history }: ConvergenceChartProps) {
  const [metricKey, setMetricKey] = useState<MetricKey>('compliance')
  const titleId = useId()

  if (history.length === 0) {
    return (
      <div className="panel-body placeholder" data-testid="convergence-empty">
        <p>Convergence history appears here after a run.</p>
      </div>
    )
  }

  const metric = METRICS.find((m) => m.key === metricKey) ?? METRICS[0]!
  const values = history.map((point) => point[metric.key])
  const scale = makeScale(values, metric.log)

  const plotWidth = WIDTH - MARGIN.left - MARGIN.right
  const plotHeight = HEIGHT - MARGIN.top - MARGIN.bottom
  const firstIteration = history[0]?.iteration ?? 1
  const lastIteration = history[history.length - 1]?.iteration ?? 1
  const iterationSpan = lastIteration - firstIteration || 1
  const toX = (iteration: number) =>
    MARGIN.left + ((iteration - firstIteration) / iterationSpan) * plotWidth

  const path =
    scale === null
      ? ''
      : history
          .map(
            (point, index) =>
              `${index === 0 ? 'M' : 'L'} ${toX(point.iteration).toFixed(2)} ${scale
                .toY(point[metric.key])
                .toFixed(2)}`,
          )
          .join(' ')

  const summary = `${metric.label} over ${history.length} iterations, from ${formatTick(
    values[0] ?? 0,
  )} to ${formatTick(values[values.length - 1] ?? 0)}.`

  return (
    <div className="panel-body">
      <div className="chart-controls" role="group" aria-label="Convergence metric">
        {METRICS.map((option) => (
          <button
            key={option.key}
            type="button"
            className={option.key === metricKey ? 'chip chip-active' : 'chip'}
            aria-pressed={option.key === metricKey}
            onClick={() => setMetricKey(option.key)}
          >
            {option.label}
          </button>
        ))}
      </div>

      <svg
        className="chart"
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        role="img"
        aria-labelledby={titleId}
        preserveAspectRatio="xMidYMid meet"
      >
        <title id={titleId}>{summary}</title>

        {scale?.ticks.map((tick) => {
          const y = scale.toY(tick)
          if (y < MARGIN.top - 1 || y > HEIGHT - MARGIN.bottom + 1) return null
          return (
            <g key={tick}>
              <line
                x1={MARGIN.left}
                x2={WIDTH - MARGIN.right}
                y1={y}
                y2={y}
                className="gridline"
              />
              <text x={MARGIN.left - 8} y={y + 4} className="tick" textAnchor="end">
                {formatTick(tick)}
              </text>
            </g>
          )
        })}

        <path d={path} className="series" fill="none" />

        <line
          x1={MARGIN.left}
          x2={WIDTH - MARGIN.right}
          y1={HEIGHT - MARGIN.bottom}
          y2={HEIGHT - MARGIN.bottom}
          className="axis"
        />
        <line
          x1={MARGIN.left}
          x2={MARGIN.left}
          y1={MARGIN.top}
          y2={HEIGHT - MARGIN.bottom}
          className="axis"
        />

        <text
          x={MARGIN.left + plotWidth / 2}
          y={HEIGHT - 8}
          className="axis-label"
          textAnchor="middle"
        >
          iteration
        </text>
        <text
          x={14}
          y={MARGIN.top + plotHeight / 2}
          className="axis-label"
          textAnchor="middle"
          transform={`rotate(-90 14 ${MARGIN.top + plotHeight / 2})`}
        >
          {metric.axis}
        </text>

        <text x={MARGIN.left} y={MARGIN.top - 4} className="tick" textAnchor="start">
          {`iteration ${firstIteration}–${lastIteration}`}
        </text>
      </svg>
    </div>
  )
}
