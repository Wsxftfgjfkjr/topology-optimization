import type { OptimizeResponse } from '../types'
import {
  formatChange,
  formatCompliance,
  formatCount,
  formatVolumeFraction,
} from '../format'

interface MetricsProps {
  result: OptimizeResponse | null
}

/**
 * Final values for the run, formatted for reading rather than for round-trip
 * precision. Convergence status is stated in words as well as in colour.
 */
export function Metrics({ result }: MetricsProps) {
  if (result === null) {
    return (
      <p className="metrics-empty">
        Final compliance, volume fraction and density change are reported here.
      </p>
    )
  }

  const items: { label: string; value: string; title?: string }[] = [
    { label: 'Iterations', value: formatCount(result.iterations) },
    { label: 'Final compliance', value: formatCompliance(result.final_compliance) },
    {
      label: 'Volume fraction',
      value: formatVolumeFraction(result.final_volume_fraction),
    },
    {
      label: 'Density change',
      value: formatChange(result.final_change),
      title: 'Maximum density change in the final iteration',
    },
    { label: 'Elements', value: formatCount(result.n_elements) },
    { label: 'Degrees of freedom', value: formatCount(result.n_dofs) },
  ]

  return (
    <div className="metrics">
      <div
        className={result.converged ? 'status status-ok' : 'status status-warn'}
        data-testid="convergence-status"
      >
        {result.converged
          ? `Converged in ${result.iterations} iterations`
          : `Stopped at the ${result.iterations}-iteration cap without converging`}
      </div>
      <dl className="metric-grid">
        {items.map((item) => (
          <div className="metric" key={item.label}>
            <dt>{item.label}</dt>
            <dd title={item.title}>{item.value}</dd>
          </div>
        ))}
      </dl>
    </div>
  )
}
