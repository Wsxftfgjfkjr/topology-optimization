import { useCallback, useRef, useState } from 'react'
import { ApiError, optimize } from './api'
import type { FieldError } from './api'
import type { OptimizeRequest, OptimizeResponse } from './types'
import { ParameterForm } from './components/ParameterForm'
import { TopologyView } from './components/TopologyView'
import { ConvergenceChart } from './components/ConvergenceChart'
import { Metrics } from './components/Metrics'

/**
 * The backend's own defaults, taken from `api/models.py`. Kept as one constant
 * so the initial form and the request it produces agree.
 */
export const DEFAULT_REQUEST: OptimizeRequest = {
  nelx: 60,
  nely: 20,
  volfrac: 0.4,
  penal: 3.0,
  rmin: 1.5,
  max_iterations: 200,
  tolerance: 0.01,
}

interface Failure {
  kind: 'validation' | 'server' | 'network'
  message: string
  fieldErrors: FieldError[]
}

export function App() {
  const [request, setRequest] = useState<OptimizeRequest>(DEFAULT_REQUEST)
  const [result, setResult] = useState<OptimizeResponse | null>(null)
  const [failure, setFailure] = useState<Failure | null>(null)
  const [running, setRunning] = useState(false)
  // Guards against a second submission slipping through before React has
  // re-rendered the disabled button.
  const inFlight = useRef(false)

  const run = useCallback(async () => {
    if (inFlight.current) return
    inFlight.current = true
    setRunning(true)
    setFailure(null)

    try {
      const response = await optimize(request)
      setResult(response)
    } catch (error) {
      if (error instanceof ApiError) {
        setFailure({
          kind: error.kind,
          message: error.message,
          fieldErrors: error.fieldErrors,
        })
      } else {
        setFailure({
          kind: 'network',
          message: 'The optimization could not be completed.',
          fieldErrors: [],
        })
      }
      // A failed request deliberately leaves the previous result on screen.
    } finally {
      inFlight.current = false
      setRunning(false)
    }
  }, [request])

  return (
    <div className="app">
      <header className="app-header">
        <h1>topoopt</h1>
        <p>Interactive 2D structural topology optimization</p>
      </header>

      <div className="layout">
        <section className="panel panel-parameters" aria-labelledby="parameters-heading">
          <h2 id="parameters-heading">Parameters</h2>
          <ParameterForm
            values={request}
            running={running}
            onChange={setRequest}
            onSubmit={run}
          />
        </section>

        <div className="column">
          <section className="panel" aria-labelledby="topology-heading">
            <h2 id="topology-heading">Optimized material layout</h2>
            <TopologyView result={result} running={running} />
          </section>

          <section className="panel" aria-labelledby="convergence-heading">
            <h2 id="convergence-heading">Convergence</h2>
            <ConvergenceChart history={result?.history ?? []} />
          </section>
        </div>
      </div>

      <footer className="app-footer">
        <div className="status-line" role="status" aria-live="polite">
          {running && <span className="running">Optimizing…</span>}
          {!running && result !== null && !failure && (
            <span>Last run completed for a {result.nelx} × {result.nely} mesh.</span>
          )}
        </div>

        {failure !== null && (
          <div className={`alert alert-${failure.kind}`} role="alert">
            <p className="alert-message">{failure.message}</p>
            {failure.fieldErrors.length > 0 && (
              <ul className="alert-fields">
                {failure.fieldErrors.map((fieldError) => (
                  <li key={`${fieldError.field}-${fieldError.message}`}>
                    <code>{fieldError.field}</code> {fieldError.message}
                  </li>
                ))}
              </ul>
            )}
          </div>
        )}

        <Metrics result={result} />
      </footer>
    </div>
  )
}
