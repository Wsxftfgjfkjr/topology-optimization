import type { ChangeEvent, FormEvent } from 'react'
import type { OptimizeRequest } from '../types'

/**
 * The seven request fields the backend accepts, with labels aimed at a human
 * rather than at the wire format.
 *
 * The HTML constraints mirror `api/models.py` where HTML can express them, and
 * are only a convenience: the backend validates every request and its 422
 * response is what the UI reports.
 *
 * `step` matters more than it looks. The backend's bounds are almost all
 * exclusive (`gt=0`), which HTML cannot express, so the float fields use
 * `step="any"` rather than a step aligned to a `min` the backend does not have.
 * Getting this wrong is not cosmetic: an input whose value does not sit on the
 * `min + k * step` lattice fails HTML constraint validation and the browser
 * silently refuses to submit the form at all. `volfrac`'s upper bound is
 * inclusive (`le=1`), so it is the one field that can carry a real `max`.
 */
interface Field {
  name: keyof OptimizeRequest
  label: string
  hint?: string
  min?: number
  max?: number
  step: number | 'any'
  integer: boolean
}

export const FIELDS: readonly Field[] = [
  { name: 'nelx', label: 'Elements X', hint: 'columns', min: 1, step: 1, integer: true },
  { name: 'nely', label: 'Elements Y', hint: 'rows', min: 1, step: 1, integer: true },
  // Step 0.1 gives the volume fraction a sensible spinner increment and starts
  // it on a tenths lattice. It is not a hard filter: with no `min` the HTML
  // step base is the input's `value` attribute, which React keeps equal to the
  // current value, so an off-lattice entry such as 0.35 stays valid and is sent
  // on to the backend. The backend accepts anything in (0, 1] and stays the
  // authority; `max` is the one bound HTML can state exactly.
  { name: 'volfrac', label: 'Volume fraction', max: 1, step: 0.1, integer: false },
  { name: 'penal', label: 'SIMP penalty', hint: 'exponent p', step: 'any', integer: false },
  {
    name: 'rmin',
    label: 'Filter radius',
    hint: 'element units',
    step: 'any',
    integer: false,
  },
  { name: 'max_iterations', label: 'Max iterations', min: 1, step: 1, integer: true },
  {
    name: 'tolerance',
    label: 'Convergence tolerance',
    hint: 'max density change',
    step: 'any',
    integer: false,
  },
]

interface ParameterFormProps {
  values: OptimizeRequest
  running: boolean
  onChange: (values: OptimizeRequest) => void
  onSubmit: () => void
}

export function ParameterForm({
  values,
  running,
  onChange,
  onSubmit,
}: ParameterFormProps) {
  const handleChange =
    (field: Field) => (event: ChangeEvent<HTMLInputElement>) => {
      const raw = event.target.value
      // Keep the field editable while it is empty or mid-typing; the backend
      // rejects anything that is not a usable number.
      const parsed = field.integer ? Number.parseInt(raw, 10) : Number.parseFloat(raw)
      onChange({ ...values, [field.name]: Number.isNaN(parsed) ? raw : parsed } as OptimizeRequest)
    }

  const handleSubmit = (event: FormEvent) => {
    event.preventDefault()
    if (!running) onSubmit()
  }

  return (
    <form className="parameter-form" onSubmit={handleSubmit} aria-busy={running}>
      {FIELDS.map((field) => (
        <div className="field" key={field.name}>
          <label htmlFor={`field-${field.name}`}>
            {field.label}
            {field.hint !== undefined && <span className="hint"> {field.hint}</span>}
          </label>
          <input
            id={`field-${field.name}`}
            name={field.name}
            type="number"
            inputMode="decimal"
            min={field.min}
            max={field.max}
            step={field.step}
            value={String(values[field.name])}
            onChange={handleChange(field)}
            disabled={running}
          />
        </div>
      ))}

      <button type="submit" className="primary" disabled={running}>
        {running ? 'Optimizing…' : 'Optimize'}
      </button>
      <p className="form-note">
        The load case and boundary conditions are the canonical cantilever: the
        left edge is fully fixed and a point load is applied mid-height on the
        right edge.
      </p>
    </form>
  )
}
