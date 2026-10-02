import { afterEach, describe, expect, it, vi } from 'vitest'
import { cleanup, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { App, DEFAULT_REQUEST } from './App'
import { jsonResponse, makeResponse, validationResponse } from './test/fixtures'

/**
 * The HTTP boundary is mocked for every test here: the frontend suite must not
 * depend on a running FastAPI server. A real end-to-end check against the live
 * backend is done separately.
 */

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

function mockFetch(...responses: (Response | Promise<Response>)[]) {
  const mock = vi.fn()
  for (const response of responses) mock.mockImplementationOnce(() => Promise.resolve(response))
  vi.stubGlobal('fetch', mock)
  return mock
}

function lastRequestBody(mock: ReturnType<typeof vi.fn>): Record<string, unknown> {
  const call = mock.mock.calls[mock.mock.calls.length - 1]
  const init = call?.[1] as RequestInit | undefined
  return JSON.parse(String(init?.body)) as Record<string, unknown>
}

describe('initial state', () => {
  it('shows the backend defaults in the form', () => {
    render(<App />)

    expect(screen.getByLabelText(/elements x/i)).toHaveValue(DEFAULT_REQUEST.nelx)
    expect(screen.getByLabelText(/elements y/i)).toHaveValue(DEFAULT_REQUEST.nely)
    expect(screen.getByLabelText(/volume fraction/i)).toHaveValue(DEFAULT_REQUEST.volfrac)
    expect(screen.getByLabelText(/simp penalty/i)).toHaveValue(DEFAULT_REQUEST.penal)
    expect(screen.getByLabelText(/filter radius/i)).toHaveValue(DEFAULT_REQUEST.rmin)
    expect(screen.getByLabelText(/max iterations/i)).toHaveValue(DEFAULT_REQUEST.max_iterations)
    expect(screen.getByLabelText(/convergence tolerance/i)).toHaveValue(DEFAULT_REQUEST.tolerance)
  })

  it('does not present a result before anything has been run', () => {
    render(<App />)

    expect(screen.getByTestId('topology-empty')).toBeInTheDocument()
    expect(screen.getByTestId('convergence-empty')).toBeInTheDocument()
    expect(screen.queryByTestId('convergence-status')).not.toBeInTheDocument()
  })
})

describe('input constraints', () => {
  it('steps the volume fraction in tenths and keeps the default on the lattice', () => {
    render(<App />)

    const volfrac = screen.getByLabelText(/volume fraction/i)
    expect(volfrac).toHaveAttribute('step', '0.1')
    expect(volfrac).toHaveAttribute('max', '1')
    // 0.4 must satisfy step=0.1, or the browser would refuse to submit the
    // form with the default values.
    expect(volfrac).toBeValid()
  })

  it('keeps the other parameters on their original steps', () => {
    render(<App />)

    // The float fields have exclusive backend bounds that HTML cannot express,
    // so they stay unconstrained by step; the integer fields step by one.
    expect(screen.getByLabelText(/simp penalty/i)).toHaveAttribute('step', 'any')
    expect(screen.getByLabelText(/filter radius/i)).toHaveAttribute('step', 'any')
    expect(screen.getByLabelText(/convergence tolerance/i)).toHaveAttribute('step', 'any')
    expect(screen.getByLabelText(/elements x/i)).toHaveAttribute('step', '1')
    expect(screen.getByLabelText(/elements y/i)).toHaveAttribute('step', '1')
    expect(screen.getByLabelText(/max iterations/i)).toHaveAttribute('step', '1')
  })

  it('still submits a volume fraction that the step does not itself constrain', async () => {
    // `step` governs the spinner increment and the initial lattice. It is not a
    // hard filter here: with no `min`, the step base is the input's `value`
    // attribute, which React keeps equal to the current value, so an off-lattice
    // value such as 0.35 stays valid and still reaches the backend -- which
    // accepts anything in (0, 1] and remains the authority.
    const user = userEvent.setup()
    const fetchMock = mockFetch(jsonResponse(makeResponse()))
    render(<App />)

    const volfrac = screen.getByLabelText(/volume fraction/i)
    await user.clear(volfrac)
    await user.type(volfrac, '0.35')
    await user.click(screen.getByRole('button', { name: /optimize/i }))

    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1))
    expect(lastRequestBody(fetchMock).volfrac).toBe(0.35)
  })
})

describe('a successful run', () => {
  it('sends the form values as the request payload', async () => {
    const user = userEvent.setup()
    const fetchMock = mockFetch(jsonResponse(makeResponse()))
    render(<App />)

    await user.clear(screen.getByLabelText(/elements x/i))
    await user.type(screen.getByLabelText(/elements x/i), '12')
    await user.click(screen.getByRole('button', { name: /optimize/i }))

    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1))
    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit]
    expect(url).toContain('/api/v1/optimize')
    expect(init.method).toBe('POST')
    expect(lastRequestBody(fetchMock)).toEqual({ ...DEFAULT_REQUEST, nelx: 12 })
  })

  it('renders the final metrics and the converged state', async () => {
    const user = userEvent.setup()
    mockFetch(jsonResponse(makeResponse()))
    render(<App />)

    await user.click(screen.getByRole('button', { name: /optimize/i }))

    expect(await screen.findByTestId('convergence-status')).toHaveTextContent(
      /converged in 5 iterations/i,
    )
    expect(screen.getByText('250.460')).toBeInTheDocument()
    expect(screen.getByText('0.4000')).toBeInTheDocument()
    expect(screen.getByText('9.01e-3')).toBeInTheDocument()
  })

  it('reports a run that hit the iteration cap without converging', async () => {
    const user = userEvent.setup()
    mockFetch(jsonResponse(makeResponse({ converged: false, iterations: 200 })))
    render(<App />)

    await user.click(screen.getByRole('button', { name: /optimize/i }))

    expect(await screen.findByTestId('convergence-status')).toHaveTextContent(
      /without converging/i,
    )
  })

  it('charts the returned history', async () => {
    const user = userEvent.setup()
    mockFetch(jsonResponse(makeResponse()))
    render(<App />)

    await user.click(screen.getByRole('button', { name: /optimize/i }))
    await screen.findByTestId('convergence-status')

    expect(screen.queryByTestId('convergence-empty')).not.toBeInTheDocument()
    expect(screen.getByRole('img', { name: /compliance over 5 iterations/i })).toBeInTheDocument()
  })
})

describe('duplicate submission', () => {
  it('disables the control while a run is in flight', async () => {
    const user = userEvent.setup()
    let release: (value: Response) => void = () => {}
    const pending = new Promise<Response>((resolve) => {
      release = resolve
    })
    const fetchMock = mockFetch(pending)
    render(<App />)

    const button = screen.getByRole('button', { name: /optimize/i })
    await user.click(button)

    const busyButton = await screen.findByRole('button', { name: /optimizing/i })
    expect(busyButton).toBeDisabled()
    expect(screen.getByRole('status')).toHaveTextContent(/optimizing/i)
    expect(screen.getByLabelText(/elements x/i)).toBeDisabled()

    // A second submission attempt must not reach the network.
    await user.click(busyButton)
    expect(fetchMock).toHaveBeenCalledTimes(1)

    release(jsonResponse(makeResponse()))
    await waitFor(() => expect(screen.getByTestId('convergence-status')).toBeInTheDocument())
    expect(fetchMock).toHaveBeenCalledTimes(1)
  })
})

describe('failures', () => {
  it('surfaces a 422 with the rejected field and its message', async () => {
    const user = userEvent.setup()
    mockFetch(validationResponse('volfrac', 'Input should be less than or equal to 1'))
    render(<App />)

    await user.click(screen.getByRole('button', { name: /optimize/i }))

    const alert = await screen.findByRole('alert')
    expect(alert).toHaveTextContent(/volfrac/)
    expect(alert).toHaveTextContent(/less than or equal to 1/i)
    // It must not dump the raw payload at the user.
    expect(alert.textContent).not.toContain('"detail"')
  })

  it('surfaces a 5xx using the message the backend supplied', async () => {
    const user = userEvent.setup()
    mockFetch(jsonResponse({ detail: 'the numerical core could not complete the optimization' }, 500))
    render(<App />)

    await user.click(screen.getByRole('button', { name: /optimize/i }))

    expect(await screen.findByRole('alert')).toHaveTextContent(
      /numerical core could not complete/i,
    )
  })

  it('hints at a stopped backend when a 5xx carries no message', async () => {
    // What the Vite dev proxy returns when uvicorn is not running.
    const user = userEvent.setup()
    mockFetch(new Response('', { status: 500 }))
    render(<App />)

    await user.click(screen.getByRole('button', { name: /optimize/i }))

    expect(await screen.findByRole('alert')).toHaveTextContent(/is the api running/i)
  })

  it('reports an unreachable backend as a network problem', async () => {
    const user = userEvent.setup()
    const fetchMock = vi.fn().mockRejectedValue(new TypeError('Failed to fetch'))
    vi.stubGlobal('fetch', fetchMock)
    render(<App />)

    await user.click(screen.getByRole('button', { name: /optimize/i }))

    expect(await screen.findByRole('alert')).toHaveTextContent(/could not reach/i)
  })

  it('keeps the previous result visible when a later run fails', async () => {
    const user = userEvent.setup()
    mockFetch(
      jsonResponse(makeResponse()),
      jsonResponse({ detail: 'boom' }, 500),
    )
    render(<App />)

    const button = screen.getByRole('button', { name: /optimize/i })
    await user.click(button)
    await screen.findByTestId('convergence-status')
    expect(screen.getByText('250.460')).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: /optimize/i }))
    await screen.findByRole('alert')

    expect(screen.getByTestId('convergence-status')).toBeInTheDocument()
    expect(screen.getByText('250.460')).toBeInTheDocument()
    expect(screen.getByTestId('topology-canvas')).toBeInTheDocument()
  })
})

describe('accessibility', () => {
  it('labels every parameter input', () => {
    render(<App />)

    for (const label of [
      /elements x/i,
      /elements y/i,
      /volume fraction/i,
      /simp penalty/i,
      /filter radius/i,
      /max iterations/i,
      /convergence tolerance/i,
    ]) {
      expect(screen.getByLabelText(label)).toBeInstanceOf(HTMLInputElement)
    }
  })

  it('exposes the primary action as a submit button', () => {
    render(<App />)

    expect(screen.getByRole('button', { name: /optimize/i })).toHaveAttribute(
      'type',
      'submit',
    )
  })

  it('announces status changes through a live region', () => {
    render(<App />)

    expect(screen.getByRole('status')).toHaveAttribute('aria-live', 'polite')
  })

  it('gives the density canvas an accessible description once populated', async () => {
    const user = userEvent.setup()
    mockFetch(jsonResponse(makeResponse()))
    render(<App />)

    await user.click(screen.getByRole('button', { name: /optimize/i }))

    const canvas = await screen.findByRole('img', { name: /density field/i })
    expect(canvas).toHaveAttribute('data-nelx', '4')
    expect(canvas).toHaveAttribute('data-nely', '2')
  })
})
