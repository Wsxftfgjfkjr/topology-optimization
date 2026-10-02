import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { cleanup, render, screen } from '@testing-library/react'
import { TopologyView } from './TopologyView'
import { makeResponse } from '../test/fixtures'

/**
 * jsdom has no canvas backend, so the 2D context is replaced with a recorder.
 * That lets these tests assert the geometry the component actually paints,
 * which is where the density orientation lives.
 */

interface Fill {
  style: string
  x: number
  y: number
  w: number
  h: number
}

let fills: Fill[] = []

beforeEach(() => {
  fills = []
  const context = {
    setTransform: vi.fn(),
    clearRect: vi.fn(),
    fillRect(x: number, y: number, w: number, h: number) {
      fills.push({ style: String(this.fillStyle), x, y, w, h })
    },
    fillStyle: '',
  }
  vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(
    context as unknown as CanvasRenderingContext2D,
  )
  vi.stubGlobal(
    'ResizeObserver',
    class {
      observe() {}
      disconnect() {}
    },
  )
})

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

describe('density orientation', () => {
  it('draws row 0 at the bottom, matching imshow(origin="lower")', () => {
    // 2 rows so the vertical order is unambiguous.
    const density = [
      [1, 1], // row 0 — must end up at the bottom
      [0, 0], // row 1 — must end up at the top
    ]
    render(<TopologyView result={makeResponse({ nelx: 2, nely: 2, density })} running={false} />)

    expect(fills).toHaveLength(4)

    // jsdom reports clientWidth 0, so the component falls back to nelx = 2
    // columns across 2 rows: one unit per cell, drawn top-down in canvas space.
    const topRow = fills.filter((f) => f.y === 0)
    const bottomRow = fills.filter((f) => f.y === 1)
    expect(topRow).toHaveLength(2)
    expect(bottomRow).toHaveLength(2)

    // The solid row (density 1) is painted at the bottom of the canvas.
    expect(bottomRow.every((f) => f.style === SOLID)).toBe(true)
    // The void row (density 0) is painted at the top.
    expect(topRow.every((f) => f.style === VOID)).toBe(true)
  })

  it('draws columns left to right without mirroring', () => {
    const density = [
      [1, 0, 0], // solid only in the first column
      [1, 0, 0],
    ]
    render(<TopologyView result={makeResponse({ nelx: 3, nely: 2, density })} running={false} />)

    const firstColumn = fills.filter((f) => f.x === 0)
    const lastColumn = fills.filter((f) => f.x === 2)
    expect(firstColumn.every((f) => f.style === SOLID)).toBe(true)
    expect(lastColumn.every((f) => f.style === VOID)).toBe(true)
  })

  it('paints one rectangle per element, with no interpolation', () => {
    render(
      <TopologyView result={makeResponse({ nelx: 6, nely: 4 })} running={false} />,
    )

    expect(fills).toHaveLength(6 * 4)
  })
})

describe('the panel around it', () => {
  it('reports the grid dimensions it was given', () => {
    render(<TopologyView result={makeResponse({ nelx: 5, nely: 3 })} running={false} />)

    const canvas = screen.getByTestId('topology-canvas')
    expect(canvas).toHaveAttribute('data-nelx', '5')
    expect(canvas).toHaveAttribute('data-nely', '3')
    expect(screen.getByText(/5 × 3/)).toBeInTheDocument()
  })

  it('shows an empty state instead of a fabricated field', () => {
    render(<TopologyView result={null} running={false} />)

    expect(screen.getByTestId('topology-empty')).toBeInTheDocument()
    expect(screen.queryByTestId('topology-canvas')).not.toBeInTheDocument()
  })
})

const VOID = 'rgb(205, 226, 251)'
const SOLID = 'rgb(13, 54, 107)'
