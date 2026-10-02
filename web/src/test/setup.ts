import '@testing-library/jest-dom/vitest'

/**
 * jsdom implements neither of these, and the components use both:
 * `ResizeObserver` to repaint the density canvas on layout changes, and a 2D
 * canvas context to draw into. Stubbing them here keeps every test free of the
 * "Not implemented" noise jsdom would otherwise print, and lets the canvas
 * tests install their own recording context on top.
 */
if (!('ResizeObserver' in globalThis)) {
  globalThis.ResizeObserver = class ResizeObserver {
    observe(): void {}
    unobserve(): void {}
    disconnect(): void {}
  } as unknown as typeof globalThis.ResizeObserver
}

const noop = (): void => {}

const stubContext = {
  setTransform: noop,
  clearRect: noop,
  fillRect: noop,
  fillStyle: '',
}

if (typeof HTMLCanvasElement !== 'undefined') {
  // Assigned through defineProperty rather than to the prototype property
  // directly: `getContext` is an overloaded signature, so a plain assignment
  // would need a cast that says nothing useful.
  Object.defineProperty(HTMLCanvasElement.prototype, 'getContext', {
    configurable: true,
    writable: true,
    value: (contextId: string) => (contextId === '2d' ? stubContext : null),
  })
}
