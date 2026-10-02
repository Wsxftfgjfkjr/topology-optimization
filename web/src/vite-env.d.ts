/// <reference types="vite/client" />

/**
 * Environment variables this app understands.
 *
 * Declared explicitly so `import.meta.env.VITE_API_BASE` is `string | undefined`
 * rather than `any`.
 */
interface ImportMetaEnv {
  /** Optional absolute base for the backend; empty in development, where the
   *  Vite dev server proxies the relative `/api` path instead. */
  readonly VITE_API_BASE?: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}
