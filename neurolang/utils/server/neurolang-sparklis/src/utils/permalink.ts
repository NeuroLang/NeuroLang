/**
 * permalink.ts
 *
 * Utility functions for generating and parsing shareable permalink URLs.
 *
 * URL format: <baseUrl>#/<engine>?q=<base64encodedquery>
 *
 * Example: http://localhost:3100/#/neurosynth?q=YW5zKHgpIDotLSBQZWFrUmVwb3J0ZWQoeCwgeSwgeiwgcyk=
 */

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/** The parsed result from a permalink hash. */
export interface PermalinkData {
  /** The NeuroLang engine name (e.g., "neurosynth", "destrieux"). */
  engine: string
  /** The decoded Datalog query string. */
  query: string
}

// ---------------------------------------------------------------------------
// buildPermalinkUrl
// ---------------------------------------------------------------------------

/**
 * Build a shareable permalink URL.
 *
 * @param baseUrl - The base URL of the app (e.g., "http://localhost:3100/")
 * @param engine  - The selected engine name (e.g., "neurosynth")
 * @param query   - The Datalog query string to encode
 * @returns       - The full URL with hash-encoded query
 */
export function buildPermalinkUrl(
  baseUrl: string,
  engine: string,
  query: string,
): string {
  const encoded = btoa(query)
  // Strip any existing hash from baseUrl
  const base = baseUrl.split('#')[0]
  return `${base}#/${engine}?q=${encoded}`
}

// ---------------------------------------------------------------------------
// parsePermalinkHash
// ---------------------------------------------------------------------------

/**
 * Parse a permalink hash string and return the engine + query.
 *
 * Expected format: `#/<engine>?q=<base64encodedquery>`
 *
 * Returns null if:
 *   - The hash is empty or doesn't start with `#/`
 *   - The engine portion is missing or empty
 *   - The `q` search param is missing
 *   - The base64 decoding fails
 *
 * @param hash - The full location.hash string (e.g., "#/neurosynth?q=...")
 * @returns PermalinkData or null if the hash is not a valid permalink
 */
export function parsePermalinkHash(hash: string): PermalinkData | null {
  if (!hash || !hash.startsWith('#/')) {
    return null
  }

  // Strip the leading '#/'
  const withoutHash = hash.slice(2) // e.g., "neurosynth?q=..."

  // Split on '?' to separate engine from search params
  const questionIdx = withoutHash.indexOf('?')
  if (questionIdx === -1) {
    // No query params at all
    return null
  }

  const engine = withoutHash.slice(0, questionIdx)
  if (!engine) {
    return null
  }

  const searchStr = withoutHash.slice(questionIdx + 1)
  const params = new URLSearchParams(searchStr)
  const encoded = params.get('q')
  if (!encoded) {
    return null
  }

  // Attempt base64 decode
  try {
    const query = atob(encoded)
    return { engine, query }
  } catch {
    // Invalid base64
    return null
  }
}
