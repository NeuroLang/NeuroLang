/**
 * SchemaContext.tsx
 *
 * React context that exposes the schema data (list of symbols) for the
 * currently selected engine.  Consumers can use this to look up a predicate
 * by name and retrieve its parameter list – useful for adding predicates with
 * proper placeholder variables from outside the PredicateBrowser (e.g. when
 * the user clicks a suggestion chip).
 */
import React, { createContext, useCallback, useEffect, useState } from 'react'
import { useEngine } from './useEngine'

// ---------------------------------------------------------------------------
// Types (mirrored from PredicateBrowser to avoid a circular import)
// ---------------------------------------------------------------------------

export interface SchemaSymbol {
  name: string
  type: 'relation' | 'function' | 'probabilistic'
  params: string[]
  docstring?: string | null
}

export interface SchemaData {
  relations: SchemaSymbol[]
  functions: SchemaSymbol[]
  probabilistic: SchemaSymbol[]
}

// ---------------------------------------------------------------------------
// Context value
// ---------------------------------------------------------------------------

export interface SchemaContextValue {
  /** The loaded schema data, or null when not yet loaded. */
  schema: SchemaData | null
  /** Whether the schema is being loaded. */
  loading: boolean
  /** Error message if the last fetch failed. */
  error: string | null
  /**
   * Look up a symbol by name across all categories.
   * Returns null if the symbol is not found.
   */
  lookupSymbol: (name: string) => SchemaSymbol | null
}

// ---------------------------------------------------------------------------
// Context
// ---------------------------------------------------------------------------

const SchemaContext = createContext<SchemaContextValue | null>(null)

// ---------------------------------------------------------------------------
// Provider
// ---------------------------------------------------------------------------

export function SchemaProvider({
  children,
}: {
  children: React.ReactNode
}): React.ReactElement {
  const { selectedEngine } = useEngine()
  const [schema, setSchema] = useState<SchemaData | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!selectedEngine) {
      setSchema(null)
      setLoading(false)
      setError(null)
      return
    }

    setLoading(true)
    setError(null)
    setSchema(null)

    const controller = new AbortController()

    fetch(`/v2/schema/${encodeURIComponent(selectedEngine)}`, {
      signal: controller.signal,
    })
      .then((res) => {
        if (!res.ok) {
          throw new Error(
            `Failed to fetch schema for ${selectedEngine}: ${res.status}`,
          )
        }
        return res.json() as Promise<{ status: string; data: SchemaData }>
      })
      .then((body) => {
        setSchema(body.data)
        setLoading(false)
      })
      .catch((err: unknown) => {
        if (err instanceof DOMException && err.name === 'AbortError') return
        const message =
          err instanceof Error ? err.message : 'Unknown error fetching schema'
        setError(message)
        setLoading(false)
      })

    return () => controller.abort()
  }, [selectedEngine])

  const lookupSymbol = useCallback(
    (name: string): SchemaSymbol | null => {
      if (!schema) return null
      return (
        schema.relations.find((s) => s.name === name) ??
        schema.functions.find((s) => s.name === name) ??
        schema.probabilistic.find((s) => s.name === name) ??
        null
      )
    },
    [schema],
  )

  const value: SchemaContextValue = {
    schema,
    loading,
    error,
    lookupSymbol,
  }

  return (
    <SchemaContext.Provider value={value}>{children}</SchemaContext.Provider>
  )
}

export default SchemaContext
