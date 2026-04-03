import React, { useEffect, useState } from 'react'
import { useEngine } from '../context/useEngine'

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface SchemaSymbol {
  name: string
  type: 'relation' | 'function' | 'probabilistic'
  params: string[]
  docstring?: string | null
}

interface SchemaData {
  relations: SchemaSymbol[]
  functions: SchemaSymbol[]
  probabilistic: SchemaSymbol[]
}

interface SchemaResponse {
  status: string
  data: SchemaData
}

export interface PredicateBrowserProps {
  /** Called when the user clicks a symbol item. */
  onSelect?: (symbol: SchemaSymbol) => void
  className?: string
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Format a symbol as "Name(param1, param2, ...)" */
function formatSymbol(sym: SchemaSymbol): string {
  if (sym.params.length === 0) {
    return sym.name
  }
  return `${sym.name}(${sym.params.join(', ')})`
}

// ---------------------------------------------------------------------------
// CollapsibleGroup sub-component
// ---------------------------------------------------------------------------

interface CollapsibleGroupProps {
  title: string
  colorClass: string
  symbols: SchemaSymbol[]
  onSelect?: (symbol: SchemaSymbol) => void
}

function CollapsibleGroup({
  title,
  colorClass,
  symbols,
  onSelect,
}: CollapsibleGroupProps): React.ReactElement {
  const [open, setOpen] = useState(true)

  return (
    <div className={`predicate-group predicate-group--${colorClass}`}>
      <button
        className="predicate-group-header"
        onClick={() => setOpen((prev) => !prev)}
        aria-expanded={open}
      >
        <span className="predicate-group-title">{title}</span>
        <span className="predicate-group-count">{symbols.length}</span>
        <span
          className={`predicate-group-chevron ${open ? 'predicate-group-chevron--open' : ''}`}
          aria-hidden="true"
        >
          ▸
        </span>
      </button>
      {open && (
        <ul className="predicate-symbol-list" role="list">
          {symbols.map((sym) => (
            <li
              key={sym.name}
              className={`predicate-symbol-item predicate-symbol-item--${colorClass}`}
              role="button"
              tabIndex={0}
              title={sym.docstring ?? sym.name}
              onClick={() => onSelect?.(sym)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                  e.preventDefault()
                  onSelect?.(sym)
                }
              }}
            >
              {formatSymbol(sym)}
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// PredicateBrowser component
// ---------------------------------------------------------------------------

function PredicateBrowser({
  onSelect,
  className,
}: PredicateBrowserProps): React.ReactElement {
  const { selectedEngine } = useEngine()
  const [schema, setSchema] = useState<SchemaData | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [search, setSearch] = useState('')

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

    fetch(`/v2/schema/${encodeURIComponent(selectedEngine)}`)
      .then((res) => {
        if (!res.ok) {
          throw new Error(
            `Failed to fetch schema for ${selectedEngine}: ${res.status}`,
          )
        }
        return res.json() as Promise<SchemaResponse>
      })
      .then((body) => {
        setSchema(body.data)
        setLoading(false)
      })
      .catch((err: unknown) => {
        const message =
          err instanceof Error ? err.message : 'Unknown error fetching schema'
        setError(message)
        setLoading(false)
      })
  }, [selectedEngine])

  if (!selectedEngine) {
    return (
      <div className={`predicate-browser ${className ?? ''}`}>
        <p className="predicate-browser-placeholder">Engine not selected</p>
      </div>
    )
  }

  if (loading) {
    return (
      <div className={`predicate-browser ${className ?? ''}`}>
        <p className="predicate-browser-loading">Loading predicates...</p>
      </div>
    )
  }

  if (error) {
    return (
      <div className={`predicate-browser ${className ?? ''}`}>
        <p className="predicate-browser-error">{error}</p>
      </div>
    )
  }

  if (!schema) {
    return (
      <div className={`predicate-browser ${className ?? ''}`}>
        <p className="predicate-browser-placeholder">No schema available</p>
      </div>
    )
  }

  const query = search.toLowerCase()

  const filteredRelations = schema.relations.filter((s) =>
    s.name.toLowerCase().includes(query),
  )
  const filteredFunctions = schema.functions.filter((s) =>
    s.name.toLowerCase().includes(query),
  )
  const filteredProbabilistic = schema.probabilistic.filter((s) =>
    s.name.toLowerCase().includes(query),
  )

  const totalCount =
    filteredRelations.length +
    filteredFunctions.length +
    filteredProbabilistic.length

  return (
    <div className={`predicate-browser ${className ?? ''}`}>
      <input
        className="predicate-search"
        type="text"
        placeholder="Search predicates…"
        value={search}
        onChange={(e) => setSearch(e.target.value)}
        aria-label="Search predicates"
      />

      {totalCount === 0 ? (
        <p className="predicate-browser-empty">
          {search
            ? `No predicates match "${search}"`
            : 'No predicates available'}
        </p>
      ) : (
        <>
          {filteredRelations.length > 0 && (
            <CollapsibleGroup
              title="Relations"
              colorClass="blue"
              symbols={filteredRelations}
              onSelect={onSelect}
            />
          )}
          {filteredFunctions.length > 0 && (
            <CollapsibleGroup
              title="Functions"
              colorClass="green"
              symbols={filteredFunctions}
              onSelect={onSelect}
            />
          )}
          {filteredProbabilistic.length > 0 && (
            <CollapsibleGroup
              title="Probabilistic"
              colorClass="orange"
              symbols={filteredProbabilistic}
              onSelect={onSelect}
            />
          )}
        </>
      )}
    </div>
  )
}

export default PredicateBrowser
