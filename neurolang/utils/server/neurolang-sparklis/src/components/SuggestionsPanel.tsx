/**
 * SuggestionsPanel.tsx
 *
 * Displays context-aware suggestions below the VisualQueryBuilder.
 *
 * Behaviour:
 *  - Watches the current QueryModel state; whenever it changes, POSTs the
 *    serialised Datalog to POST /v2/suggest/:engine (debounced 300 ms).
 *  - Renders the returned suggestions as clickable chips grouped by category
 *    (Identifiers, Operators, Signs, …).
 *  - Fires the `onSuggestionSelect` callback when a chip is clicked.
 *  - Shows a loading spinner while suggestions are being fetched.
 *  - Shows an empty-state message when there are no suggestions.
 */
import React, { useCallback, useEffect, useRef, useState } from 'react'
import { useEngine } from '../context/useEngine'
import { useQuery } from '../context/useQuery'
import { serializeToDatalog } from '../models/QueryModel'

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/** The raw response shape from POST /v2/suggest/:engine. */
export interface SuggestionsResponse {
  status: string
  data: Record<string, string[]>
  /** May be present when status === "error" */
  message?: string
}

/** Props for the SuggestionsPanel component. */
export interface SuggestionsPanelProps {
  /**
   * Called when a suggestion chip is clicked.
   * Receives the suggestion text and category name.
   */
  onSuggestionSelect?: (suggestion: string, category: string) => void
  className?: string
  /**
   * Debounce delay in milliseconds before fetching suggestions.
   * Defaults to 300ms. Can be set to 0 in tests for immediate fetching.
   */
  debounceMs?: number
}

// ---------------------------------------------------------------------------
// Category ordering (most useful first)
// ---------------------------------------------------------------------------

const CATEGORY_ORDER = [
  'Identifiers',
  'Operators',
  'Signs',
  'Reserved words',
  'Numbers',
  'Text',
  'Functions',
  'Cmd_identifier',
  'Boleans',
  'Expression symbols',
  'Python string',
  'Strings',
  'commands',
  'functions',
  'base symbols',
  'query symbols',
  'Identifier_regexp',
]

/**
 * Sort category entries using CATEGORY_ORDER.
 * Unknown categories are placed after the known ones, in their original order.
 */
function sortedCategories(
  data: Record<string, string[]>,
): Array<[string, string[]]> {
  const entries = Object.entries(data).filter(([, vals]) => vals.length > 0)
  const known = CATEGORY_ORDER.filter((cat) => data[cat]?.length)
  const unknown = entries
    .map(([cat]) => cat)
    .filter((cat) => !CATEGORY_ORDER.includes(cat))
  return [...known, ...unknown].map((cat) => [cat, data[cat]])
}

// ---------------------------------------------------------------------------
// SuggestionChip sub-component
// ---------------------------------------------------------------------------

interface SuggestionChipProps {
  text: string
  category: string
  onSelect: (text: string, category: string) => void
}

function SuggestionChip({
  text,
  category,
  onSelect,
}: SuggestionChipProps): React.ReactElement {
  return (
    <button
      className="suggestions-chip"
      onClick={() => onSelect(text, category)}
      title={`${category}: ${text}`}
      aria-label={`Suggestion: ${text}`}
    >
      {text}
    </button>
  )
}

// ---------------------------------------------------------------------------
// CategoryGroup sub-component
// ---------------------------------------------------------------------------

interface CategoryGroupProps {
  category: string
  suggestions: string[]
  onSelect: (text: string, category: string) => void
}

function CategoryGroup({
  category,
  suggestions,
  onSelect,
}: CategoryGroupProps): React.ReactElement {
  return (
    <div className="suggestions-category-group">
      <span className="suggestions-category-label">{category}</span>
      <div className="suggestions-chip-list" role="list">
        {suggestions.map((s) => (
          <SuggestionChip
            key={s}
            text={s}
            category={category}
            onSelect={onSelect}
          />
        ))}
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// LoadingSpinner sub-component
// ---------------------------------------------------------------------------

function LoadingSpinner(): React.ReactElement {
  return (
    <div className="suggestions-loading" aria-label="Loading suggestions">
      <span className="suggestions-spinner" aria-hidden="true" />
      <span>Loading suggestions…</span>
    </div>
  )
}

// ---------------------------------------------------------------------------
// SuggestionsPanel
// ---------------------------------------------------------------------------

function SuggestionsPanel({
  onSuggestionSelect,
  className,
  debounceMs = 300,
}: SuggestionsPanelProps): React.ReactElement {
  const { selectedEngine } = useEngine()
  const { state } = useQuery()

  const [suggestions, setSuggestions] = useState<Record<string, string[]>>({})
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // We keep an AbortController ref so we can cancel in-flight requests.
  const abortRef = useRef<AbortController | null>(null)
  // Timer ref for debouncing.
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  /** Fetch suggestions from the backend for the given program. */
  const fetchSuggestions = useCallback(
    async (engine: string, program: string) => {
      // Cancel any previous in-flight request.
      if (abortRef.current) {
        abortRef.current.abort()
      }
      const controller = new AbortController()
      abortRef.current = controller

      setLoading(true)
      setError(null)

      try {
        const res = await fetch(
          `/v2/suggest/${encodeURIComponent(engine)}`,
          {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ program }),
            signal: controller.signal,
          },
        )

        if (!res.ok) {
          throw new Error(`Suggestions request failed: ${res.status}`)
        }

        const body = (await res.json()) as SuggestionsResponse

        if (body.status === 'error') {
          setError(body.message ?? 'Error fetching suggestions')
          setSuggestions({})
        } else {
          setSuggestions((body.data as Record<string, string[]>) ?? {})
        }
      } catch (err: unknown) {
        if (err instanceof DOMException && err.name === 'AbortError') {
          // Request was cancelled – do not update state.
          return
        }
        const message =
          err instanceof Error ? err.message : 'Unknown error'
        setError(message)
        setSuggestions({})
      } finally {
        setLoading(false)
      }
    },
    [],
  )

  // Whenever the query state or selected engine changes, debounce a new
  // suggestions request.
  useEffect(() => {
    if (!selectedEngine) {
      setSuggestions({})
      setError(null)
      setLoading(false)
      return
    }

    const program = serializeToDatalog(state)

    // Clear any pending debounce timer.
    if (timerRef.current !== null) {
      clearTimeout(timerRef.current)
    }

    timerRef.current = setTimeout(() => {
      void fetchSuggestions(selectedEngine, program)
    }, debounceMs)

    return () => {
      if (timerRef.current !== null) {
        clearTimeout(timerRef.current)
      }
    }
  }, [state, selectedEngine, fetchSuggestions, debounceMs])

  // Cleanup on unmount.
  useEffect(() => {
    return () => {
      abortRef.current?.abort()
      if (timerRef.current !== null) clearTimeout(timerRef.current)
    }
  }, [])

  const handleSelect = useCallback(
    (text: string, category: string) => {
      onSuggestionSelect?.(text, category)
    },
    [onSuggestionSelect],
  )

  const categories = sortedCategories(suggestions)
  const hasSuggestions = categories.length > 0

  return (
    <div
      className={`suggestions-panel${className ? ` ${className}` : ''}`}
      aria-label="Suggestions panel"
    >
      {/* Panel header */}
      <div className="suggestions-panel-header">
        <span className="suggestions-panel-title">Suggestions</span>
        {loading && <span className="suggestions-panel-status">updating…</span>}
      </div>

      {/* Content */}
      <div className="suggestions-panel-body">
        {loading && !hasSuggestions ? (
          <LoadingSpinner />
        ) : error ? (
          <p className="suggestions-error">{error}</p>
        ) : !selectedEngine ? (
          <p className="suggestions-empty">Select an engine to see suggestions.</p>
        ) : !hasSuggestions ? (
          <p className="suggestions-empty">No suggestions available.</p>
        ) : (
          <div className="suggestions-categories">
            {categories.map(([cat, items]) => (
              <CategoryGroup
                key={cat}
                category={cat}
                suggestions={items}
                onSelect={handleSelect}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

export default SuggestionsPanel
