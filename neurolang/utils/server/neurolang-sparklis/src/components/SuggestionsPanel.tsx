/**
 * SuggestionsPanel.tsx
 *
 * Displays context-aware suggestions below the VisualQueryBuilder.
 *
 * Behaviour:
 *  - Watches the current QueryModel state; whenever it changes, POSTs the
 *    serialised Datalog to POST /v2/suggest/:engine (debounced 300 ms).
 *  - Renders the returned suggestions as clickable chips grouped by category.
 *    Groups are ordered: Predicates first, Operators second, Other last.
 *  - Includes a search/filter input above the chips for substring filtering.
 *  - Limits displayed suggestions to 20 per category with a "Show more" button.
 *  - Supports keyboard navigation: ArrowDown/ArrowUp moves between chips,
 *    ArrowDown from the search input focuses the first chip, ArrowUp from
 *    the first chip returns focus to the search input.
 *  - Fires the `onSuggestionSelect` callback when a chip is clicked.
 *  - Shows a loading spinner while suggestions are being fetched.
 *  - Shows an empty-state message when there are no suggestions.
 */
import React, {
  useCallback,
  useEffect,
  useRef,
  useState,
} from 'react'
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
   * Receives the suggestion text and group name (Predicates/Operators/Other).
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
// Constants
// ---------------------------------------------------------------------------

/** Max suggestions shown per group before the "Show more" button appears. */
const PAGE_SIZE = 20

// ---------------------------------------------------------------------------
// Group mapping
// ---------------------------------------------------------------------------

/**
 * Map raw API category names to display group names.
 * - "Predicates" group: Identifiers, Cmd_identifier, Identifier_regexp, base symbols,
 *   query symbols, functions, Functions
 * - "Operators" group: Operators, Signs, Reserved words, commands
 * - "Other" group: everything else
 */
const PREDICATE_CATEGORIES = new Set([
  'Identifiers',
  'Cmd_identifier',
  'Identifier_regexp',
  'base symbols',
  'query symbols',
  'functions',
  'Functions',
])

const OPERATOR_CATEGORIES = new Set([
  'Operators',
  'Signs',
  'Reserved words',
  'commands',
])

function getGroupName(category: string): 'Predicates' | 'Operators' | 'Other' {
  if (PREDICATE_CATEGORIES.has(category)) return 'Predicates'
  if (OPERATOR_CATEGORIES.has(category)) return 'Operators'
  return 'Other'
}

/** Represents a display group aggregating one or more API categories. */
interface SuggestionGroup {
  group: 'Predicates' | 'Operators' | 'Other'
  items: string[]
}

const GROUP_ORDER: Array<'Predicates' | 'Operators' | 'Other'> = [
  'Predicates',
  'Operators',
  'Other',
]

/**
 * Merge raw API categories into the three display groups and return them in
 * the canonical order (Predicates → Operators → Other).
 * Only includes groups that have at least one item.
 */
function buildGroups(data: Record<string, string[]>): SuggestionGroup[] {
  const groupMap = new Map<'Predicates' | 'Operators' | 'Other', string[]>()

  for (const [cat, vals] of Object.entries(data)) {
    if (!Array.isArray(vals) || vals.length === 0) continue
    const group = getGroupName(cat)
    if (!groupMap.has(group)) {
      groupMap.set(group, [])
    }
    groupMap.get(group)!.push(...vals)
  }

  return GROUP_ORDER.filter((g) => groupMap.has(g)).map((g) => ({
    group: g,
    items: groupMap.get(g)!,
  }))
}

// ---------------------------------------------------------------------------
// CategoryGroup sub-component
// ---------------------------------------------------------------------------

interface CategoryGroupProps {
  group: 'Predicates' | 'Operators' | 'Other'
  items: string[]
  onSelect: (text: string, group: string) => void
  /** Map from chip index (across all groups) to ref */
  chipRefs: React.MutableRefObject<(HTMLButtonElement | null)[]>
  /** Starting index for chips in this group */
  startIndex: number
  /** KeyDown handler forwarded to each chip */
  onChipKeyDown: (
    e: React.KeyboardEvent<HTMLButtonElement>,
    globalIdx: number,
  ) => void
}

function CategoryGroup({
  group,
  items,
  onSelect,
  chipRefs,
  startIndex,
  onChipKeyDown,
}: CategoryGroupProps): React.ReactElement {
  const [expanded, setExpanded] = useState(false)

  const displayedItems = expanded ? items : items.slice(0, PAGE_SIZE)
  const hasMore = !expanded && items.length > PAGE_SIZE
  const hiddenCount = items.length - PAGE_SIZE

  return (
    <div className="suggestions-category-group">
      <span className="suggestions-category-label">{group}</span>
      <div className="suggestions-chip-list" role="list">
        {displayedItems.map((s, localIdx) => {
          const globalIdx = startIndex + localIdx
          const assignRef = (el: HTMLButtonElement | null) => {
            chipRefs.current[globalIdx] = el
          }
          return (
            <button
              key={s}
              ref={assignRef}
              className="suggestions-chip"
              onClick={() => onSelect(s, group)}
              onKeyDown={(e) => onChipKeyDown(e, globalIdx)}
              title={`${group}: ${s}`}
              aria-label={`Suggestion: ${s}`}
            >
              {s}
            </button>
          )
        })}
        {hasMore && (
          <button
            className="suggestions-show-more"
            onClick={() => setExpanded(true)}
          >
            Show more ({hiddenCount} more)
          </button>
        )}
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
  const [filter, setFilter] = useState('')

  // We keep an AbortController ref so we can cancel in-flight requests.
  const abortRef = useRef<AbortController | null>(null)
  // Timer ref for debouncing.
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  // Ref to the search input for keyboard navigation
  const searchInputRef = useRef<HTMLInputElement | null>(null)
  // Flat array of chip DOM refs for keyboard navigation
  const chipRefs = useRef<(HTMLButtonElement | null)[]>([])

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
          // Reset filter when new suggestions arrive
          setFilter('')
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

    // Serialize the current query.  For a non-empty query, strip the
    // trailing "." and append "," so that the autocompletion engine
    // sees the cursor positioned after a body predicate (i.e. ready to
    // add the next one).  This causes the engine to suggest valid body
    // predicates rather than returning empty results.
    const rawProgram = serializeToDatalog(state)
    let program = rawProgram
    if (program.endsWith('.')) {
      // Strip trailing period and add comma to signal "body extendable"
      program = program.slice(0, -1) + ','
    }

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
    (text: string, group: string) => {
      onSuggestionSelect?.(text, group)
    },
    [onSuggestionSelect],
  )

  // Build display groups
  const allGroups = buildGroups(suggestions)

  // Apply filter: substring match (case-insensitive) across each group's items
  const filterLower = filter.toLowerCase()
  const filteredGroups = filterLower
    ? allGroups
        .map((g) => ({
          ...g,
          items: g.items.filter((item) =>
            item.toLowerCase().includes(filterLower),
          ),
        }))
        .filter((g) => g.items.length > 0)
    : allGroups

  const hasSuggestions = filteredGroups.length > 0

  // Build flat chip list for keyboard navigation (only visible items)
  // Each group's displayed items (up to PAGE_SIZE unless expanded)
  // We need to count visible chips across all groups.
  // CategoryGroup manages "expand" state internally; for keyboard nav purposes
  // we track all items - keyboard nav will work with whatever is rendered.
  // We use chipRefs array and reset its length before render.
  const totalVisibleChips = filteredGroups.reduce(
    (sum, g) => sum + Math.min(g.items.length, PAGE_SIZE),
    0,
  )
  chipRefs.current = new Array(totalVisibleChips).fill(null)

  // Keyboard navigation handler for search input
  const handleSearchKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLInputElement>) => {
      if (e.key === 'ArrowDown') {
        e.preventDefault()
        const firstChip = chipRefs.current[0]
        if (firstChip) {
          firstChip.focus()
        }
      }
    },
    [],
  )

  // Keyboard navigation handler for chips
  const handleChipKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLButtonElement>, globalIdx: number) => {
      if (e.key === 'ArrowDown') {
        e.preventDefault()
        const nextChip = chipRefs.current[globalIdx + 1]
        if (nextChip) {
          nextChip.focus()
        } else {
          // Wrap back to search input
          searchInputRef.current?.focus()
        }
      } else if (e.key === 'ArrowUp') {
        e.preventDefault()
        if (globalIdx === 0) {
          // Return focus to search input
          searchInputRef.current?.focus()
        } else {
          const prevChip = chipRefs.current[globalIdx - 1]
          if (prevChip) {
            prevChip.focus()
          }
        }
      }
    },
    [],
  )

  // Compute per-group start indices for chip refs
  let chipOffset = 0
  const groupsWithOffset = filteredGroups.map((g) => {
    const start = chipOffset
    chipOffset += Math.min(g.items.length, PAGE_SIZE)
    return { ...g, startIndex: start }
  })

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
        {loading && !hasSuggestions && allGroups.length === 0 ? (
          <LoadingSpinner />
        ) : error ? (
          <p className="suggestions-error">{error}</p>
        ) : !selectedEngine ? (
          <p className="suggestions-empty">Select an engine to see suggestions.</p>
        ) : (
          <>
            {/* Search/filter input – shown whenever there's an engine selected */}
            {(hasSuggestions || filter) && (
              <input
                ref={searchInputRef}
                className="suggestions-filter-input"
                type="text"
                placeholder="Filter suggestions…"
                value={filter}
                onChange={(e) => setFilter(e.target.value)}
                onKeyDown={handleSearchKeyDown}
                aria-label="Filter suggestions"
              />
            )}

            {!hasSuggestions && filter ? (
              <p className="suggestions-empty">
                No suggestions match &ldquo;{filter}&rdquo;
              </p>
            ) : !hasSuggestions ? (
              <p className="suggestions-empty">No suggestions available.</p>
            ) : (
              <div className="suggestions-categories">
                {groupsWithOffset.map(({ group, items, startIndex }) => (
                  <CategoryGroup
                    key={group}
                    group={group}
                    items={items}
                    onSelect={handleSelect}
                    chipRefs={chipRefs}
                    startIndex={startIndex}
                    onChipKeyDown={handleChipKeyDown}
                  />
                ))}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  )
}

export default SuggestionsPanel
