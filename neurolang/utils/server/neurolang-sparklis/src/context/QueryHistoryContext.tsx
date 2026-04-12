/**
 * QueryHistoryContext.tsx
 *
 * React context that manages query execution history:
 *   - Persists history to localStorage (key: QUERY_HISTORY_KEY)
 *   - Max 50 entries, FIFO eviction when full
 *   - Each entry: { query, engine, timestamp, resultSummary }
 *
 * React context pattern: provider in this file, hook in useQueryHistory.ts
 * (satisfying the react-refresh lint rule about mixed exports).
 */
import React, { createContext, useCallback, useEffect, useState } from 'react'

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/** localStorage key for persisted query history. */
export const QUERY_HISTORY_KEY = 'neurolang_query_history'

/** Maximum number of history entries to store. */
export const QUERY_HISTORY_MAX = 50

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/** A single entry in the query history. */
export interface HistoryEntry {
  /** The Datalog query string. */
  query: string
  /** The engine used (e.g., "neurosynth", "destrieux"). */
  engine: string
  /** ISO-8601 timestamp of when the query was executed. */
  timestamp: string
  /** Short summary of results (e.g., "42 rows"). */
  resultSummary: string
}

/** Input for adding a new entry (timestamp is auto-generated). */
export interface AddHistoryEntryInput {
  query: string
  engine: string
  resultSummary: string
}

/** The value exposed by QueryHistoryContext. */
export interface QueryHistoryContextValue {
  /** Ordered history entries (most recent first). */
  entries: HistoryEntry[]
  /** Add a new entry to the history. */
  addEntry: (input: AddHistoryEntryInput) => void
  /** Clear all history entries. */
  clearHistory: () => void
}

// ---------------------------------------------------------------------------
// Context
// ---------------------------------------------------------------------------

const QueryHistoryContext = createContext<QueryHistoryContextValue | null>(null)

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Load history from localStorage, returning an empty array on parse error. */
function loadHistory(): HistoryEntry[] {
  try {
    const stored = localStorage.getItem(QUERY_HISTORY_KEY)
    if (!stored) return []
    const parsed = JSON.parse(stored) as unknown
    if (!Array.isArray(parsed)) return []
    return parsed as HistoryEntry[]
  } catch {
    return []
  }
}

/** Persist history to localStorage. */
function saveHistory(entries: HistoryEntry[]): void {
  try {
    localStorage.setItem(QUERY_HISTORY_KEY, JSON.stringify(entries))
  } catch {
    // Storage may be full or unavailable — silently ignore.
  }
}

// ---------------------------------------------------------------------------
// Provider
// ---------------------------------------------------------------------------

export function QueryHistoryProvider({
  children,
}: {
  children: React.ReactNode
}): React.ReactElement {
  // Load initial state from localStorage on mount.
  const [entries, setEntries] = useState<HistoryEntry[]>(() => loadHistory())

  // Persist to localStorage whenever entries change.
  useEffect(() => {
    saveHistory(entries)
  }, [entries])

  const addEntry = useCallback((input: AddHistoryEntryInput): void => {
    const newEntry: HistoryEntry = {
      query: input.query,
      engine: input.engine,
      timestamp: new Date().toISOString(),
      resultSummary: input.resultSummary,
    }

    setEntries((prev) => {
      // Prepend the new entry (most recent first), then cap at MAX.
      const updated = [newEntry, ...prev]
      if (updated.length > QUERY_HISTORY_MAX) {
        return updated.slice(0, QUERY_HISTORY_MAX)
      }
      return updated
    })
  }, [])

  const clearHistory = useCallback((): void => {
    setEntries([])
  }, [])

  const value: QueryHistoryContextValue = {
    entries,
    addEntry,
    clearHistory,
  }

  return (
    <QueryHistoryContext.Provider value={value}>
      {children}
    </QueryHistoryContext.Provider>
  )
}

export default QueryHistoryContext
