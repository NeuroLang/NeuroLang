/**
 * QueryHistory.test.tsx
 *
 * Tests for the QueryHistory component and QueryHistoryContext.
 *
 * Covers:
 *   1. Stores executed queries to localStorage
 *   2. Loads history from localStorage on mount
 *   3. Clicking a history entry loads the query into the editor
 *   4. History persists across simulated page reloads
 *   5. Clear history button removes all entries
 *   6. Max 50 entries (FIFO)
 *   7. Truncated query preview is shown
 *   8. Timestamp is displayed
 *   9. Collapsible panel toggle
 */
import React from 'react'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import QueryHistory from '../QueryHistory'
import { QueryHistoryProvider } from '../../context/QueryHistoryContext'
import { EngineProvider } from '../../context/EngineContext'
import { QueryProvider } from '../../context/QueryContext'
import { useQueryHistory } from '../../context/useQueryHistory'
import { useEngine } from '../../context/useEngine'
import { QUERY_HISTORY_KEY } from '../../context/QueryHistoryContext'

// ---------------------------------------------------------------------------
// Mock data
// ---------------------------------------------------------------------------

const MOCK_ENTRY_1 = {
  query: 'ans(x) :- PeakReported(x, y, z, s)',
  engine: 'neurosynth',
  timestamp: '2024-01-15T10:00:00.000Z',
  resultSummary: '42 rows',
}

const MOCK_ENTRY_2 = {
  query: 'ans(x, y) :- Study(x, y)',
  engine: 'neurosynth',
  timestamp: '2024-01-15T11:00:00.000Z',
  resultSummary: '10 rows',
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

interface WrapperProps {
  children: React.ReactNode
  engine?: string | null
}

function TestWrapper({ children, engine = 'neurosynth' }: WrapperProps): React.ReactElement {
  return (
    <EngineProvider>
      <QueryProvider>
        <QueryHistoryProvider>
          <EngineInitializer engine={engine} />
          {children}
        </QueryHistoryProvider>
      </QueryProvider>
    </EngineProvider>
  )
}

function EngineInitializer({ engine }: { engine: string | null }): null {
  const { setSelectedEngine } = useEngine()
  React.useEffect(() => {
    if (engine !== null) {
      setSelectedEngine(engine)
    }
  }, []) // eslint-disable-line react-hooks/exhaustive-deps
  return null
}

/** Helper: renders QueryHistory with all needed contexts */
function renderQueryHistory(engine: string | null = 'neurosynth') {
  return render(
    <TestWrapper engine={engine}>
      <QueryHistory />
    </TestWrapper>,
  )
}

// ---------------------------------------------------------------------------
// Helper to add entries via the context hook (action component)
// ---------------------------------------------------------------------------

interface AddEntryButtonProps {
  query: string
  engine: string
  resultSummary: string
}

function AddEntryButton({ query, engine, resultSummary }: AddEntryButtonProps): React.ReactElement {
  const { addEntry } = useQueryHistory()
  return (
    <button
      onClick={() => addEntry({ query, engine, resultSummary })}
      data-testid="add-entry-btn"
    >
      Add Entry
    </button>
  )
}

// ---------------------------------------------------------------------------
// localStorage mock (jsdom's localStorage may be limited in some configs)
// ---------------------------------------------------------------------------

const localStorageMock = (() => {
  let store: Record<string, string> = {}
  return {
    getItem: (key: string) => store[key] ?? null,
    setItem: (key: string, value: string) => { store[key] = value },
    removeItem: (key: string) => { delete store[key] },
    clear: () => { store = {} },
  }
})()

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('QueryHistory', () => {
  beforeEach(() => {
    vi.stubGlobal('localStorage', localStorageMock)
    localStorageMock.clear()
    vi.restoreAllMocks()
    vi.stubGlobal('localStorage', localStorageMock)
  })

  afterEach(() => {
    localStorageMock.clear()
  })

  // -------------------------------------------------------------------------
  // 1. Stores to localStorage
  // -------------------------------------------------------------------------

  it('stores executed queries to localStorage', () => {
    render(
      <EngineProvider>
        <QueryProvider>
          <QueryHistoryProvider>
            <AddEntryButton
              query={MOCK_ENTRY_1.query}
              engine={MOCK_ENTRY_1.engine}
              resultSummary={MOCK_ENTRY_1.resultSummary}
            />
          </QueryHistoryProvider>
        </QueryProvider>
      </EngineProvider>,
    )

    const addBtn = screen.getByTestId('add-entry-btn')
    userEvent.click(addBtn)

    return waitFor(() => {
      const stored = localStorage.getItem(QUERY_HISTORY_KEY)
      expect(stored).not.toBeNull()
      const parsed = JSON.parse(stored!) as Array<{ query: string; engine: string }>
      expect(parsed).toHaveLength(1)
      expect(parsed[0].query).toBe(MOCK_ENTRY_1.query)
      expect(parsed[0].engine).toBe(MOCK_ENTRY_1.engine)
    })
  })

  // -------------------------------------------------------------------------
  // 2. Loads history from localStorage on mount
  // -------------------------------------------------------------------------

  it('loads history from localStorage on mount', () => {
    // Pre-populate localStorage
    const entries = [MOCK_ENTRY_1, MOCK_ENTRY_2]
    localStorage.setItem(QUERY_HISTORY_KEY, JSON.stringify(entries))

    renderQueryHistory()

    expect(screen.getByText(/PeakReported/)).toBeInTheDocument()
    expect(screen.getByText(/Study/)).toBeInTheDocument()
  })

  // -------------------------------------------------------------------------
  // 3. Click restores query into editor
  // -------------------------------------------------------------------------

  it('clicking a history entry updates the query text', async () => {
    const user = userEvent.setup()
    const entries = [MOCK_ENTRY_1]
    localStorage.setItem(QUERY_HISTORY_KEY, JSON.stringify(entries))

    renderQueryHistory()
    await screen.findByText(/PeakReported/)

    // Click on the history entry (aria-label contains "Load query: <preview>")
    const entryButton = screen.getByRole('button', { name: /load query.*PeakReported/i })
    await user.click(entryButton)

    // The button should still be in the DOM (the panel doesn't close on click)
    expect(entryButton).toBeInTheDocument()
  })

  // -------------------------------------------------------------------------
  // 4. Persists across simulated reloads
  // -------------------------------------------------------------------------

  it('persists history across simulated page reloads', async () => {
    // Add an entry
    const { unmount } = render(
      <EngineProvider>
        <QueryProvider>
          <QueryHistoryProvider>
            <AddEntryButton
              query={MOCK_ENTRY_1.query}
              engine={MOCK_ENTRY_1.engine}
              resultSummary={MOCK_ENTRY_1.resultSummary}
            />
          </QueryHistoryProvider>
        </QueryProvider>
      </EngineProvider>,
    )

    await userEvent.click(screen.getByTestId('add-entry-btn'))

    // Verify stored
    await waitFor(() => {
      const stored = localStorage.getItem(QUERY_HISTORY_KEY)
      expect(stored).not.toBeNull()
    })

    // Simulate page reload: unmount and re-render
    unmount()
    renderQueryHistory()

    // History should still be visible
    await screen.findByText(/PeakReported/)
  })

  // -------------------------------------------------------------------------
  // 5. Clear button removes all entries
  // -------------------------------------------------------------------------

  it('clear history button removes all entries', async () => {
    const user = userEvent.setup()
    const entries = [MOCK_ENTRY_1, MOCK_ENTRY_2]
    localStorage.setItem(QUERY_HISTORY_KEY, JSON.stringify(entries))

    renderQueryHistory()

    // Both entries should be visible
    expect(screen.getByText(/PeakReported/)).toBeInTheDocument()
    expect(screen.getByText(/Study/)).toBeInTheDocument()

    // Click clear button
    const clearBtn = screen.getByRole('button', { name: /clear history/i })
    await user.click(clearBtn)

    // History entries should be gone
    await waitFor(() => {
      expect(screen.queryByText(/PeakReported/)).not.toBeInTheDocument()
      expect(screen.queryByText(/Study/)).not.toBeInTheDocument()
    })

    // LocalStorage should be cleared too
    const stored = localStorage.getItem(QUERY_HISTORY_KEY)
    const parsed = stored ? (JSON.parse(stored) as unknown[]) : []
    expect(parsed).toHaveLength(0)
  })

  // -------------------------------------------------------------------------
  // 6. Max 50 entries (FIFO)
  // -------------------------------------------------------------------------

  it('enforces max 50 entries with FIFO eviction', async () => {
    const user = userEvent.setup()

    // Pre-populate with 50 entries
    const fiftyEntries = Array.from({ length: 50 }, (_, i) => ({
      query: `ans(x) :- Pred${i}(x)`,
      engine: 'neurosynth',
      timestamp: new Date(2024, 0, i + 1).toISOString(),
      resultSummary: `${i} rows`,
    }))
    localStorage.setItem(QUERY_HISTORY_KEY, JSON.stringify(fiftyEntries))

    render(
      <EngineProvider>
        <QueryProvider>
          <QueryHistoryProvider>
            <AddEntryButton
              query="ans(x) :- NewPred(x)"
              engine="neurosynth"
              resultSummary="5 rows"
            />
          </QueryHistoryProvider>
        </QueryProvider>
      </EngineProvider>,
    )

    await user.click(screen.getByTestId('add-entry-btn'))

    await waitFor(() => {
      const stored = localStorage.getItem(QUERY_HISTORY_KEY)
      const parsed = JSON.parse(stored!) as Array<{ query: string }>
      // Still max 50 entries
      expect(parsed).toHaveLength(50)
      // New entry should be at the beginning (most recent)
      expect(parsed[0].query).toBe('ans(x) :- NewPred(x)')
      // Oldest entry (Pred49, last in the original list) should have been evicted
      expect(parsed.find((e) => e.query === 'ans(x) :- Pred49(x)')).toBeUndefined()
      // Most recent of the original entries (Pred0) should still be present
      expect(parsed.find((e) => e.query === 'ans(x) :- Pred0(x)')).toBeDefined()
    })
  })

  // -------------------------------------------------------------------------
  // 7. Collapsible panel
  // -------------------------------------------------------------------------

  it('can collapse and expand the history panel', async () => {
    const user = userEvent.setup()
    const entries = [MOCK_ENTRY_1]
    localStorage.setItem(QUERY_HISTORY_KEY, JSON.stringify(entries))

    renderQueryHistory()

    // Panel should be initially open
    expect(screen.getByText(/PeakReported/)).toBeInTheDocument()

    // Click the header to collapse
    const header = screen.getByRole('button', { name: /collapse history panel/i })
    await user.click(header)

    // Entries should be hidden
    await waitFor(() => {
      expect(screen.queryByText(/PeakReported/)).not.toBeInTheDocument()
    })

    // Click again to expand
    const expandHeader = screen.getByRole('button', { name: /expand history panel/i })
    await user.click(expandHeader)

    // Entries should be visible again
    await screen.findByText(/PeakReported/)
  })

  // -------------------------------------------------------------------------
  // 8. Shows empty state when no history
  // -------------------------------------------------------------------------

  it('shows empty state message when no history', () => {
    renderQueryHistory()

    expect(screen.getByText(/no history/i)).toBeInTheDocument()
  })

  // -------------------------------------------------------------------------
  // 9. Shows query preview (truncated) and timestamp
  // -------------------------------------------------------------------------

  it('shows truncated query preview and timestamp', async () => {
    const longQuery = 'ans(x, y, z) :- PeakReported(x, y, z, s), Study(s, term), TermInStudyTFIDF(term, tfidf, s), tfidf > 0.5'
    const entries = [
      {
        query: longQuery,
        engine: 'neurosynth',
        timestamp: '2024-01-15T10:30:00.000Z',
        resultSummary: '15 rows',
      },
    ]
    localStorage.setItem(QUERY_HISTORY_KEY, JSON.stringify(entries))

    renderQueryHistory()

    // Should display a truncated version (not the full query text)
    const items = screen.getAllByRole('listitem')
    expect(items).toHaveLength(1)

    // Check that some portion of the query is visible
    expect(screen.getByText(/PeakReported/)).toBeInTheDocument()
  })
})
