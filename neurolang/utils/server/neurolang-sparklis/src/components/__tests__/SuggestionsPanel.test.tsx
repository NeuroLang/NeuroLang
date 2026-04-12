/**
 * SuggestionsPanel.test.tsx
 *
 * Tests for the SuggestionsPanel component.
 * Covers: renders suggestions, click fires callback, loading state, empty state.
 *
 * NOTE: We pass `debounceMs={0}` to SuggestionsPanel in most tests so that
 * fetch is triggered immediately after state changes, avoiding fake timer
 * complexity. Debounce behaviour is tested separately using fake timers.
 */
import React from 'react'
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react'
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import SuggestionsPanel from '../SuggestionsPanel'
import { EngineProvider } from '../../context/EngineContext'
import { QueryProvider } from '../../context/QueryContext'
import { useEngine } from '../../context/useEngine'
import { useQuery } from '../../context/useQuery'

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Helper component that selects an engine via context. */
function EngineSelector({ engine }: { engine: string }): React.ReactElement {
  const { setSelectedEngine } = useEngine()
  return (
    <button data-testid="select-engine" onClick={() => setSelectedEngine(engine)}>
      Select {engine}
    </button>
  )
}

/** Helper component that adds a predicate via context. */
function PredicateAdder({
  name,
  params,
}: {
  name: string
  params: string[]
}): React.ReactElement {
  const { model, refresh } = useQuery()
  return (
    <button
      data-testid="add-predicate"
      onClick={() => {
        model.addPredicate(name, params)
        refresh()
      }}
    >
      Add {name}
    </button>
  )
}

type PanelProps = React.ComponentProps<typeof SuggestionsPanel>

/**
 * Render SuggestionsPanel with providers, engine selector, and predicate adder.
 * Pass debounceMs=0 by default to make tests synchronous (no fake timers needed).
 */
function renderWithControls(
  props: PanelProps = {},
): ReturnType<typeof render> {
  const { debounceMs = 0, ...rest } = props
  return render(
    <EngineProvider>
      <QueryProvider>
        <EngineSelector engine="neurosynth" />
        <PredicateAdder name="Study" params={['study']} />
        <SuggestionsPanel debounceMs={debounceMs} {...rest} />
      </QueryProvider>
    </EngineProvider>,
  )
}

/** Build a resolved fetch Response with the given suggestion data. */
function makeSuggestionsResponse(data: Record<string, string[]>): Response {
  return {
    ok: true,
    json: async () => ({ status: 'ok', data }),
  } as Response
}

/** Build a resolved fetch Response for an error result. */
function makeErrorResponse(message: string): Response {
  return {
    ok: true,
    json: async () => ({ status: 'error', message }),
  } as Response
}

/** Build a resolved fetch Response for an HTTP error. */
function makeHttpErrorResponse(status: number): Response {
  return {
    ok: false,
    status,
    json: async () => ({ status: 'error' }),
  } as Response
}

// ---------------------------------------------------------------------------
// Setup / teardown
// ---------------------------------------------------------------------------

beforeEach(() => {
  vi.stubGlobal('fetch', vi.fn())
})

afterEach(() => {
  vi.restoreAllMocks()
})

// ---------------------------------------------------------------------------
// No-engine state
// ---------------------------------------------------------------------------

describe('SuggestionsPanel – no engine selected', () => {
  it('renders the panel title', () => {
    render(
      <EngineProvider>
        <QueryProvider>
          <SuggestionsPanel debounceMs={0} />
        </QueryProvider>
      </EngineProvider>,
    )
    expect(screen.getByText('Suggestions')).toBeInTheDocument()
  })

  it('shows "select an engine" placeholder when no engine is chosen', () => {
    render(
      <EngineProvider>
        <QueryProvider>
          <SuggestionsPanel debounceMs={0} />
        </QueryProvider>
      </EngineProvider>,
    )
    expect(
      screen.getByText(/Select an engine to see suggestions/i),
    ).toBeInTheDocument()
  })

  it('does not call fetch when no engine is selected', async () => {
    render(
      <EngineProvider>
        <QueryProvider>
          <SuggestionsPanel debounceMs={0} />
        </QueryProvider>
      </EngineProvider>,
    )
    // Small wait to confirm no fetch happened
    await new Promise((r) => setTimeout(r, 20))
    expect(fetch).not.toHaveBeenCalled()
  })
})

// ---------------------------------------------------------------------------
// Loading state
// ---------------------------------------------------------------------------

describe('SuggestionsPanel – loading state', () => {
  it('shows loading spinner while the first fetch is in progress', async () => {
    // Controlled deferred promise
    let resolveFetch!: (v: Response) => void
    const deferredFetch = new Promise<Response>((resolve) => {
      resolveFetch = resolve
    })
    vi.mocked(fetch).mockReturnValue(deferredFetch)

    renderWithControls()

    // Click engine selector – with debounceMs=0, fetch starts on next tick
    fireEvent.click(screen.getByTestId('select-engine'))

    // Spinner should appear while fetch is pending
    await waitFor(() => {
      expect(screen.getByLabelText('Loading suggestions')).toBeInTheDocument()
    })

    // Resolve fetch and verify spinner disappears
    await act(async () => {
      resolveFetch(makeSuggestionsResponse({ Identifiers: ['Study'] }))
    })

    expect(screen.queryByLabelText('Loading suggestions')).not.toBeInTheDocument()
    expect(screen.getByText('Study')).toBeInTheDocument()
  })

  it('shows "updating…" status text while re-fetching with existing suggestions', async () => {
    let resolveSecond!: (v: Response) => void
    const secondFetch = new Promise<Response>((resolve) => {
      resolveSecond = resolve
    })

    vi.mocked(fetch)
      .mockResolvedValueOnce(
        makeSuggestionsResponse({ Identifiers: ['Study', 'PeakReported'] }),
      )
      .mockReturnValueOnce(secondFetch)

    renderWithControls()

    // First fetch: select engine → wait for suggestions
    fireEvent.click(screen.getByTestId('select-engine'))
    await waitFor(() => {
      expect(screen.getByText('Study')).toBeInTheDocument()
    })

    // Second fetch: add predicate → loading while deferred
    fireEvent.click(screen.getByTestId('add-predicate'))

    await waitFor(() => {
      expect(screen.getByText('updating…')).toBeInTheDocument()
    })
    // Previous suggestions remain visible
    expect(screen.getByText('Study')).toBeInTheDocument()

    // Cleanup
    await act(async () => {
      resolveSecond(makeSuggestionsResponse({ Identifiers: ['Study'] }))
    })
  })
})

// ---------------------------------------------------------------------------
// Rendering suggestions
// ---------------------------------------------------------------------------

describe('SuggestionsPanel – renders suggestions', () => {
  it('renders category labels and chips after fetch resolves', async () => {
    vi.mocked(fetch).mockResolvedValue(
      makeSuggestionsResponse({
        Identifiers: ['Study', 'PeakReported'],
        Operators: [':-', ','],
      }),
    )

    renderWithControls()
    fireEvent.click(screen.getByTestId('select-engine'))

    await waitFor(() => {
      expect(screen.getByText('Study')).toBeInTheDocument()
    })

    expect(screen.getByText('PeakReported')).toBeInTheDocument()
    // Identifiers → mapped to "Predicates" group
    expect(screen.getByText('Predicates')).toBeInTheDocument()
    // Operators → mapped to "Operators" group
    expect(screen.getByText('Operators')).toBeInTheDocument()
    expect(screen.getByText(':-')).toBeInTheDocument()
    expect(screen.getByText(',')).toBeInTheDocument()
  })

  it('renders chips as buttons', async () => {
    vi.mocked(fetch).mockResolvedValue(
      makeSuggestionsResponse({ Identifiers: ['Study'] }),
    )

    renderWithControls()
    fireEvent.click(screen.getByTestId('select-engine'))

    await waitFor(() => {
      expect(
        screen.getByRole('button', { name: 'Suggestion: Study' }),
      ).toBeInTheDocument()
    })
  })

  it('shows empty state when suggestions data is empty', async () => {
    vi.mocked(fetch).mockResolvedValue(makeSuggestionsResponse({}))

    renderWithControls()
    fireEvent.click(screen.getByTestId('select-engine'))

    await waitFor(() => {
      expect(screen.getByText(/No suggestions available/i)).toBeInTheDocument()
    })
  })

  it('hides categories whose suggestion list is empty', async () => {
    vi.mocked(fetch).mockResolvedValue(
      makeSuggestionsResponse({
        Identifiers: ['Study'],
        Operators: [],
      }),
    )

    renderWithControls()
    fireEvent.click(screen.getByTestId('select-engine'))

    await waitFor(() => {
      expect(screen.getByText('Study')).toBeInTheDocument()
    })

    expect(screen.queryByText('Operators')).not.toBeInTheDocument()
  })
})

// ---------------------------------------------------------------------------
// Click fires callback
// ---------------------------------------------------------------------------

describe('SuggestionsPanel – click fires callback', () => {
  it('calls onSuggestionSelect with the chip text and category', async () => {
    const handleSelect = vi.fn()

    vi.mocked(fetch).mockResolvedValue(
      makeSuggestionsResponse({ Identifiers: ['Study', 'PeakReported'] }),
    )

    renderWithControls({ onSuggestionSelect: handleSelect })
    fireEvent.click(screen.getByTestId('select-engine'))

    await waitFor(() => {
      expect(screen.getByText('Study')).toBeInTheDocument()
    })

    fireEvent.click(screen.getByRole('button', { name: 'Suggestion: Study' }))

    expect(handleSelect).toHaveBeenCalledTimes(1)
    // Identifiers category is now mapped to "Predicates" group
    expect(handleSelect).toHaveBeenCalledWith('Study', 'Predicates')
  })

  it('calls onSuggestionSelect with correct category for operators', async () => {
    const handleSelect = vi.fn()

    vi.mocked(fetch).mockResolvedValue(
      makeSuggestionsResponse({ Operators: [':-', ','] }),
    )

    renderWithControls({ onSuggestionSelect: handleSelect })
    fireEvent.click(screen.getByTestId('select-engine'))

    await waitFor(() => {
      expect(screen.getByText(':-')).toBeInTheDocument()
    })

    fireEvent.click(screen.getByRole('button', { name: 'Suggestion: :-' }))

    expect(handleSelect).toHaveBeenCalledWith(':-', 'Operators')
  })

  it('does not throw when no onSuggestionSelect callback is provided', async () => {
    vi.mocked(fetch).mockResolvedValue(
      makeSuggestionsResponse({ Identifiers: ['Study'] }),
    )

    renderWithControls() // no callback
    fireEvent.click(screen.getByTestId('select-engine'))

    await waitFor(() => {
      expect(screen.getByText('Study')).toBeInTheDocument()
    })

    expect(() => {
      fireEvent.click(screen.getByRole('button', { name: 'Suggestion: Study' }))
    }).not.toThrow()
  })
})

// ---------------------------------------------------------------------------
// Error handling
// ---------------------------------------------------------------------------

describe('SuggestionsPanel – error handling', () => {
  it('shows error message when fetch throws', async () => {
    vi.mocked(fetch).mockRejectedValue(new Error('Network failure'))

    renderWithControls()
    fireEvent.click(screen.getByTestId('select-engine'))

    await waitFor(() => {
      expect(screen.getByText('Network failure')).toBeInTheDocument()
    })
  })

  it('shows error message when server returns non-ok status', async () => {
    vi.mocked(fetch).mockResolvedValue(makeHttpErrorResponse(503))

    renderWithControls()
    fireEvent.click(screen.getByTestId('select-engine'))

    await waitFor(() => {
      expect(
        screen.getByText(/Suggestions request failed: 503/i),
      ).toBeInTheDocument()
    })
  })

  it('shows error message from error-status response body', async () => {
    vi.mocked(fetch).mockResolvedValue(
      makeErrorResponse('Parse error at line 1'),
    )

    renderWithControls()
    fireEvent.click(screen.getByTestId('select-engine'))

    await waitFor(() => {
      expect(screen.getByText('Parse error at line 1')).toBeInTheDocument()
    })
  })
})

// ---------------------------------------------------------------------------
// Debouncing (uses fake timers)
// ---------------------------------------------------------------------------

describe('SuggestionsPanel – debounce', () => {
  afterEach(() => {
    vi.useRealTimers()
  })

  it('does not call fetch before the debounce delay elapses', () => {
    vi.useFakeTimers()
    vi.mocked(fetch).mockResolvedValue(
      makeSuggestionsResponse({ Identifiers: ['Study'] }),
    )

    // Use explicit 300ms debounce (no override)
    renderWithControls({ debounceMs: 300 })

    act(() => {
      fireEvent.click(screen.getByTestId('select-engine'))
      // Advance 299ms – timer should NOT have fired yet
      vi.advanceTimersByTime(299)
    })

    expect(fetch).not.toHaveBeenCalled()
    vi.useRealTimers()
  })

  it('sends the correct request after the debounce delay', async () => {
    vi.useFakeTimers()
    vi.mocked(fetch).mockResolvedValue(
      makeSuggestionsResponse({ Identifiers: ['Study'] }),
    )

    renderWithControls({ debounceMs: 300 })

    // First flush React scheduler so the effect runs and registers the timer
    act(() => {
      fireEvent.click(screen.getByTestId('select-engine'))
    })
    // Now advance 300ms so the debounce timer fires
    act(() => {
      vi.advanceTimersByTime(300)
    })

    expect(fetch).toHaveBeenCalledTimes(1)
    expect(fetch).toHaveBeenCalledWith(
      '/v2/suggest/neurosynth',
      expect.objectContaining({ method: 'POST' }),
    )

    const callArgs = vi.mocked(fetch).mock.calls[0]
    const body = JSON.parse(callArgs[1]?.body as string)
    expect(typeof body.program).toBe('string')

    // Restore real timers before awaiting async cleanup
    vi.useRealTimers()
    // Allow async state updates from fetchSuggestions to settle
    await act(async () => {})
  })
})

// ---------------------------------------------------------------------------
// Serialization fix: non-empty queries should not send trailing period
// ---------------------------------------------------------------------------

describe('SuggestionsPanel – program serialization for non-empty queries', () => {
  it('sends empty string program when query is empty', async () => {
    vi.mocked(fetch).mockResolvedValue(
      makeSuggestionsResponse({ Identifiers: ['Study'] }),
    )

    renderWithControls()
    fireEvent.click(screen.getByTestId('select-engine'))

    await waitFor(() => {
      expect(fetch).toHaveBeenCalled()
    })

    const callArgs = vi.mocked(fetch).mock.calls[0]
    const body = JSON.parse(callArgs[1]?.body as string)
    expect(body.program).toBe('')
  })

  it('strips trailing period from serialized program before sending', async () => {
    vi.mocked(fetch).mockResolvedValue(
      makeSuggestionsResponse({ Identifiers: ['PeakReported', 'Study'] }),
    )

    renderWithControls()
    // Select engine first
    fireEvent.click(screen.getByTestId('select-engine'))

    await waitFor(() => expect(fetch).toHaveBeenCalledTimes(1))

    vi.mocked(fetch).mockClear()
    vi.mocked(fetch).mockResolvedValue(
      makeSuggestionsResponse({ Identifiers: ['PeakReported'] }),
    )

    // Add a predicate so the query is non-empty
    fireEvent.click(screen.getByTestId('add-predicate'))

    await waitFor(() => {
      expect(fetch).toHaveBeenCalled()
    })

    const callArgs = vi.mocked(fetch).mock.calls[0]
    const body = JSON.parse(callArgs[1]?.body as string)
    // Non-empty query program must NOT end with "."; should end with ","
    expect(body.program).not.toMatch(/\.$/)
    // Should end with "," for body-extensible context
    expect(body.program).toMatch(/,$/)
    expect(body.program.length).toBeGreaterThan(0)
  })
})

// ---------------------------------------------------------------------------
// Search/filter
// ---------------------------------------------------------------------------

describe('SuggestionsPanel – search/filter input', () => {
  it('renders a search input above the suggestion chips', async () => {
    vi.mocked(fetch).mockResolvedValue(
      makeSuggestionsResponse({ Identifiers: ['Study', 'PeakReported'] }),
    )

    renderWithControls()
    fireEvent.click(screen.getByTestId('select-engine'))

    await waitFor(() => {
      expect(screen.getByText('Study')).toBeInTheDocument()
    })

    expect(screen.getByPlaceholderText(/filter suggestions/i)).toBeInTheDocument()
  })

  it('filters suggestions by substring when user types in search input', async () => {
    vi.mocked(fetch).mockResolvedValue(
      makeSuggestionsResponse({
        Identifiers: ['Study', 'PeakReported', 'TermInStudyTFIDF'],
        Operators: [':-', ','],
      }),
    )

    renderWithControls()
    fireEvent.click(screen.getByTestId('select-engine'))

    await waitFor(() => {
      expect(screen.getByText('Study')).toBeInTheDocument()
    })

    const searchInput = screen.getByPlaceholderText(/filter suggestions/i)
    fireEvent.change(searchInput, { target: { value: 'Peak' } })

    // PeakReported should be visible
    expect(screen.getByText('PeakReported')).toBeInTheDocument()
    // Study and TermInStudyTFIDF should be hidden (Study doesn't match 'Peak')
    expect(screen.queryByText('Study')).not.toBeInTheDocument()
    expect(screen.queryByText('TermInStudyTFIDF')).not.toBeInTheDocument()
    // Operators don't match 'Peak' either
    expect(screen.queryByText(':-')).not.toBeInTheDocument()
  })

  it('shows all suggestions when search input is cleared', async () => {
    vi.mocked(fetch).mockResolvedValue(
      makeSuggestionsResponse({
        Identifiers: ['Study', 'PeakReported'],
        Operators: [':-'],
      }),
    )

    renderWithControls()
    fireEvent.click(screen.getByTestId('select-engine'))

    await waitFor(() => {
      expect(screen.getByText('Study')).toBeInTheDocument()
    })

    const searchInput = screen.getByPlaceholderText(/filter suggestions/i)
    fireEvent.change(searchInput, { target: { value: 'Study' } })
    expect(screen.queryByText('PeakReported')).not.toBeInTheDocument()

    fireEvent.change(searchInput, { target: { value: '' } })
    expect(screen.getByText('PeakReported')).toBeInTheDocument()
    expect(screen.getByText('Study')).toBeInTheDocument()
  })

  it('filter is case-insensitive', async () => {
    vi.mocked(fetch).mockResolvedValue(
      makeSuggestionsResponse({ Identifiers: ['PeakReported'] }),
    )

    renderWithControls()
    fireEvent.click(screen.getByTestId('select-engine'))

    await waitFor(() => {
      expect(screen.getByText('PeakReported')).toBeInTheDocument()
    })

    const searchInput = screen.getByPlaceholderText(/filter suggestions/i)
    fireEvent.change(searchInput, { target: { value: 'peak' } })

    expect(screen.getByText('PeakReported')).toBeInTheDocument()
  })

  it('shows empty state message when filter matches nothing', async () => {
    vi.mocked(fetch).mockResolvedValue(
      makeSuggestionsResponse({ Identifiers: ['Study', 'PeakReported'] }),
    )

    renderWithControls()
    fireEvent.click(screen.getByTestId('select-engine'))

    await waitFor(() => {
      expect(screen.getByText('Study')).toBeInTheDocument()
    })

    const searchInput = screen.getByPlaceholderText(/filter suggestions/i)
    fireEvent.change(searchInput, { target: { value: 'xyz_no_match' } })

    expect(screen.getByText(/no suggestions match/i)).toBeInTheDocument()
  })

  it('resets search filter when new suggestions arrive', async () => {
    vi.mocked(fetch)
      .mockResolvedValueOnce(
        makeSuggestionsResponse({ Identifiers: ['Study', 'PeakReported'] }),
      )
      .mockResolvedValueOnce(
        makeSuggestionsResponse({ Identifiers: ['TermInStudyTFIDF'] }),
      )

    renderWithControls()
    fireEvent.click(screen.getByTestId('select-engine'))

    await waitFor(() => {
      expect(screen.getByText('Study')).toBeInTheDocument()
    })

    const searchInput = screen.getByPlaceholderText(/filter suggestions/i)
    fireEvent.change(searchInput, { target: { value: 'Peak' } })
    expect(screen.queryByText('Study')).not.toBeInTheDocument()

    // When new suggestions arrive (predicate added), filter should be reset
    fireEvent.click(screen.getByTestId('add-predicate'))

    await waitFor(() => {
      expect(screen.getByText('TermInStudyTFIDF')).toBeInTheDocument()
    })
    // Filter input should be cleared
    expect(searchInput).toHaveValue('')
  })
})

// ---------------------------------------------------------------------------
// Grouping (Predicates / Operators / Other)
// ---------------------------------------------------------------------------

describe('SuggestionsPanel – intelligent grouping', () => {
  it('shows Predicates group for Identifiers category', async () => {
    vi.mocked(fetch).mockResolvedValue(
      makeSuggestionsResponse({ Identifiers: ['Study'] }),
    )

    renderWithControls()
    fireEvent.click(screen.getByTestId('select-engine'))

    await waitFor(() => {
      expect(screen.getByText('Study')).toBeInTheDocument()
    })

    expect(screen.getByText('Predicates')).toBeInTheDocument()
  })

  it('shows Operators group for Operators category', async () => {
    vi.mocked(fetch).mockResolvedValue(
      makeSuggestionsResponse({ Operators: [':-', ','] }),
    )

    renderWithControls()
    fireEvent.click(screen.getByTestId('select-engine'))

    await waitFor(() => {
      expect(screen.getByText(':-')).toBeInTheDocument()
    })

    expect(screen.getByText('Operators')).toBeInTheDocument()
  })

  it('shows Other group for unknown categories', async () => {
    vi.mocked(fetch).mockResolvedValue(
      makeSuggestionsResponse({ Numbers: ['42', '0'] }),
    )

    renderWithControls()
    fireEvent.click(screen.getByTestId('select-engine'))

    await waitFor(() => {
      expect(screen.getByText('42')).toBeInTheDocument()
    })

    expect(screen.getByText('Other')).toBeInTheDocument()
  })

  it('renders Predicates group before Operators group before Other', async () => {
    vi.mocked(fetch).mockResolvedValue(
      makeSuggestionsResponse({
        Numbers: ['42'],
        Operators: [':-'],
        Identifiers: ['Study'],
      }),
    )

    renderWithControls()
    fireEvent.click(screen.getByTestId('select-engine'))

    await waitFor(() => {
      expect(screen.getByText('Study')).toBeInTheDocument()
    })

    const labels = screen
      .getAllByText(/Predicates|Operators|Other/)
      .map((el) => el.textContent)

    expect(labels).toEqual(['Predicates', 'Operators', 'Other'])
  })
})

// ---------------------------------------------------------------------------
// Show more
// ---------------------------------------------------------------------------

describe('SuggestionsPanel – show more', () => {
  /** Build an array of N unique suggestion strings */
  function makeManyItems(n: number): string[] {
    return Array.from({ length: n }, (_, i) => `Item${i + 1}`)
  }

  it('limits displayed suggestions to 20 per category when more than 20 exist', async () => {
    vi.mocked(fetch).mockResolvedValue(
      makeSuggestionsResponse({ Identifiers: makeManyItems(25) }),
    )

    renderWithControls()
    fireEvent.click(screen.getByTestId('select-engine'))

    await waitFor(() => {
      expect(screen.getByText('Item1')).toBeInTheDocument()
    })

    // Only first 20 items should be visible
    expect(screen.getByText('Item20')).toBeInTheDocument()
    expect(screen.queryByText('Item21')).not.toBeInTheDocument()
  })

  it('shows all suggestions when count is 20 or fewer', async () => {
    vi.mocked(fetch).mockResolvedValue(
      makeSuggestionsResponse({ Identifiers: makeManyItems(20) }),
    )

    renderWithControls()
    fireEvent.click(screen.getByTestId('select-engine'))

    await waitFor(() => {
      expect(screen.getByText('Item1')).toBeInTheDocument()
    })

    expect(screen.getByText('Item20')).toBeInTheDocument()
    expect(screen.queryByText(/Show more/i)).not.toBeInTheDocument()
  })

  it('shows "Show more" button when more than 20 suggestions exist in a category', async () => {
    vi.mocked(fetch).mockResolvedValue(
      makeSuggestionsResponse({ Identifiers: makeManyItems(25) }),
    )

    renderWithControls()
    fireEvent.click(screen.getByTestId('select-engine'))

    await waitFor(() => {
      expect(screen.getByText('Item1')).toBeInTheDocument()
    })

    expect(screen.getByRole('button', { name: /Show more/i })).toBeInTheDocument()
  })

  it('reveals additional items after clicking "Show more"', async () => {
    vi.mocked(fetch).mockResolvedValue(
      makeSuggestionsResponse({ Identifiers: makeManyItems(25) }),
    )

    renderWithControls()
    fireEvent.click(screen.getByTestId('select-engine'))

    await waitFor(() => {
      expect(screen.getByText('Item1')).toBeInTheDocument()
    })

    expect(screen.queryByText('Item21')).not.toBeInTheDocument()

    fireEvent.click(screen.getByRole('button', { name: /Show more/i }))

    expect(screen.getByText('Item21')).toBeInTheDocument()
    expect(screen.getByText('Item25')).toBeInTheDocument()
  })

  it('hides "Show more" button after all items are revealed', async () => {
    vi.mocked(fetch).mockResolvedValue(
      makeSuggestionsResponse({ Identifiers: makeManyItems(25) }),
    )

    renderWithControls()
    fireEvent.click(screen.getByTestId('select-engine'))

    await waitFor(() => {
      expect(screen.getByText('Item1')).toBeInTheDocument()
    })

    fireEvent.click(screen.getByRole('button', { name: /Show more/i }))

    expect(screen.queryByRole('button', { name: /Show more/i })).not.toBeInTheDocument()
  })
})

// ---------------------------------------------------------------------------
// Keyboard navigation
// ---------------------------------------------------------------------------

describe('SuggestionsPanel – keyboard navigation', () => {
  it('focuses first chip when ArrowDown is pressed from search input', async () => {
    vi.mocked(fetch).mockResolvedValue(
      makeSuggestionsResponse({ Identifiers: ['Study', 'PeakReported'] }),
    )

    renderWithControls()
    fireEvent.click(screen.getByTestId('select-engine'))

    await waitFor(() => {
      expect(screen.getByText('Study')).toBeInTheDocument()
    })

    const searchInput = screen.getByPlaceholderText(/filter suggestions/i)
    fireEvent.keyDown(searchInput, { key: 'ArrowDown' })

    const firstChip = screen.getByRole('button', { name: 'Suggestion: Study' })
    expect(firstChip).toHaveFocus()
  })

  it('moves focus to next chip when ArrowDown is pressed on a chip', async () => {
    vi.mocked(fetch).mockResolvedValue(
      makeSuggestionsResponse({ Identifiers: ['Study', 'PeakReported'] }),
    )

    renderWithControls()
    fireEvent.click(screen.getByTestId('select-engine'))

    await waitFor(() => {
      expect(screen.getByText('Study')).toBeInTheDocument()
    })

    const searchInput = screen.getByPlaceholderText(/filter suggestions/i)
    fireEvent.keyDown(searchInput, { key: 'ArrowDown' })

    const firstChip = screen.getByRole('button', { name: 'Suggestion: Study' })
    expect(firstChip).toHaveFocus()

    fireEvent.keyDown(firstChip, { key: 'ArrowDown' })
    const secondChip = screen.getByRole('button', { name: 'Suggestion: PeakReported' })
    expect(secondChip).toHaveFocus()
  })

  it('moves focus to previous chip when ArrowUp is pressed on a chip', async () => {
    vi.mocked(fetch).mockResolvedValue(
      makeSuggestionsResponse({ Identifiers: ['Study', 'PeakReported'] }),
    )

    renderWithControls()
    fireEvent.click(screen.getByTestId('select-engine'))

    await waitFor(() => {
      expect(screen.getByText('Study')).toBeInTheDocument()
    })

    const searchInput = screen.getByPlaceholderText(/filter suggestions/i)
    // Navigate down to second chip
    fireEvent.keyDown(searchInput, { key: 'ArrowDown' })
    const firstChip = screen.getByRole('button', { name: 'Suggestion: Study' })
    fireEvent.keyDown(firstChip, { key: 'ArrowDown' })

    const secondChip = screen.getByRole('button', { name: 'Suggestion: PeakReported' })
    expect(secondChip).toHaveFocus()

    // Navigate back up
    fireEvent.keyDown(secondChip, { key: 'ArrowUp' })
    expect(firstChip).toHaveFocus()
  })

  it('pressing ArrowUp on first chip moves focus to search input', async () => {
    vi.mocked(fetch).mockResolvedValue(
      makeSuggestionsResponse({ Identifiers: ['Study', 'PeakReported'] }),
    )

    renderWithControls()
    fireEvent.click(screen.getByTestId('select-engine'))

    await waitFor(() => {
      expect(screen.getByText('Study')).toBeInTheDocument()
    })

    const searchInput = screen.getByPlaceholderText(/filter suggestions/i)
    fireEvent.keyDown(searchInput, { key: 'ArrowDown' })

    const firstChip = screen.getByRole('button', { name: 'Suggestion: Study' })
    expect(firstChip).toHaveFocus()

    fireEvent.keyDown(firstChip, { key: 'ArrowUp' })
    expect(searchInput).toHaveFocus()
  })

  it('pressing Enter on a focused chip selects it', async () => {
    const handleSelect = vi.fn()

    vi.mocked(fetch).mockResolvedValue(
      makeSuggestionsResponse({ Identifiers: ['Study'] }),
    )

    renderWithControls({ onSuggestionSelect: handleSelect })
    fireEvent.click(screen.getByTestId('select-engine'))

    await waitFor(() => {
      expect(screen.getByText('Study')).toBeInTheDocument()
    })

    const searchInput = screen.getByPlaceholderText(/filter suggestions/i)
    fireEvent.keyDown(searchInput, { key: 'ArrowDown' })

    const chip = screen.getByRole('button', { name: 'Suggestion: Study' })
    expect(chip).toHaveFocus()

    fireEvent.click(chip)
    expect(handleSelect).toHaveBeenCalledWith('Study', 'Predicates')
  })

  it('ArrowDown on last chip wraps focus to search input', async () => {
    vi.mocked(fetch).mockResolvedValue(
      makeSuggestionsResponse({ Identifiers: ['Study'] }),
    )

    renderWithControls()
    fireEvent.click(screen.getByTestId('select-engine'))

    await waitFor(() => {
      expect(screen.getByText('Study')).toBeInTheDocument()
    })

    const searchInput = screen.getByPlaceholderText(/filter suggestions/i)
    fireEvent.keyDown(searchInput, { key: 'ArrowDown' })

    const chip = screen.getByRole('button', { name: 'Suggestion: Study' })
    expect(chip).toHaveFocus()

    // Only one chip, so ArrowDown should wrap back to search
    fireEvent.keyDown(chip, { key: 'ArrowDown' })
    expect(searchInput).toHaveFocus()
  })
})
