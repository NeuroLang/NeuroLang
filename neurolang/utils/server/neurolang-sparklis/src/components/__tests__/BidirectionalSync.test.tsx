/**
 * BidirectionalSync.test.tsx
 *
 * Tests for bidirectional synchronization between the VisualQueryBuilder
 * and the CodeEditor via the shared QueryContext.
 *
 * Covers:
 *   1. Builder → Editor sync: adding a predicate visually updates datalogText
 *   2. Editor → Builder sync: typing valid Datalog updates the visual builder
 *   3. Desynchronized state: invalid/incomplete Datalog shows desync indicator
 *   4. No infinite loops: model changes from parsing don't re-trigger parsing
 *   5. parseDatalog: unit tests for the parse function
 */
import React from 'react'
import {
  render,
  screen,
  fireEvent,
  act,
} from '@testing-library/react'
import { describe, it, expect, beforeEach, vi } from 'vitest'
import { QueryProvider } from '../../context/QueryContext'
import { useQuery } from '../../context/useQuery'
import VisualQueryBuilder from '../VisualQueryBuilder'
import CodeEditor from '../CodeEditor'
import { resetIdCounter, parseDatalog } from '../../models/QueryModel'

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

/**
 * Component that renders both VisualQueryBuilder and CodeEditor with full
 * bidirectional sync wired up, plus an "Add predicate" button for tests.
 */
function SyncTestHarness({
  predicateName = 'Study',
  params = ['study'],
}: {
  predicateName?: string
  params?: string[]
}): React.ReactElement {
  const { model, refresh, datalogText, setDatalogText } = useQuery()

  return (
    <div>
      <button
        data-testid="add-predicate"
        onClick={() => {
          model.addPredicate(predicateName, params)
          refresh()
        }}
      >
        Add {predicateName}
      </button>
      <VisualQueryBuilder />
      <CodeEditor
        value={datalogText}
        onChange={setDatalogText}
        data-testid="code-editor"
      />
    </div>
  )
}

function renderSync(props?: {
  predicateName?: string
  params?: string[]
}): ReturnType<typeof render> {
  return render(
    <QueryProvider>
      <SyncTestHarness {...props} />
    </QueryProvider>,
  )
}

/**
 * Component to expose isSynced for assertions.
 */
function SyncStateReader({
  onRender,
}: {
  onRender: (isSynced: boolean, datalogText: string) => void
}): React.ReactElement {
  const { isSynced, datalogText } = useQuery()
  onRender(isSynced, datalogText)
  return <></>
}

beforeEach(() => {
  resetIdCounter()
  vi.useFakeTimers()
})

// ---------------------------------------------------------------------------
// parseDatalog unit tests
// ---------------------------------------------------------------------------

describe('parseDatalog', () => {
  it('returns null for empty string', () => {
    expect(parseDatalog('')).toBeNull()
  })

  it('returns null for whitespace-only string', () => {
    expect(parseDatalog('   ')).toBeNull()
  })

  it('returns null for plain text (no rule)', () => {
    expect(parseDatalog('hello world')).toBeNull()
  })

  it('returns null for incomplete rule (no :-)', () => {
    expect(parseDatalog('ans(x) Study(x).')).toBeNull()
  })

  it('returns null for rule without trailing period', () => {
    expect(parseDatalog('ans(x) :- Study(x)')).toBeNull()
  })

  it('parses a single-predicate rule correctly', () => {
    const result = parseDatalog('ans(s) :- Study(s).')
    expect(result).not.toBeNull()
    expect(result!.predicates).toHaveLength(1)
    expect(result!.predicates[0].name).toBe('Study')
    expect(result!.predicates[0].params).toHaveLength(1)
    expect(result!.predicates[0].params[0].varName).toBe('s')
  })

  it('parses a two-predicate rule correctly', () => {
    const result = parseDatalog('ans(x, s) :- PeakReported(x, y, z, s), Study(s).')
    expect(result).not.toBeNull()
    expect(result!.predicates).toHaveLength(2)
    expect(result!.predicates[0].name).toBe('PeakReported')
    expect(result!.predicates[0].params).toHaveLength(4)
    expect(result!.predicates[1].name).toBe('Study')
    expect(result!.predicates[1].params).toHaveLength(1)
    expect(result!.predicates[1].params[0].varName).toBe('s')
  })

  it('parses a rule with multiple shared variables', () => {
    const result = parseDatalog('ans(s) :- PeakReported(x, y, z, s), Study(s).')
    expect(result).not.toBeNull()
    // The shared variable 's' appears in both predicates
    const peak = result!.predicates[0]
    const study = result!.predicates[1]
    const peakStudyVar = peak.params[3].varName
    const studyVar = study.params[0].varName
    expect(peakStudyVar).toBe(studyVar)
  })

  it('parses a zero-arity predicate', () => {
    const result = parseDatalog('ans() :- Empty().')
    expect(result).not.toBeNull()
    expect(result!.predicates[0].name).toBe('Empty')
    expect(result!.predicates[0].params).toHaveLength(0)
  })

  it('returns null for an arg that is not a valid identifier', () => {
    // e.g. a number literal
    expect(parseDatalog('ans(x) :- Study(42).')).toBeNull()
  })

  it('returns null for invalid predicate syntax', () => {
    expect(parseDatalog('ans(x) :- Study x.')).toBeNull()
  })
})

// ---------------------------------------------------------------------------
// Builder → Editor sync tests
// ---------------------------------------------------------------------------

describe('BidirectionalSync – builder to editor', () => {
  it('initially datalogText is empty', () => {
    let syncState = { isSynced: true, text: '' }
    render(
      <QueryProvider>
        <SyncStateReader
          onRender={(isSynced, datalogText) => {
            syncState = { isSynced, text: datalogText }
          }}
        />
      </QueryProvider>,
    )
    expect(syncState.text).toBe('')
    expect(syncState.isSynced).toBe(true)
  })

  it('adding a predicate visually updates datalogText', () => {
    renderSync()

    // Add a predicate by clicking the button
    act(() => {
      fireEvent.click(screen.getByTestId('add-predicate'))
    })

    // The code editor should now show Datalog text
    const codeEditor = screen.getByTestId('code-editor')
    const content = codeEditor.querySelector('.cm-content')
    expect(content?.textContent).toMatch(/Study/)
  })

  it('adding a predicate produces valid Datalog text in the editor', () => {
    renderSync()

    act(() => {
      fireEvent.click(screen.getByTestId('add-predicate'))
    })

    const codeEditor = screen.getByTestId('code-editor')
    const content = codeEditor.querySelector('.cm-content')
    // Should match the serialized format: ans(...) :- Study(...)
    expect(content?.textContent).toMatch(/ans\(.+\) :- Study\(.+\)\./)
  })

  it('isSynced remains true after adding a predicate visually', () => {
    let syncState = { isSynced: true, text: '' }
    render(
      <QueryProvider>
        <SyncStateReader
          onRender={(isSynced, datalogText) => {
            syncState = { isSynced, text: datalogText }
          }}
        />
        <SyncTestHarness />
      </QueryProvider>,
    )

    act(() => {
      fireEvent.click(screen.getByTestId('add-predicate'))
    })

    expect(syncState.isSynced).toBe(true)
  })

  it('removing a predicate updates the code editor to empty string', () => {
    renderSync()

    // Add then remove
    act(() => {
      fireEvent.click(screen.getByTestId('add-predicate'))
    })
    act(() => {
      fireEvent.click(screen.getByRole('button', { name: /Remove Study/i }))
    })

    const codeEditor = screen.getByTestId('code-editor')
    const content = codeEditor.querySelector('.cm-content')
    // After removing the only predicate, text should be empty
    expect(content?.textContent?.trim()).toBe('')
  })
})

// ---------------------------------------------------------------------------
// Editor → Builder sync tests
// ---------------------------------------------------------------------------

describe('BidirectionalSync – editor to builder (via context)', () => {
  it('typing valid Datalog in context updates isSynced after debounce', async () => {
    let syncState = { isSynced: true, text: '' }

    function ContextMonitor(): React.ReactElement {
      const { isSynced, datalogText, setDatalogText } = useQuery()
      syncState = { isSynced, text: datalogText }
      return (
        <button
          data-testid="set-text"
          onClick={() => setDatalogText('ans(s) :- Study(s).')}
        />
      )
    }

    render(
      <QueryProvider>
        <ContextMonitor />
        <VisualQueryBuilder />
      </QueryProvider>,
    )

    // Simulate editor change
    act(() => {
      fireEvent.click(screen.getByTestId('set-text'))
    })

    // Before debounce, isSynced should still be true (no parse yet)
    expect(syncState.isSynced).toBe(true)
    expect(syncState.text).toBe('ans(s) :- Study(s).')

    // Advance timers by debounce delay
    await act(async () => {
      vi.advanceTimersByTime(500)
    })

    // After debounce, parsing should succeed and builder should show 'Study'
    expect(syncState.isSynced).toBe(true)
    expect(screen.getByText('Study')).toBeInTheDocument()
  })

  it('typing valid multi-predicate Datalog updates the visual builder', async () => {
    function ContextSetter(): React.ReactElement {
      const { setDatalogText } = useQuery()
      return (
        <button
          data-testid="set-text"
          onClick={() =>
            setDatalogText('ans(x, s) :- PeakReported(x, y, z, s), Study(s).')
          }
        />
      )
    }

    render(
      <QueryProvider>
        <ContextSetter />
        <VisualQueryBuilder />
      </QueryProvider>,
    )

    act(() => {
      fireEvent.click(screen.getByTestId('set-text'))
    })

    await act(async () => {
      vi.advanceTimersByTime(500)
    })

    // Both predicates should appear in the visual builder
    expect(screen.getByText('PeakReported')).toBeInTheDocument()
    expect(screen.getByText('Study')).toBeInTheDocument()
  })

  it('typing empty string resets the model', async () => {
    function ContextSetter(): React.ReactElement {
      const { setDatalogText } = useQuery()
      return (
        <>
          <button
            data-testid="set-text"
            onClick={() => setDatalogText('ans(s) :- Study(s).')}
          />
          <button
            data-testid="clear-text"
            onClick={() => setDatalogText('')}
          />
        </>
      )
    }

    render(
      <QueryProvider>
        <ContextSetter />
        <VisualQueryBuilder />
      </QueryProvider>,
    )

    // Set then clear
    act(() => {
      fireEvent.click(screen.getByTestId('set-text'))
    })
    await act(async () => {
      vi.advanceTimersByTime(500)
    })
    expect(screen.getByText('Study')).toBeInTheDocument()

    act(() => {
      fireEvent.click(screen.getByTestId('clear-text'))
    })
    await act(async () => {
      vi.advanceTimersByTime(500)
    })

    // Builder should be reset to empty state
    expect(screen.queryByText('Study')).not.toBeInTheDocument()
    expect(screen.getByText(/Click a predicate in the sidebar/i)).toBeInTheDocument()
  })
})

// ---------------------------------------------------------------------------
// Desynchronized state tests
// ---------------------------------------------------------------------------

describe('BidirectionalSync – desynchronized state', () => {
  it('invalid Datalog sets isSynced to false after debounce', async () => {
    let syncState = { isSynced: true }

    function ContextMonitor(): React.ReactElement {
      const { isSynced, setDatalogText } = useQuery()
      syncState = { isSynced }
      return (
        <button
          data-testid="set-invalid"
          onClick={() => setDatalogText('this is not valid datalog')}
        />
      )
    }

    render(
      <QueryProvider>
        <ContextMonitor />
        <VisualQueryBuilder />
      </QueryProvider>,
    )

    act(() => {
      fireEvent.click(screen.getByTestId('set-invalid'))
    })

    // Before debounce, isSynced is still true
    expect(syncState.isSynced).toBe(true)

    await act(async () => {
      vi.advanceTimersByTime(500)
    })

    // After debounce, parse fails → isSynced = false
    expect(syncState.isSynced).toBe(false)
  })

  it('desync indicator is shown when isSynced is false', async () => {
    function ContextSetter(): React.ReactElement {
      const { setDatalogText } = useQuery()
      return (
        <button
          data-testid="set-invalid"
          onClick={() => setDatalogText('invalid syntax!!!')}
        />
      )
    }

    render(
      <QueryProvider>
        <ContextSetter />
        <VisualQueryBuilder />
      </QueryProvider>,
    )

    // No desync indicator before typing
    expect(screen.queryByTestId('desync-indicator')).not.toBeInTheDocument()

    act(() => {
      fireEvent.click(screen.getByTestId('set-invalid'))
    })

    await act(async () => {
      vi.advanceTimersByTime(500)
    })

    // Desync indicator should be visible
    expect(screen.getByTestId('desync-indicator')).toBeInTheDocument()
    expect(screen.getByTestId('desync-indicator')).toHaveTextContent(
      /out of sync/i,
    )
  })

  it('desync indicator disappears after typing valid Datalog', async () => {
    function ContextSetter(): React.ReactElement {
      const { setDatalogText } = useQuery()
      return (
        <>
          <button
            data-testid="set-invalid"
            onClick={() => setDatalogText('invalid!!!')}
          />
          <button
            data-testid="set-valid"
            onClick={() => setDatalogText('ans(s) :- Study(s).')}
          />
        </>
      )
    }

    render(
      <QueryProvider>
        <ContextSetter />
        <VisualQueryBuilder />
      </QueryProvider>,
    )

    // Trigger desync
    act(() => {
      fireEvent.click(screen.getByTestId('set-invalid'))
    })
    await act(async () => {
      vi.advanceTimersByTime(500)
    })
    expect(screen.getByTestId('desync-indicator')).toBeInTheDocument()

    // Now type valid Datalog
    act(() => {
      fireEvent.click(screen.getByTestId('set-valid'))
    })
    await act(async () => {
      vi.advanceTimersByTime(500)
    })

    // Desync indicator should be gone
    expect(screen.queryByTestId('desync-indicator')).not.toBeInTheDocument()
  })

  it('incomplete Datalog (missing period) shows desync indicator', async () => {
    function ContextSetter(): React.ReactElement {
      const { setDatalogText } = useQuery()
      return (
        <button
          data-testid="set-incomplete"
          onClick={() => setDatalogText('ans(s) :- Study(s)')} // missing period
        />
      )
    }

    render(
      <QueryProvider>
        <ContextSetter />
        <VisualQueryBuilder />
      </QueryProvider>,
    )

    act(() => {
      fireEvent.click(screen.getByTestId('set-incomplete'))
    })
    await act(async () => {
      vi.advanceTimersByTime(500)
    })

    expect(screen.getByTestId('desync-indicator')).toBeInTheDocument()
  })

  it('visual builder actions resync after desync', async () => {
    function TestComponent(): React.ReactElement {
      const { setDatalogText, model, refresh, isSynced } = useQuery()
      return (
        <>
          <span data-testid="sync-status">{isSynced ? 'synced' : 'desynced'}</span>
          <button
            data-testid="set-invalid"
            onClick={() => setDatalogText('bad text')}
          />
          <button
            data-testid="add-study"
            onClick={() => {
              model.addPredicate('Study', ['study'])
              refresh()
            }}
          />
          <VisualQueryBuilder />
        </>
      )
    }

    render(
      <QueryProvider>
        <TestComponent />
      </QueryProvider>,
    )

    // Trigger desync
    act(() => {
      fireEvent.click(screen.getByTestId('set-invalid'))
    })
    await act(async () => {
      vi.advanceTimersByTime(500)
    })
    expect(screen.getByTestId('sync-status')).toHaveTextContent('desynced')

    // Now add a predicate via visual builder → should resync
    act(() => {
      fireEvent.click(screen.getByTestId('add-study'))
    })
    expect(screen.getByTestId('sync-status')).toHaveTextContent('synced')
  })
})

// ---------------------------------------------------------------------------
// No infinite loop tests
// ---------------------------------------------------------------------------

describe('BidirectionalSync – no infinite loops', () => {
  it('does not cause infinite re-renders when typing valid Datalog', async () => {
    let renderCount = 0

    function RenderCounter(): React.ReactElement {
      renderCount++
      const { setDatalogText } = useQuery()
      return (
        <button
          data-testid="set-text"
          onClick={() => setDatalogText('ans(s) :- Study(s).')}
        />
      )
    }

    render(
      <QueryProvider>
        <RenderCounter />
        <VisualQueryBuilder />
      </QueryProvider>,
    )

    const beforeClick = renderCount

    act(() => {
      fireEvent.click(screen.getByTestId('set-text'))
    })
    await act(async () => {
      vi.advanceTimersByTime(500)
    })

    // Renders should be bounded (not exponential/infinite)
    // Allow a reasonable number of renders for state updates (< 20)
    const rendersDuringAction = renderCount - beforeClick
    expect(rendersDuringAction).toBeLessThan(20)
  })
})
