/**
 * QueryExecution.test.tsx
 *
 * Tests for the query execution flow:
 *   1. ExecutionContext: state transitions (idle → running → done/error/cancelled)
 *   2. RunQueryButton: rendering, click submits query, loading/cancel states
 *   3. ErrorDisplay: rendering for error state, cancelled state, idle (null)
 *   4. Error line highlighting callback
 *   5. WebSocket mock for all states
 *
 * Uses a mock WebSocket class to avoid real network calls.
 */
import React, { useEffect } from 'react'
import { render, screen, fireEvent, act, waitFor } from '@testing-library/react'
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import {
  ExecutionProvider,
  ExecutionContextValue,
} from '../../context/ExecutionContext'
import { useExecution } from '../../context/useExecution'
import { QueryProvider } from '../../context/QueryContext'
import { EngineProvider } from '../../context/EngineContext'
import { useQuery } from '../../context/useQuery'
import { useEngine } from '../../context/useEngine'
import RunQueryButton from '../RunQueryButton'
import ErrorDisplay from '../ErrorDisplay'

// ---------------------------------------------------------------------------
// Mock WebSocket
// ---------------------------------------------------------------------------

/**
 * A mock WebSocket class for testing. Captures the last instance created so
 * tests can trigger events manually.
 */
class MockWebSocket {
  static lastInstance: MockWebSocket | null = null

  url: string
  onopen: ((event: Event) => void) | null = null
  onmessage: ((event: MessageEvent) => void) | null = null
  onerror: ((event: Event) => void) | null = null
  onclose: ((event: CloseEvent) => void) | null = null

  sentMessages: string[] = []
  closed = false

  constructor(url: string) {
    this.url = url
    MockWebSocket.lastInstance = this
  }

  send(data: string) {
    this.sentMessages.push(data)
  }

  close() {
    this.closed = true
    // Simulate close event
    const closeEvent = new CloseEvent('close', { wasClean: true })
    this.onclose?.(closeEvent)
  }

  // Test helpers to trigger events from outside
  triggerOpen() {
    this.onopen?.(new Event('open'))
  }

  triggerMessage(data: unknown) {
    const event = new MessageEvent('message', {
      data: JSON.stringify(data),
    })
    this.onmessage?.(event)
  }

  triggerError() {
    this.onerror?.(new Event('error'))
  }

  triggerCloseUnclean() {
    const closeEvent = new CloseEvent('close', { wasClean: false })
    this.onclose?.(closeEvent)
  }
}

// ---------------------------------------------------------------------------
// Helper components & wrappers
// ---------------------------------------------------------------------------

/** Provider stack for most tests. */
function TestProviders({
  children,
}: {
  children: React.ReactNode
}): React.ReactElement {
  return (
    <EngineProvider>
      <QueryProvider>
        <ExecutionProvider>{children}</ExecutionProvider>
      </QueryProvider>
    </EngineProvider>
  )
}

/** Exposes ExecutionContext value via a spy function. */
function ExecutionStateReader({
  onRender,
}: {
  onRender: (val: ExecutionContextValue) => void
}): React.ReactElement {
  const val = useExecution()
  onRender(val)
  return <></>
}

/** Sets the engine via EngineContext on mount. */
function EngineInitializer({
  engine,
}: {
  engine: string
}): React.ReactElement {
  const { setSelectedEngine } = useEngine()
  useEffect(() => {
    setSelectedEngine(engine)
  }, [engine, setSelectedEngine])
  return <></>
}

/** Sets Datalog text in QueryContext on mount. */
function QueryInitializer({
  query,
}: {
  query: string
}): React.ReactElement {
  const { setDatalogText } = useQuery()
  useEffect(() => {
    setDatalogText(query)
  }, [query, setDatalogText])
  return <></>
}

/** Renders RunQueryButton with engine + query set up. */
function renderRunQueryButton(engine = 'neurosynth', query = 'ans(x) :- Study(x).') {
  return render(
    <TestProviders>
      <EngineInitializer engine={engine} />
      <QueryInitializer query={query} />
      <RunQueryButton />
    </TestProviders>,
  )
}

// ---------------------------------------------------------------------------
// Setup/teardown
// ---------------------------------------------------------------------------

beforeEach(() => {
  MockWebSocket.lastInstance = null
  vi.stubGlobal('WebSocket', MockWebSocket)
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, json: async () => ({ cancelled: true }) }))
})

afterEach(() => {
  vi.restoreAllMocks()
})

// ---------------------------------------------------------------------------
// ExecutionContext: initial state
// ---------------------------------------------------------------------------

describe('ExecutionContext – initial state', () => {
  it('starts with idle status', () => {
    let captured: ExecutionContextValue | null = null

    render(
      <TestProviders>
        <ExecutionStateReader onRender={(v) => { captured = v }} />
      </TestProviders>,
    )

    expect(captured!.executionStatus).toBe('idle')
  })

  it('starts with null queryError', () => {
    let captured: ExecutionContextValue | null = null

    render(
      <TestProviders>
        <ExecutionStateReader onRender={(v) => { captured = v }} />
      </TestProviders>,
    )

    expect(captured!.queryError).toBeNull()
  })

  it('starts with null queryResults', () => {
    let captured: ExecutionContextValue | null = null

    render(
      <TestProviders>
        <ExecutionStateReader onRender={(v) => { captured = v }} />
      </TestProviders>,
    )

    expect(captured!.queryResults).toBeNull()
  })
})

// ---------------------------------------------------------------------------
// ExecutionContext: submitQuery transitions to running
// ---------------------------------------------------------------------------

describe('ExecutionContext – submitQuery', () => {
  it('transitions to running when submitQuery is called', () => {
    let captured: ExecutionContextValue | null = null

    render(
      <TestProviders>
        <ExecutionStateReader onRender={(v) => { captured = v }} />
      </TestProviders>,
    )

    act(() => {
      captured!.submitQuery('ans(x) :- Study(x).', 'neurosynth')
    })

    expect(captured!.executionStatus).toBe('running')
  })

  it('opens a WebSocket connection on submit', () => {
    let captured: ExecutionContextValue | null = null

    render(
      <TestProviders>
        <ExecutionStateReader onRender={(v) => { captured = v }} />
      </TestProviders>,
    )

    act(() => {
      captured!.submitQuery('ans(x) :- Study(x).', 'neurosynth')
    })

    expect(MockWebSocket.lastInstance).not.toBeNull()
    expect(MockWebSocket.lastInstance!.url).toContain('/v1/statementsocket')
  })

  it('sends the query message when the WebSocket opens', () => {
    let captured: ExecutionContextValue | null = null

    render(
      <TestProviders>
        <ExecutionStateReader onRender={(v) => { captured = v }} />
      </TestProviders>,
    )

    act(() => {
      captured!.submitQuery('ans(x) :- Study(x).', 'neurosynth')
    })

    const ws = MockWebSocket.lastInstance!

    act(() => {
      ws.triggerOpen()
    })

    expect(ws.sentMessages).toHaveLength(1)
    const msg = JSON.parse(ws.sentMessages[0])
    expect(msg).toMatchObject({
      query: 'ans(x) :- Study(x).',
      engine: 'neurosynth',
    })
  })
})

// ---------------------------------------------------------------------------
// ExecutionContext: WebSocket message handling – running state
// ---------------------------------------------------------------------------

describe('ExecutionContext – running state message', () => {
  it('stays in running state when server sends running=true', () => {
    let captured: ExecutionContextValue | null = null

    render(
      <TestProviders>
        <ExecutionStateReader onRender={(v) => { captured = v }} />
      </TestProviders>,
    )

    act(() => {
      captured!.submitQuery('ans(x) :- Study(x).', 'neurosynth')
    })
    const ws = MockWebSocket.lastInstance!
    act(() => { ws.triggerOpen() })

    act(() => {
      ws.triggerMessage({
        status: 'ok',
        data: { uuid: 'test-uuid', cancelled: false, running: true, done: false },
      })
    })

    expect(captured!.executionStatus).toBe('running')
  })

  it('captures the uuid from running message', () => {
    let captured: ExecutionContextValue | null = null

    render(
      <TestProviders>
        <ExecutionStateReader onRender={(v) => { captured = v }} />
      </TestProviders>,
    )

    act(() => {
      captured!.submitQuery('ans(x) :- Study(x).', 'neurosynth')
    })
    const ws = MockWebSocket.lastInstance!
    act(() => { ws.triggerOpen() })

    act(() => {
      ws.triggerMessage({
        status: 'ok',
        data: { uuid: 'my-query-uuid', cancelled: false, running: true, done: false },
      })
    })

    expect(captured!.queryUuid).toBe('my-query-uuid')
  })
})

// ---------------------------------------------------------------------------
// ExecutionContext: WebSocket message handling – done with results
// ---------------------------------------------------------------------------

describe('ExecutionContext – done (success) state', () => {
  it('transitions to done when server sends done=true with results', () => {
    let captured: ExecutionContextValue | null = null

    render(
      <TestProviders>
        <ExecutionStateReader onRender={(v) => { captured = v }} />
      </TestProviders>,
    )

    act(() => {
      captured!.submitQuery('ans(x) :- Study(x).', 'neurosynth')
    })
    const ws = MockWebSocket.lastInstance!
    act(() => { ws.triggerOpen() })

    act(() => {
      ws.triggerMessage({
        status: 'ok',
        data: {
          uuid: 'done-uuid',
          cancelled: false,
          running: false,
          done: true,
          results: { ans: { columns: ['x'], size: 1 } },
        },
      })
    })

    expect(captured!.executionStatus).toBe('done')
  })

  it('stores queryResults when done with results', () => {
    let captured: ExecutionContextValue | null = null

    render(
      <TestProviders>
        <ExecutionStateReader onRender={(v) => { captured = v }} />
      </TestProviders>,
    )

    act(() => {
      captured!.submitQuery('ans(x) :- Study(x).', 'neurosynth')
    })
    const ws = MockWebSocket.lastInstance!
    act(() => { ws.triggerOpen() })

    const mockResults = { ans: { columns: ['x'], size: 3 } }
    act(() => {
      ws.triggerMessage({
        status: 'ok',
        data: {
          uuid: 'done-uuid',
          cancelled: false,
          running: false,
          done: true,
          results: mockResults,
        },
      })
    })

    expect(captured!.queryResults).toEqual(mockResults)
  })

  it('queryError is null after successful execution', () => {
    let captured: ExecutionContextValue | null = null

    render(
      <TestProviders>
        <ExecutionStateReader onRender={(v) => { captured = v }} />
      </TestProviders>,
    )

    act(() => {
      captured!.submitQuery('ans(x) :- Study(x).', 'neurosynth')
    })
    const ws = MockWebSocket.lastInstance!
    act(() => { ws.triggerOpen() })

    act(() => {
      ws.triggerMessage({
        status: 'ok',
        data: {
          uuid: 'done-uuid',
          cancelled: false,
          running: false,
          done: true,
          results: {},
        },
      })
    })

    expect(captured!.queryError).toBeNull()
  })
})

// ---------------------------------------------------------------------------
// ExecutionContext: WebSocket message handling – done with error
// ---------------------------------------------------------------------------

describe('ExecutionContext – done (error) state', () => {
  it('transitions to error when server sends done=true with errorName', () => {
    let captured: ExecutionContextValue | null = null

    render(
      <TestProviders>
        <ExecutionStateReader onRender={(v) => { captured = v }} />
      </TestProviders>,
    )

    act(() => {
      captured!.submitQuery('invalid query', 'neurosynth')
    })
    const ws = MockWebSocket.lastInstance!
    act(() => { ws.triggerOpen() })

    act(() => {
      ws.triggerMessage({
        status: 'ok',
        data: {
          uuid: 'err-uuid',
          cancelled: false,
          running: false,
          done: true,
          errorName: "<class 'neurolang.exceptions.ParserError'>",
          message: 'An error occurred while parsing your query.',
          line_info: { line: 1, col: 5, text: 'Expected predicate' },
        },
      })
    })

    expect(captured!.executionStatus).toBe('error')
  })

  it('stores queryError with full details', () => {
    let captured: ExecutionContextValue | null = null

    render(
      <TestProviders>
        <ExecutionStateReader onRender={(v) => { captured = v }} />
      </TestProviders>,
    )

    act(() => {
      captured!.submitQuery('bad query', 'neurosynth')
    })
    const ws = MockWebSocket.lastInstance!
    act(() => { ws.triggerOpen() })

    act(() => {
      ws.triggerMessage({
        status: 'ok',
        data: {
          uuid: 'err-uuid',
          cancelled: false,
          running: false,
          done: true,
          errorName: "<class 'neurolang.exceptions.ParserError'>",
          message: 'An error occurred while parsing your query.',
          errorDoc: 'Parser error details',
          line_info: { line: 1, col: 5, text: 'Expected predicate' },
        },
      })
    })

    expect(captured!.queryError).not.toBeNull()
    expect(captured!.queryError!.errorName).toContain('ParserError')
    expect(captured!.queryError!.message).toBe(
      'An error occurred while parsing your query.',
    )
    expect(captured!.queryError!.line_info?.line).toBe(1)
    expect(captured!.queryError!.line_info?.col).toBe(5)
  })

  it('queryResults is null after error', () => {
    let captured: ExecutionContextValue | null = null

    render(
      <TestProviders>
        <ExecutionStateReader onRender={(v) => { captured = v }} />
      </TestProviders>,
    )

    act(() => {
      captured!.submitQuery('bad query', 'neurosynth')
    })
    const ws = MockWebSocket.lastInstance!
    act(() => { ws.triggerOpen() })

    act(() => {
      ws.triggerMessage({
        status: 'ok',
        data: {
          uuid: 'err-uuid',
          cancelled: false,
          running: false,
          done: true,
          errorName: "<class 'Exception'>",
          message: 'Something went wrong',
        },
      })
    })

    expect(captured!.queryResults).toBeNull()
  })
})

// ---------------------------------------------------------------------------
// ExecutionContext: cancelled state
// ---------------------------------------------------------------------------

describe('ExecutionContext – cancelled state', () => {
  it('transitions to cancelled when server sends cancelled=true', () => {
    let captured: ExecutionContextValue | null = null

    render(
      <TestProviders>
        <ExecutionStateReader onRender={(v) => { captured = v }} />
      </TestProviders>,
    )

    act(() => {
      captured!.submitQuery('ans(x) :- Study(x).', 'neurosynth')
    })
    const ws = MockWebSocket.lastInstance!
    act(() => { ws.triggerOpen() })

    act(() => {
      ws.triggerMessage({
        status: 'ok',
        data: { uuid: 'cancel-uuid', cancelled: true, running: false, done: false },
      })
    })

    expect(captured!.executionStatus).toBe('cancelled')
  })

  it('cancelQuery() transitions to cancelled state', () => {
    let captured: ExecutionContextValue | null = null

    render(
      <TestProviders>
        <ExecutionStateReader onRender={(v) => { captured = v }} />
      </TestProviders>,
    )

    act(() => {
      captured!.submitQuery('ans(x) :- Study(x).', 'neurosynth')
    })

    act(() => {
      captured!.cancelQuery()
    })

    expect(captured!.executionStatus).toBe('cancelled')
  })

  it('resetExecution() returns to idle state', () => {
    let captured: ExecutionContextValue | null = null

    render(
      <TestProviders>
        <ExecutionStateReader onRender={(v) => { captured = v }} />
      </TestProviders>,
    )

    act(() => {
      captured!.submitQuery('ans(x) :- Study(x).', 'neurosynth')
    })
    act(() => {
      captured!.cancelQuery()
    })
    expect(captured!.executionStatus).toBe('cancelled')

    act(() => {
      captured!.resetExecution()
    })
    expect(captured!.executionStatus).toBe('idle')
  })
})

// ---------------------------------------------------------------------------
// RunQueryButton – rendering
// ---------------------------------------------------------------------------

describe('RunQueryButton – rendering', () => {
  it('renders the Run Query button initially (when engine and query set)', async () => {
    renderRunQueryButton()
    await waitFor(() => {
      expect(screen.getByTestId('run-query-btn')).toBeInTheDocument()
    })
    expect(screen.getByTestId('run-query-btn')).toHaveTextContent(/Run Query/i)
  })

  it('button is disabled when no engine is selected', async () => {
    render(
      <TestProviders>
        {/* No EngineInitializer – engine stays null */}
        <QueryInitializer query="ans(x) :- Study(x)." />
        <RunQueryButton />
      </TestProviders>,
    )
    await waitFor(() => {
      expect(screen.getByTestId('run-query-btn')).toBeDisabled()
    })
  })

  it('button is disabled when query is empty', async () => {
    render(
      <TestProviders>
        <EngineInitializer engine="neurosynth" />
        <QueryInitializer query="" />
        <RunQueryButton />
      </TestProviders>,
    )
    await waitFor(() => {
      expect(screen.getByTestId('run-query-btn')).toBeDisabled()
    })
  })

  it('button is enabled when engine and non-empty query are set', async () => {
    renderRunQueryButton()
    await waitFor(() => {
      const btn = screen.getByTestId('run-query-btn')
      expect(btn).not.toBeDisabled()
    })
  })
})

// ---------------------------------------------------------------------------
// RunQueryButton – click behavior
// ---------------------------------------------------------------------------

describe('RunQueryButton – click behavior', () => {
  it('clicking Run Query opens a WebSocket', async () => {
    renderRunQueryButton()
    await waitFor(() => {
      expect(screen.getByTestId('run-query-btn')).not.toBeDisabled()
    })

    act(() => {
      fireEvent.click(screen.getByTestId('run-query-btn'))
    })

    expect(MockWebSocket.lastInstance).not.toBeNull()
  })

  it('shows loading state (spinner + cancel button) after clicking Run', async () => {
    renderRunQueryButton()
    await waitFor(() => {
      expect(screen.getByTestId('run-query-btn')).not.toBeDisabled()
    })

    act(() => {
      fireEvent.click(screen.getByTestId('run-query-btn'))
    })

    // Running state: spinner button + cancel button visible
    expect(screen.getByTestId('run-query-btn-running')).toBeInTheDocument()
    expect(screen.getByTestId('cancel-query-btn')).toBeInTheDocument()
    // Original run button should be gone
    expect(screen.queryByTestId('run-query-btn')).not.toBeInTheDocument()
  })

  it('clicking Cancel transitions to cancelled state', async () => {
    renderRunQueryButton()
    await waitFor(() => {
      expect(screen.getByTestId('run-query-btn')).not.toBeDisabled()
    })

    act(() => {
      fireEvent.click(screen.getByTestId('run-query-btn'))
    })

    expect(screen.getByTestId('cancel-query-btn')).toBeInTheDocument()

    act(() => {
      fireEvent.click(screen.getByTestId('cancel-query-btn'))
    })

    // After cancel, Run Query button is shown again
    await waitFor(() => {
      expect(screen.getByTestId('run-query-controls')).toBeInTheDocument()
    })
  })

  it('run button returns after query completes successfully', async () => {
    renderRunQueryButton()
    await waitFor(() => {
      expect(screen.getByTestId('run-query-btn')).not.toBeDisabled()
    })

    act(() => {
      fireEvent.click(screen.getByTestId('run-query-btn'))
    })

    const ws = MockWebSocket.lastInstance!
    act(() => { ws.triggerOpen() })

    act(() => {
      ws.triggerMessage({
        status: 'ok',
        data: {
          uuid: 'done',
          cancelled: false,
          running: false,
          done: true,
          results: {},
        },
      })
    })

    await waitFor(() => {
      expect(screen.queryByTestId('run-query-btn-running')).not.toBeInTheDocument()
      expect(screen.getByTestId('run-query-btn')).toBeInTheDocument()
    })
  })
})

// ---------------------------------------------------------------------------
// ErrorDisplay – rendering
// ---------------------------------------------------------------------------

describe('ErrorDisplay – rendering', () => {
  it('renders nothing when status is idle', () => {
    render(
      <TestProviders>
        <ErrorDisplay />
      </TestProviders>,
    )
    expect(screen.queryByTestId('query-error-display')).not.toBeInTheDocument()
    expect(screen.queryByTestId('query-cancelled-message')).not.toBeInTheDocument()
  })

  it('renders nothing when status is running', () => {
    let captured: ExecutionContextValue | null = null

    render(
      <TestProviders>
        <ExecutionStateReader onRender={(v) => { captured = v }} />
        <ErrorDisplay />
      </TestProviders>,
    )

    act(() => {
      captured!.submitQuery('ans(x) :- Study(x).', 'neurosynth')
    })

    expect(screen.queryByTestId('query-error-display')).not.toBeInTheDocument()
    expect(screen.queryByTestId('query-cancelled-message')).not.toBeInTheDocument()
  })

  it('renders error display when status is error', () => {
    let captured: ExecutionContextValue | null = null

    render(
      <TestProviders>
        <ExecutionStateReader onRender={(v) => { captured = v }} />
        <ErrorDisplay />
      </TestProviders>,
    )

    act(() => {
      captured!.submitQuery('bad query', 'neurosynth')
    })
    const ws = MockWebSocket.lastInstance!
    act(() => { ws.triggerOpen() })

    act(() => {
      ws.triggerMessage({
        status: 'ok',
        data: {
          uuid: 'err',
          cancelled: false,
          running: false,
          done: true,
          errorName: "<class 'Exception'>",
          message: 'Something failed',
        },
      })
    })

    expect(screen.getByTestId('query-error-display')).toBeInTheDocument()
  })

  it('renders error name and message in error state', () => {
    let captured: ExecutionContextValue | null = null

    render(
      <TestProviders>
        <ExecutionStateReader onRender={(v) => { captured = v }} />
        <ErrorDisplay />
      </TestProviders>,
    )

    act(() => {
      captured!.submitQuery('bad query', 'neurosynth')
    })
    const ws = MockWebSocket.lastInstance!
    act(() => { ws.triggerOpen() })

    act(() => {
      ws.triggerMessage({
        status: 'ok',
        data: {
          uuid: 'err',
          cancelled: false,
          running: false,
          done: true,
          errorName: "<class 'neurolang.exceptions.ParserError'>",
          message: 'Parse error on line 2',
        },
      })
    })

    expect(screen.getByTestId('error-name')).toHaveTextContent('ParserError')
    expect(screen.getByTestId('error-message')).toHaveTextContent(
      'Parse error on line 2',
    )
  })

  it('renders line info when available', () => {
    let captured: ExecutionContextValue | null = null

    render(
      <TestProviders>
        <ExecutionStateReader onRender={(v) => { captured = v }} />
        <ErrorDisplay />
      </TestProviders>,
    )

    act(() => {
      captured!.submitQuery('bad query', 'neurosynth')
    })
    const ws = MockWebSocket.lastInstance!
    act(() => { ws.triggerOpen() })

    act(() => {
      ws.triggerMessage({
        status: 'ok',
        data: {
          uuid: 'err',
          cancelled: false,
          running: false,
          done: true,
          errorName: "<class 'neurolang.exceptions.ParserError'>",
          message: 'Parse error',
          line_info: { line: 3, col: 7, text: 'Unexpected token' },
        },
      })
    })

    expect(screen.getByTestId('error-location')).toBeInTheDocument()
    expect(screen.getByTestId('error-location')).toHaveTextContent('Line 3')
    expect(screen.getByTestId('error-location')).toHaveTextContent('column 7')
  })

  it('calls onHighlightError with line/col when error has line info', () => {
    const onHighlight = vi.fn()
    let captured: ExecutionContextValue | null = null

    render(
      <TestProviders>
        <ExecutionStateReader onRender={(v) => { captured = v }} />
        <ErrorDisplay onHighlightError={onHighlight} />
      </TestProviders>,
    )

    act(() => {
      captured!.submitQuery('bad query', 'neurosynth')
    })
    const ws = MockWebSocket.lastInstance!
    act(() => { ws.triggerOpen() })

    act(() => {
      ws.triggerMessage({
        status: 'ok',
        data: {
          uuid: 'err',
          cancelled: false,
          running: false,
          done: true,
          errorName: "<class 'neurolang.exceptions.ParserError'>",
          message: 'Parse error',
          line_info: { line: 2, col: 4, text: 'Unexpected token' },
        },
      })
    })

    expect(onHighlight).toHaveBeenCalledWith(2, 4)
  })

  it('does NOT call onHighlightError when error has no line info', () => {
    const onHighlight = vi.fn()
    let captured: ExecutionContextValue | null = null

    render(
      <TestProviders>
        <ExecutionStateReader onRender={(v) => { captured = v }} />
        <ErrorDisplay onHighlightError={onHighlight} />
      </TestProviders>,
    )

    act(() => {
      captured!.submitQuery('bad query', 'neurosynth')
    })
    const ws = MockWebSocket.lastInstance!
    act(() => { ws.triggerOpen() })

    act(() => {
      ws.triggerMessage({
        status: 'ok',
        data: {
          uuid: 'err',
          cancelled: false,
          running: false,
          done: true,
          errorName: "<class 'Exception'>",
          message: 'Generic error',
        },
      })
    })

    expect(onHighlight).not.toHaveBeenCalled()
  })

  it('renders cancelled message when status is cancelled', () => {
    let captured: ExecutionContextValue | null = null

    render(
      <TestProviders>
        <ExecutionStateReader onRender={(v) => { captured = v }} />
        <ErrorDisplay />
      </TestProviders>,
    )

    act(() => {
      captured!.submitQuery('ans(x) :- Study(x).', 'neurosynth')
    })
    act(() => {
      captured!.cancelQuery()
    })

    expect(screen.getByTestId('query-cancelled-message')).toBeInTheDocument()
    expect(screen.getByTestId('query-cancelled-message')).toHaveTextContent(
      /cancelled/i,
    )
  })
})

// ---------------------------------------------------------------------------
// ExecutionContext: WebSocket error
// ---------------------------------------------------------------------------

describe('ExecutionContext – WebSocket error', () => {
  it('transitions to error on WebSocket error', () => {
    let captured: ExecutionContextValue | null = null

    render(
      <TestProviders>
        <ExecutionStateReader onRender={(v) => { captured = v }} />
      </TestProviders>,
    )

    act(() => {
      captured!.submitQuery('ans(x) :- Study(x).', 'neurosynth')
    })

    const ws = MockWebSocket.lastInstance!
    act(() => { ws.triggerError() })

    expect(captured!.executionStatus).toBe('error')
    expect(captured!.queryError).not.toBeNull()
    expect(captured!.queryError!.errorName).toBe('WebSocketError')
  })
})

// ---------------------------------------------------------------------------
// ExecutionContext: cancel race condition guard
// ---------------------------------------------------------------------------

describe('ExecutionContext – cancel race condition guard (isCancellingRef)', () => {
  it('onerror after cancelQuery() does NOT overwrite cancelled status with error', () => {
    /**
     * Regression test for the cancel race condition:
     * cancelQuery() calls ws.close() which may asynchronously fire onerror
     * in some browsers. Without the isCancellingRef guard, the onerror handler
     * would overwrite executionStatus from 'cancelled' to 'error'.
     */
    let captured: ExecutionContextValue | null = null

    render(
      <TestProviders>
        <ExecutionStateReader onRender={(v) => { captured = v }} />
      </TestProviders>,
    )

    act(() => {
      captured!.submitQuery('ans(x) :- Study(x).', 'neurosynth')
    })

    const ws = MockWebSocket.lastInstance!
    act(() => { ws.triggerOpen() })

    // Cancel the query (this sets isCancellingRef.current = true)
    act(() => {
      captured!.cancelQuery()
    })

    // Simulate onerror firing after cancel (race condition scenario)
    act(() => {
      ws.triggerError()
    })

    // Status should remain 'cancelled', not 'error'
    expect(captured!.executionStatus).toBe('cancelled')
    expect(captured!.queryError).toBeNull()
  })

  it('onerror after cancelQuery() does NOT set queryError', () => {
    let captured: ExecutionContextValue | null = null

    render(
      <TestProviders>
        <ExecutionStateReader onRender={(v) => { captured = v }} />
      </TestProviders>,
    )

    act(() => {
      captured!.submitQuery('ans(x) :- Study(x).', 'neurosynth')
    })

    const ws = MockWebSocket.lastInstance!
    act(() => { ws.triggerOpen() })

    act(() => {
      captured!.cancelQuery()
    })

    act(() => {
      ws.triggerError()
    })

    expect(captured!.queryError).toBeNull()
  })

  it('onerror without prior cancel DOES set error status (non-cancel errors still work)', () => {
    /**
     * Verify that the isCancellingRef guard does not suppress genuine
     * WebSocket errors that occur outside of a cancellation flow.
     */
    let captured: ExecutionContextValue | null = null

    render(
      <TestProviders>
        <ExecutionStateReader onRender={(v) => { captured = v }} />
      </TestProviders>,
    )

    act(() => {
      captured!.submitQuery('ans(x) :- Study(x).', 'neurosynth')
    })

    const ws = MockWebSocket.lastInstance!
    act(() => { ws.triggerOpen() })

    // Trigger error WITHOUT cancelling first
    act(() => {
      ws.triggerError()
    })

    expect(captured!.executionStatus).toBe('error')
    expect(captured!.queryError).not.toBeNull()
    expect(captured!.queryError!.errorName).toBe('WebSocketError')
  })

  it('after cancelQuery(), new submitQuery() clears isCancellingRef so errors are reported', () => {
    /**
     * After cancelling, isCancellingRef is reset to false by submitQuery().
     * This ensures a subsequent query's genuine errors are not suppressed.
     */
    let captured: ExecutionContextValue | null = null

    render(
      <TestProviders>
        <ExecutionStateReader onRender={(v) => { captured = v }} />
      </TestProviders>,
    )

    // First query: submit then cancel
    act(() => {
      captured!.submitQuery('ans(x) :- Study(x).', 'neurosynth')
    })
    act(() => {
      captured!.cancelQuery()
    })
    expect(captured!.executionStatus).toBe('cancelled')

    // Second query: submit (should reset isCancellingRef)
    act(() => {
      captured!.submitQuery('ans(x) :- Study(x).', 'neurosynth')
    })
    const ws2 = MockWebSocket.lastInstance!
    act(() => { ws2.triggerOpen() })

    // Genuine error on the new query should be reported
    act(() => {
      ws2.triggerError()
    })

    expect(captured!.executionStatus).toBe('error')
    expect(captured!.queryError).not.toBeNull()
    expect(captured!.queryError!.errorName).toBe('WebSocketError')
  })
})
