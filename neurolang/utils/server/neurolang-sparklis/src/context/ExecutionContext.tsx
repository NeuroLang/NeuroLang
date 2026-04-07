/**
 * ExecutionContext.tsx
 *
 * React context that manages the lifecycle of a query execution:
 *   - idle: no query running
 *   - running: query submitted, waiting for result
 *   - done: query completed successfully (results available)
 *   - error: query completed with an error
 *   - cancelled: user cancelled the running query
 *
 * The QueryExecutor service lives here and manages the WebSocket connection.
 * Components consume this context to render loading states, results, errors,
 * and the cancel button.
 */
import React, {
  createContext,
  useCallback,
  useRef,
  useState,
} from 'react'

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/** Line/column information for parser errors. */
export interface ErrorLineInfo {
  line: number
  col: number
  text: string
}

/** Details of a query error. */
export interface QueryError {
  /** The Python exception class name (e.g., "<class 'neurolang.exceptions.ParserError'>"). */
  errorName: string
  /** Human-readable error message. */
  message: string
  /** Full error doc string (may contain more detail). */
  errorDoc?: string
  /** Optional line/column info for parser errors. */
  line_info?: ErrorLineInfo
}

/** The status of the current query execution. */
export type ExecutionStatus = 'idle' | 'running' | 'done' | 'error' | 'cancelled'

/**
 * Raw WebSocket response payload from /v1/statementsocket.
 * The server sends JSON: { status: "ok", data: QueryResults }
 */
export interface QueryResultsData {
  uuid: string
  cancelled: boolean
  running: boolean
  done: boolean
  errorName?: string
  message?: string
  errorDoc?: string
  line_info?: ErrorLineInfo
  results?: Record<string, unknown>
}

export interface WebSocketMessage {
  status: string
  data?: QueryResultsData
}

/** The full execution state exposed by the context. */
export interface ExecutionContextValue {
  /** Current execution status. */
  executionStatus: ExecutionStatus
  /** Error details when executionStatus === 'error'. */
  queryError: QueryError | null
  /** Raw query results data when executionStatus === 'done'. */
  queryResults: Record<string, unknown> | null
  /** UUID of the last submitted query (for result polling/cancellation). */
  queryUuid: string | null
  /** Submit a query. Opens a new WebSocket connection. */
  submitQuery: (query: string, engine: string) => void
  /** Cancel the currently running query. */
  cancelQuery: () => void
  /** Reset state back to idle. */
  resetExecution: () => void
}

// ---------------------------------------------------------------------------
// Context
// ---------------------------------------------------------------------------

const ExecutionContext = createContext<ExecutionContextValue | null>(null)

// ---------------------------------------------------------------------------
// Provider
// ---------------------------------------------------------------------------

/**
 * Determines the WebSocket URL for /v1/statementsocket.
 * In development (Vite proxy), we use a relative ws:// URL that routes through
 * the dev server's proxy.  In production (served from Tornado), the page
 * origin IS the backend.
 */
function getWebSocketUrl(): string {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  const host = window.location.host
  return `${protocol}//${host}/v1/statementsocket`
}

export function ExecutionProvider({
  children,
}: {
  children: React.ReactNode
}): React.ReactElement {
  const [executionStatus, setExecutionStatus] =
    useState<ExecutionStatus>('idle')
  const [queryError, setQueryError] = useState<QueryError | null>(null)
  const [queryResults, setQueryResults] = useState<Record<
    string,
    unknown
  > | null>(null)
  const [queryUuid, setQueryUuid] = useState<string | null>(null)

  /** The active WebSocket connection (if any). */
  const wsRef = useRef<WebSocket | null>(null)
  /** UUID of the currently running query (for cancellation). */
  const currentUuidRef = useRef<string | null>(null)
  /**
   * Guards against the onerror handler overwriting 'cancelled' with 'error'.
   * ws.close() in cancelQuery() asynchronously fires onerror in some browsers,
   * so we set this ref to true before closing to suppress that transition.
   */
  const isCancellingRef = useRef<boolean>(false)

  // ---------------------------------------------------------------------------
  // cancelQuery – close the WebSocket and send a cancel HTTP request
  // ---------------------------------------------------------------------------
  const cancelQuery = useCallback(() => {
    // Set cancelling flag BEFORE closing the socket so onerror cannot
    // overwrite the 'cancelled' status with 'error'.
    isCancellingRef.current = true

    // Close the WebSocket (server will clean up).
    if (wsRef.current) {
      wsRef.current.close()
      wsRef.current = null
    }

    // If we have a UUID, call the cancel endpoint.
    const uuid = currentUuidRef.current
    if (uuid) {
      fetch(`/v1/cancel/${uuid}`, { method: 'DELETE' }).catch(() => {
        // Best-effort; ignore errors
      })
    }

    setExecutionStatus('cancelled')
  }, [])

  // ---------------------------------------------------------------------------
  // resetExecution – reset to idle
  // ---------------------------------------------------------------------------
  const resetExecution = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close()
      wsRef.current = null
    }
    currentUuidRef.current = null
    isCancellingRef.current = false
    setExecutionStatus('idle')
    setQueryError(null)
    setQueryResults(null)
    setQueryUuid(null)
  }, [])

  // ---------------------------------------------------------------------------
  // submitQuery – opens WebSocket, sends query, handles responses
  // ---------------------------------------------------------------------------
  const submitQuery = useCallback(
    (query: string, engine: string) => {
      // Close any existing connection.
      if (wsRef.current) {
        wsRef.current.close()
        wsRef.current = null
      }

      // Reset state and move to running.
      setExecutionStatus('running')
      setQueryError(null)
      setQueryResults(null)
      setQueryUuid(null)
      currentUuidRef.current = null
      isCancellingRef.current = false

      const wsUrl = getWebSocketUrl()
      let ws: WebSocket

      try {
        ws = new WebSocket(wsUrl)
      } catch {
        setExecutionStatus('error')
        setQueryError({
          errorName: 'ConnectionError',
          message: 'Could not connect to the backend WebSocket.',
        })
        return
      }

      wsRef.current = ws

      ws.onopen = () => {
        // Send the query message once the connection is open.
        ws.send(JSON.stringify({ query, engine }))
      }

      ws.onmessage = (event: MessageEvent) => {
        let msg: WebSocketMessage
        try {
          msg = JSON.parse(event.data as string) as WebSocketMessage
        } catch {
          // Malformed message – ignore
          return
        }

        const data = msg.data

        if (!data) {
          // Status-only message (e.g. initial "ok" before running)
          return
        }

        // Track UUID for cancellation.
        if (data.uuid && !currentUuidRef.current) {
          currentUuidRef.current = data.uuid
          setQueryUuid(data.uuid)
        }

        if (data.cancelled) {
          ws.close()
          wsRef.current = null
          setExecutionStatus('cancelled')
          return
        }

        if (data.running) {
          // Still running – keep status as running (no-op state change needed)
          setExecutionStatus('running')
          return
        }

        if (data.done) {
          ws.close()
          wsRef.current = null

          if (data.errorName) {
            // Query completed with error
            setQueryError({
              errorName: data.errorName,
              message: data.message ?? 'An error occurred.',
              errorDoc: data.errorDoc,
              line_info: data.line_info,
            })
            setExecutionStatus('error')
          } else {
            // Query completed successfully
            setQueryResults(
              (data.results as Record<string, unknown>) ?? {},
            )
            setExecutionStatus('done')
          }
          return
        }
      }

      ws.onerror = () => {
        // If we are in the middle of cancelling, ignore the error — ws.close()
        // can fire onerror asynchronously in some browsers.
        if (isCancellingRef.current) {
          return
        }
        wsRef.current = null
        setExecutionStatus('error')
        setQueryError({
          errorName: 'WebSocketError',
          message:
            'A WebSocket error occurred. The backend may be unavailable.',
        })
      }

      ws.onclose = (event: CloseEvent) => {
        if (wsRef.current === ws) {
          wsRef.current = null
        }
        // If the socket closed unexpectedly while still running, treat as error.
        if (!event.wasClean && executionStatus === 'running') {
          setExecutionStatus('error')
          setQueryError({
            errorName: 'ConnectionClosed',
            message: 'The WebSocket connection was closed unexpectedly.',
          })
        }
      }
    },
    // executionStatus is NOT in deps to avoid re-creating on every status change
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [],
  )

  const value: ExecutionContextValue = {
    executionStatus,
    queryError,
    queryResults,
    queryUuid,
    submitQuery,
    cancelQuery,
    resetExecution,
  }

  return (
    <ExecutionContext.Provider value={value}>
      {children}
    </ExecutionContext.Provider>
  )
}

export default ExecutionContext
