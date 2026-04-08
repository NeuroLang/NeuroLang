/**
 * ConnectionContext.tsx
 *
 * React context that tracks the backend connection status. The EngineSelector
 * updates this context when it succeeds or fails to fetch the engine list.
 * The navbar's ConnectionStatus indicator consumes it to show a green/red dot.
 */
import React, { createContext, useState, useCallback } from 'react'

export type ConnectionState = 'connected' | 'disconnected' | 'unknown'

export interface ConnectionContextValue {
  /** Current connection state. */
  connectionState: ConnectionState
  /** Called on successful API contact. */
  setConnected: () => void
  /** Called when API is unreachable. */
  setDisconnected: () => void
}

const ConnectionContext = createContext<ConnectionContextValue | null>(null)

export function ConnectionProvider({
  children,
}: {
  children: React.ReactNode
}): React.ReactElement {
  const [connectionState, setConnectionState] =
    useState<ConnectionState>('unknown')

  const setConnected = useCallback((): void => {
    setConnectionState('connected')
  }, [])

  const setDisconnected = useCallback((): void => {
    setConnectionState('disconnected')
  }, [])

  return (
    <ConnectionContext.Provider
      value={{ connectionState, setConnected, setDisconnected }}
    >
      {children}
    </ConnectionContext.Provider>
  )
}

export default ConnectionContext
