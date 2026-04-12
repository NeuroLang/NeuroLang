/**
 * useConnection.ts
 *
 * Hook to consume the ConnectionContext.
 */
import { useContext } from 'react'
import ConnectionContext from './ConnectionContext'
import type { ConnectionContextValue } from './ConnectionContext'

export function useConnection(): ConnectionContextValue {
  const ctx = useContext(ConnectionContext)
  if (!ctx) {
    throw new Error('useConnection must be used within a ConnectionProvider')
  }
  return ctx
}
