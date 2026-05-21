/**
 * useExecution.ts
 *
 * Hook to consume the ExecutionContext.
 */
import { useContext } from 'react'
import ExecutionContext, { ExecutionContextValue } from './ExecutionContext'

export function useExecution(): ExecutionContextValue {
  const ctx = useContext(ExecutionContext)
  if (!ctx) {
    throw new Error('useExecution must be used within an ExecutionProvider')
  }
  return ctx
}
