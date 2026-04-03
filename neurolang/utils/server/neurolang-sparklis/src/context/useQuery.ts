/**
 * useQuery.ts
 *
 * Hook to consume the QueryContext.
 */
import { useContext } from 'react'
import QueryContext, { QueryContextValue } from './QueryContext'

export function useQuery(): QueryContextValue {
  const ctx = useContext(QueryContext)
  if (!ctx) {
    throw new Error('useQuery must be used within a QueryProvider')
  }
  return ctx
}
