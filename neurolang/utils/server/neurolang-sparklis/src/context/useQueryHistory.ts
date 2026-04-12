/**
 * useQueryHistory.ts
 *
 * Hook to consume the QueryHistoryContext.
 */
import { useContext } from 'react'
import QueryHistoryContext, {
  QueryHistoryContextValue,
} from './QueryHistoryContext'

export function useQueryHistory(): QueryHistoryContextValue {
  const ctx = useContext(QueryHistoryContext)
  if (!ctx) {
    throw new Error(
      'useQueryHistory must be used within a QueryHistoryProvider',
    )
  }
  return ctx
}
