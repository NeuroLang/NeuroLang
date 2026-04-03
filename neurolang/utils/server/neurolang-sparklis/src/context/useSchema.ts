/**
 * useSchema.ts
 *
 * Hook to consume the SchemaContext.
 */
import { useContext } from 'react'
import SchemaContext, { SchemaContextValue } from './SchemaContext'

export function useSchema(): SchemaContextValue {
  const ctx = useContext(SchemaContext)
  if (!ctx) {
    throw new Error('useSchema must be used within a SchemaProvider')
  }
  return ctx
}
