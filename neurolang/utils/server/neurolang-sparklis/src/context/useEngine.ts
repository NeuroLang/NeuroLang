import { useContext } from 'react'
import EngineContext, { type EngineContextValue } from './EngineContext'

export function useEngine(): EngineContextValue {
  const ctx = useContext(EngineContext)
  if (!ctx) {
    throw new Error('useEngine must be used within an EngineProvider')
  }
  return ctx
}
