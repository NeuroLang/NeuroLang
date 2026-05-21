import { useContext } from 'react'
import EngineContext from './EngineContext'

interface EngineContextValue {
  selectedEngine: string | null
  setSelectedEngine: (engine: string | null) => void
}

export function useEngine(): EngineContextValue {
  const ctx = useContext(EngineContext)
  if (!ctx) {
    throw new Error('useEngine must be used within an EngineProvider')
  }
  return ctx
}
