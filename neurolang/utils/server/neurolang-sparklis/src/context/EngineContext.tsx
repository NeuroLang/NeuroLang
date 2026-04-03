import React, { createContext, useContext, useState } from 'react'

interface EngineContextValue {
  selectedEngine: string | null
  setSelectedEngine: (engine: string | null) => void
}

const EngineContext = createContext<EngineContextValue | null>(null)

export function EngineProvider({
  children,
}: {
  children: React.ReactNode
}): React.ReactElement {
  const [selectedEngine, setSelectedEngine] = useState<string | null>(null)

  return (
    <EngineContext.Provider value={{ selectedEngine, setSelectedEngine }}>
      {children}
    </EngineContext.Provider>
  )
}

export function useEngine(): EngineContextValue {
  const ctx = useContext(EngineContext)
  if (!ctx) {
    throw new Error('useEngine must be used within an EngineProvider')
  }
  return ctx
}

export default EngineContext
