import React, { createContext, useState } from 'react'

export interface EngineContextValue {
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

export default EngineContext
