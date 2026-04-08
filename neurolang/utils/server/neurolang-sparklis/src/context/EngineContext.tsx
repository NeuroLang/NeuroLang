import React, { createContext, useCallback, useRef, useState } from 'react'

/** Duration (ms) to display the "Switching engine…" banner. */
const SWITCHING_DURATION_MS = 800

export interface EngineContextValue {
  selectedEngine: string | null
  /** Whether an engine switch transition is in progress. */
  isSwitching: boolean
  /**
   * Update the selected engine.  Prefer `switchEngine` for user-initiated
   * engine changes; `setSelectedEngine` is kept for programmatic use (e.g.,
   * permalink loading) where no transition banner is needed.
   */
  setSelectedEngine: (engine: string | null) => void
  /**
   * Switch to a new engine.  Briefly sets `isSwitching = true` so UI can
   * display a transition banner, then updates `selectedEngine`.
   */
  switchEngine: (engine: string) => void
}

const EngineContext = createContext<EngineContextValue | null>(null)

export function EngineProvider({
  children,
}: {
  children: React.ReactNode
}): React.ReactElement {
  const [selectedEngine, setSelectedEngine] = useState<string | null>(null)
  const [isSwitching, setIsSwitching] = useState(false)
  const switchTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  const switchEngine = useCallback((engine: string) => {
    // Don't switch if it's the same engine already selected.
    setSelectedEngine((prev) => {
      if (prev === engine) return prev
      // Show transition banner.
      setIsSwitching(true)
      if (switchTimerRef.current !== null) {
        clearTimeout(switchTimerRef.current)
      }
      switchTimerRef.current = setTimeout(() => {
        setIsSwitching(false)
        switchTimerRef.current = null
      }, SWITCHING_DURATION_MS)
      return engine
    })
  }, [])

  return (
    <EngineContext.Provider
      value={{ selectedEngine, isSwitching, setSelectedEngine, switchEngine }}
    >
      {children}
    </EngineContext.Provider>
  )
}

export default EngineContext
