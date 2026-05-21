import React, { useEffect, useState } from 'react'
import { useEngine } from '../context/useEngine'

interface EngineSelectorProps {
  className?: string
}

interface EnginesResponse {
  status: string
  data: string[]
}

function EngineSelector({
  className,
}: EngineSelectorProps): React.ReactElement {
  const { selectedEngine, setSelectedEngine } = useEngine()
  const [engines, setEngines] = useState<string[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    setLoading(true)
    setError(null)
    fetch('/v2/engines')
      .then((res) => {
        if (!res.ok) {
          throw new Error(`Failed to fetch engines: ${res.status}`)
        }
        return res.json() as Promise<EnginesResponse | string[]>
      })
      .then((data) => {
        // Handle both {"status":"ok","data":[...]} and plain array responses
        if (Array.isArray(data)) {
          setEngines(data)
        } else {
          setEngines(data.data)
        }
        setLoading(false)
      })
      .catch((err: unknown) => {
        const message =
          err instanceof Error ? err.message : 'Unknown error fetching engines'
        setError(message)
        setLoading(false)
      })
  }, [])

  if (loading) {
    return (
      <div className={`engine-selector ${className ?? ''}`}>
        <p className="engine-selector-loading">Loading engines...</p>
      </div>
    )
  }

  if (error) {
    return (
      <div className={`engine-selector ${className ?? ''}`}>
        <p className="engine-selector-error">{error}</p>
      </div>
    )
  }

  if (engines.length === 0) {
    return (
      <div className={`engine-selector ${className ?? ''}`}>
        <p className="engine-selector-empty">No engines available</p>
      </div>
    )
  }

  return (
    <div className={`engine-selector ${className ?? ''}`}>
      <ul className="engine-list" role="listbox" aria-label="Available engines">
        {engines.map((engine) => (
          <li
            key={engine}
            role="option"
            aria-selected={selectedEngine === engine}
            className={`engine-item ${selectedEngine === engine ? 'engine-item--active' : ''}`}
            onClick={() => setSelectedEngine(engine)}
          >
            {engine}
          </li>
        ))}
      </ul>
    </div>
  )
}

export default EngineSelector
