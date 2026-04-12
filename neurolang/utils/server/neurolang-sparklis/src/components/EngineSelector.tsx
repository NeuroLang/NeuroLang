import React, { useEffect, useState, useCallback } from 'react'
import { useEngine } from '../context/useEngine'
import { useConnection } from '../context/useConnection'

interface EngineSelectorProps {
  className?: string
}

interface EnginesResponse {
  status: string
  data: string[]
}

/** Skeleton placeholder for the engine list while loading. */
function EngineListSkeleton(): React.ReactElement {
  return (
    <div
      className="engine-selector-skeleton"
      aria-busy="true"
      aria-label="Loading engines"
      data-testid="engine-list-skeleton"
    >
      {[1, 2].map((i) => (
        <div key={i} className="skeleton skeleton--engine-item" />
      ))}
    </div>
  )
}

function EngineSelector({
  className,
}: EngineSelectorProps): React.ReactElement {
  const { selectedEngine, switchEngine } = useEngine()
  const { setConnected, setDisconnected } = useConnection()
  const [engines, setEngines] = useState<string[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [isConnectionError, setIsConnectionError] = useState(false)

  const fetchEngines = useCallback((): void => {
    setLoading(true)
    setError(null)
    setIsConnectionError(false)

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
        setConnected()
        setLoading(false)
      })
      .catch((err: unknown) => {
        const message =
          err instanceof Error ? err.message : 'Unknown error fetching engines'
        // Detect network/connection errors:
        // - TypeError from fetch() means the server is unreachable (network error)
        // - The canonical browser message is exactly "Failed to fetch"
        const isNetworkError =
          err instanceof TypeError &&
          (message === 'Failed to fetch' ||
            message.toLowerCase().includes('networkerror') ||
            message.toLowerCase().includes('network request failed'))
        setIsConnectionError(isNetworkError)
        setError(message)
        setDisconnected()
        setLoading(false)
      })
  }, [setConnected, setDisconnected])

  useEffect(() => {
    fetchEngines()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  if (loading) {
    return (
      <div className={`engine-selector ${className ?? ''}`}>
        <EngineListSkeleton />
      </div>
    )
  }

  if (error) {
    if (isConnectionError) {
      return (
        <div
          className={`engine-selector ${className ?? ''}`}
          data-testid="connection-error"
        >
          <p
            className="connection-error-message"
            data-testid="connection-error-message"
          >
            Cannot connect to server
          </p>
          <p className="connection-error-hint">
            Make sure the NeuroLang backend is running.
          </p>
          <button
            className="connection-retry-btn"
            onClick={fetchEngines}
            data-testid="connection-retry-btn"
          >
            Retry
          </button>
        </div>
      )
    }
    return (
      <div className={`engine-selector ${className ?? ''}`}>
        <p className="engine-selector-error">{error}</p>
        <button
          className="connection-retry-btn"
          onClick={fetchEngines}
          data-testid="connection-retry-btn"
        >
          Retry
        </button>
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
            onClick={() => switchEngine(engine)}
          >
            {engine}
          </li>
        ))}
      </ul>
    </div>
  )
}

export default EngineSelector
