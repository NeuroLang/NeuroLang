/**
 * ConnectionStatus.tsx
 *
 * A small indicator shown in the navbar that displays a green dot when the
 * backend is reachable and a red dot when disconnected. The status is driven
 * by the ConnectionContext which is updated by EngineSelector.
 */
import React from 'react'
import { useConnection } from '../context/useConnection'

function ConnectionStatus(): React.ReactElement {
  const { connectionState } = useConnection()

  const label =
    connectionState === 'connected'
      ? 'Connected'
      : connectionState === 'disconnected'
        ? 'Disconnected'
        : 'Connecting…'

  const colorClass =
    connectionState === 'connected'
      ? 'connection-status--connected'
      : connectionState === 'disconnected'
        ? 'connection-status--disconnected'
        : 'connection-status--unknown'

  return (
    <div
      className={`connection-status ${colorClass}`}
      title={label}
      aria-label={`Server status: ${label}`}
      data-testid="connection-status"
    >
      <span
        className="connection-status-dot"
        aria-hidden="true"
        data-testid="connection-status-dot"
      />
      <span className="connection-status-label">{label}</span>
    </div>
  )
}

export default ConnectionStatus
