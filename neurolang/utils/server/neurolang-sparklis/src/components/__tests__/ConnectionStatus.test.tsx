/**
 * ConnectionStatus.test.tsx
 *
 * Tests for the ConnectionStatus navbar indicator.
 */
import { render, screen } from '@testing-library/react'
import { describe, it, expect } from 'vitest'
import ConnectionStatus from '../ConnectionStatus'
import {
  ConnectionProvider,
  ConnectionContextValue,
} from '../../context/ConnectionContext'
import ConnectionContext from '../../context/ConnectionContext'

// Helper to render with a specific connection state
function renderWithState(state: 'connected' | 'disconnected' | 'unknown') {
  const value: ConnectionContextValue = {
    connectionState: state,
    setConnected: () => undefined,
    setDisconnected: () => undefined,
  }
  return render(
    <ConnectionContext.Provider value={value}>
      <ConnectionStatus />
    </ConnectionContext.Provider>
  )
}

describe('ConnectionStatus', () => {
  it('renders connection status indicator', () => {
    render(
      <ConnectionProvider>
        <ConnectionStatus />
      </ConnectionProvider>
    )
    expect(screen.getByTestId('connection-status')).toBeInTheDocument()
  })

  it('shows "Connected" when connection state is connected', () => {
    renderWithState('connected')
    expect(screen.getByTestId('connection-status')).toHaveTextContent(
      'Connected'
    )
  })

  it('shows "Disconnected" when connection state is disconnected', () => {
    renderWithState('disconnected')
    expect(screen.getByTestId('connection-status')).toHaveTextContent(
      'Disconnected'
    )
  })

  it('shows "Connecting…" when connection state is unknown', () => {
    renderWithState('unknown')
    expect(screen.getByTestId('connection-status')).toHaveTextContent(
      'Connecting'
    )
  })

  it('applies connected CSS class when connected', () => {
    renderWithState('connected')
    expect(screen.getByTestId('connection-status')).toHaveClass(
      'connection-status--connected'
    )
  })

  it('applies disconnected CSS class when disconnected', () => {
    renderWithState('disconnected')
    expect(screen.getByTestId('connection-status')).toHaveClass(
      'connection-status--disconnected'
    )
  })

  it('applies unknown CSS class when unknown', () => {
    renderWithState('unknown')
    expect(screen.getByTestId('connection-status')).toHaveClass(
      'connection-status--unknown'
    )
  })

  it('has appropriate aria-label', () => {
    renderWithState('connected')
    expect(screen.getByTestId('connection-status')).toHaveAttribute(
      'aria-label',
      'Server status: Connected'
    )
  })

  it('renders the dot element', () => {
    renderWithState('connected')
    expect(screen.getByTestId('connection-status-dot')).toBeInTheDocument()
  })
})
