/**
 * DatalogHelpTooltip.test.tsx
 *
 * Tests for the Datalog syntax help tooltip/popover.
 */
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, it, expect } from 'vitest'
import DatalogHelpTooltip from '../DatalogHelpTooltip'

describe('DatalogHelpTooltip', () => {
  it('renders the help button', () => {
    render(<DatalogHelpTooltip />)
    expect(screen.getByTestId('datalog-help-btn')).toBeInTheDocument()
    expect(
      screen.getByRole('button', { name: /datalog syntax help/i })
    ).toBeInTheDocument()
  })

  it('does not show popover initially', () => {
    render(<DatalogHelpTooltip />)
    expect(screen.queryByTestId('datalog-help-popover')).not.toBeInTheDocument()
  })

  it('shows popover on button click', async () => {
    render(<DatalogHelpTooltip />)
    await userEvent.click(screen.getByTestId('datalog-help-btn'))
    expect(screen.getByTestId('datalog-help-popover')).toBeInTheDocument()
  })

  it('shows Datalog syntax reference in popover', async () => {
    render(<DatalogHelpTooltip />)
    await userEvent.click(screen.getByTestId('datalog-help-btn'))

    // Should explain key syntax
    expect(screen.getByText(':-')).toBeInTheDocument()
    expect(screen.getByText(/means "if"/i)).toBeInTheDocument()
    expect(screen.getByText('&')).toBeInTheDocument()
    expect(screen.getByText(/means "and"/i)).toBeInTheDocument()
    expect(screen.getByText('~')).toBeInTheDocument()
    expect(screen.getByText(/means "not"/i)).toBeInTheDocument()
    expect(screen.getByText('PROB')).toBeInTheDocument()
    expect(screen.getByText(/probabilistic/i)).toBeInTheDocument()
  })

  it('closes popover when close button is clicked', async () => {
    render(<DatalogHelpTooltip />)
    await userEvent.click(screen.getByTestId('datalog-help-btn'))
    expect(screen.getByTestId('datalog-help-popover')).toBeInTheDocument()

    await userEvent.click(screen.getByTestId('datalog-help-close'))
    expect(screen.queryByTestId('datalog-help-popover')).not.toBeInTheDocument()
  })

  it('toggles popover on repeated clicks of help button', async () => {
    render(<DatalogHelpTooltip />)

    // Open
    await userEvent.click(screen.getByTestId('datalog-help-btn'))
    expect(screen.getByTestId('datalog-help-popover')).toBeInTheDocument()

    // Close via same button
    await userEvent.click(screen.getByTestId('datalog-help-btn'))
    expect(screen.queryByTestId('datalog-help-popover')).not.toBeInTheDocument()
  })

  it('closes popover when Escape key is pressed', async () => {
    render(<DatalogHelpTooltip />)
    await userEvent.click(screen.getByTestId('datalog-help-btn'))
    expect(screen.getByTestId('datalog-help-popover')).toBeInTheDocument()

    fireEvent.keyDown(document, { key: 'Escape' })
    await waitFor(() => {
      expect(
        screen.queryByTestId('datalog-help-popover')
      ).not.toBeInTheDocument()
    })
  })

  it('popover has role="dialog" for accessibility', async () => {
    render(<DatalogHelpTooltip />)
    await userEvent.click(screen.getByTestId('datalog-help-btn'))
    expect(
      screen.getByRole('dialog', { name: /datalog syntax reference/i })
    ).toBeInTheDocument()
  })

  it('help button has aria-expanded set correctly', async () => {
    render(<DatalogHelpTooltip />)
    const btn = screen.getByTestId('datalog-help-btn')

    expect(btn).toHaveAttribute('aria-expanded', 'false')
    await userEvent.click(btn)
    expect(btn).toHaveAttribute('aria-expanded', 'true')
    await userEvent.click(btn)
    expect(btn).toHaveAttribute('aria-expanded', 'false')
  })

  it('closes popover when clicking outside', async () => {
    render(
      <div>
        <DatalogHelpTooltip />
        <div data-testid="outside">Outside</div>
      </div>
    )

    await userEvent.click(screen.getByTestId('datalog-help-btn'))
    expect(screen.getByTestId('datalog-help-popover')).toBeInTheDocument()

    // Click outside the component
    fireEvent.mouseDown(screen.getByTestId('outside'))
    await waitFor(() => {
      expect(
        screen.queryByTestId('datalog-help-popover')
      ).not.toBeInTheDocument()
    })
  })
})
