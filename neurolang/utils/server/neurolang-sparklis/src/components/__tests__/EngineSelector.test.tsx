import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import EngineSelector from '../EngineSelector'
import { EngineProvider } from '../../context/EngineContext'

// Helper to wrap component with required context
function renderWithContext(ui: React.ReactElement) {
  return render(<EngineProvider>{ui}</EngineProvider>)
}

// API response format: {"status":"ok","data":[...]}
function makeEnginesResponse(engines: string[]) {
  return {
    ok: true,
    json: async () => ({ status: 'ok', data: engines }),
  } as Response
}

describe('EngineSelector', () => {
  beforeEach(() => {
    vi.stubGlobal('fetch', vi.fn())
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('shows loading state initially', () => {
    vi.mocked(fetch).mockReturnValue(new Promise(() => {})) // never resolves
    renderWithContext(<EngineSelector />)
    expect(screen.getByText('Loading engines...')).toBeInTheDocument()
  })

  it('fetches engines from /v2/engines on mount', async () => {
    vi.mocked(fetch).mockResolvedValue(
      makeEnginesResponse(['neurosynth', 'destrieux'])
    )

    renderWithContext(<EngineSelector />)

    expect(fetch).toHaveBeenCalledWith('/v2/engines')
    await waitFor(() => {
      expect(screen.getByText('neurosynth')).toBeInTheDocument()
      expect(screen.getByText('destrieux')).toBeInTheDocument()
    })
  })

  it('renders engine list with correct items', async () => {
    vi.mocked(fetch).mockResolvedValue(
      makeEnginesResponse(['neurosynth', 'destrieux'])
    )

    renderWithContext(<EngineSelector />)

    await waitFor(() => {
      const items = screen.getAllByRole('option')
      expect(items).toHaveLength(2)
      expect(items[0]).toHaveTextContent('neurosynth')
      expect(items[1]).toHaveTextContent('destrieux')
    })
  })

  it('clicking an engine selects it and highlights it as active', async () => {
    vi.mocked(fetch).mockResolvedValue(
      makeEnginesResponse(['neurosynth', 'destrieux'])
    )

    renderWithContext(<EngineSelector />)

    await waitFor(() => {
      expect(screen.getByText('neurosynth')).toBeInTheDocument()
    })

    const neurosynthItem = screen.getByText('neurosynth')
    await userEvent.click(neurosynthItem)

    expect(neurosynthItem.closest('li')).toHaveClass('engine-item--active')
  })

  it('highlights only the selected engine', async () => {
    vi.mocked(fetch).mockResolvedValue(
      makeEnginesResponse(['neurosynth', 'destrieux'])
    )

    renderWithContext(<EngineSelector />)

    await waitFor(() => {
      expect(screen.getByText('neurosynth')).toBeInTheDocument()
    })

    // Click neurosynth first
    await userEvent.click(screen.getByText('neurosynth'))
    expect(screen.getByText('neurosynth').closest('li')).toHaveClass(
      'engine-item--active'
    )
    expect(screen.getByText('destrieux').closest('li')).not.toHaveClass(
      'engine-item--active'
    )

    // Click destrieux - should transfer highlight
    await userEvent.click(screen.getByText('destrieux'))
    expect(screen.getByText('destrieux').closest('li')).toHaveClass(
      'engine-item--active'
    )
    expect(screen.getByText('neurosynth').closest('li')).not.toHaveClass(
      'engine-item--active'
    )
  })

  it('sets aria-selected on the active engine', async () => {
    vi.mocked(fetch).mockResolvedValue(
      makeEnginesResponse(['neurosynth', 'destrieux'])
    )

    renderWithContext(<EngineSelector />)

    await waitFor(() => {
      expect(screen.getByText('neurosynth')).toBeInTheDocument()
    })

    await userEvent.click(screen.getByText('neurosynth'))

    const neurosynthItem = screen.getByText('neurosynth').closest('li')
    const destrieuxItem = screen.getByText('destrieux').closest('li')
    expect(neurosynthItem).toHaveAttribute('aria-selected', 'true')
    expect(destrieuxItem).toHaveAttribute('aria-selected', 'false')
  })

  it('shows error message when fetch fails', async () => {
    vi.mocked(fetch).mockRejectedValue(new Error('Network error'))

    renderWithContext(<EngineSelector />)

    await waitFor(() => {
      expect(screen.getByText('Network error')).toBeInTheDocument()
    })
  })

  it('shows error message when response is not ok', async () => {
    vi.mocked(fetch).mockResolvedValue({
      ok: false,
      status: 500,
    } as Response)

    renderWithContext(<EngineSelector />)

    await waitFor(() => {
      expect(
        screen.getByText('Failed to fetch engines: 500')
      ).toBeInTheDocument()
    })
  })

  it('shows empty message when no engines are returned', async () => {
    vi.mocked(fetch).mockResolvedValue(makeEnginesResponse([]))

    renderWithContext(<EngineSelector />)

    await waitFor(() => {
      expect(screen.getByText('No engines available')).toBeInTheDocument()
    })
  })

  it('handles plain array response format', async () => {
    vi.mocked(fetch).mockResolvedValue({
      ok: true,
      json: async () => ['neurosynth', 'destrieux'],
    } as Response)

    renderWithContext(<EngineSelector />)

    await waitFor(() => {
      expect(screen.getByText('neurosynth')).toBeInTheDocument()
      expect(screen.getByText('destrieux')).toBeInTheDocument()
    })
  })
})

