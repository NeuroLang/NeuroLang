import { render, screen, waitFor } from '@testing-library/react'
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import App from '../App'

describe('App', () => {
  beforeEach(() => {
    // Mock fetch so EngineSelector doesn't fail
    // API returns {"status":"ok","data":[...]} format
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({
        ok: true,
        json: async () => ({ status: 'ok', data: ['neurosynth', 'destrieux'] }),
      } as Response)
    )
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('renders without crashing', async () => {
    render(<App />)
    await waitFor(() =>
      expect(screen.getByText('NeuroLang Sparklis')).toBeInTheDocument()
    )
  })

  it('renders the navbar with title', async () => {
    render(<App />)
    await waitFor(() => {
      const navbar = screen.getByRole('banner')
      expect(navbar).toBeInTheDocument()
      expect(screen.getByText('NeuroLang Sparklis')).toBeInTheDocument()
    })
  })

  it('renders sidebar with engine section', async () => {
    render(<App />)
    await waitFor(() => {
      expect(screen.getByText('Engines')).toBeInTheDocument()
    })
  })

  it('renders sidebar with predicates section', async () => {
    render(<App />)
    await waitFor(() => {
      expect(screen.getByText('Predicates')).toBeInTheDocument()
      expect(screen.getByText('Engine not selected')).toBeInTheDocument()
    })
  })

  it('renders main content area with welcome message when no engine selected', async () => {
    render(<App />)
    await waitFor(() => {
      expect(screen.getByRole('main')).toBeInTheDocument()
      expect(
        screen.getByText(/Welcome to NeuroLang Sparklis/i)
      ).toBeInTheDocument()
    })
  })
})
