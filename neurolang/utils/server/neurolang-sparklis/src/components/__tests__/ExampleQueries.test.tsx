/**
 * ExampleQueries.test.tsx
 *
 * Tests for the ExampleQueries component.
 *
 * Covers:
 *   1. Renders a list of example queries fetched from the API
 *   2. Clicking an example loads its query into the code editor (via context)
 *   3. Description section is expandable/collapsible
 *   4. Loading and error states
 *   5. No examples state
 */
import React from 'react'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import ExampleQueries from '../ExampleQueries'
import { EngineProvider } from '../../context/EngineContext'
import { QueryProvider } from '../../context/QueryContext'
import { useEngine } from '../../context/useEngine'
import { useQuery } from '../../context/useQuery'

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface Example {
  id: string
  title: string
  shortTitle: string
  query: string
  description: string
}

// ---------------------------------------------------------------------------
// Mock data
// ---------------------------------------------------------------------------

const MOCK_EXAMPLES: Example[] = [
  {
    id: 'neuro1',
    title: 'Coordinate-based meta-analysis (CBMA) on the Neurosynth database',
    shortTitle: 'CBMA Single Term',
    query: 'TermInStudy(term, study) :- TermInStudyTFIDF(term, tfidf, study)',
    description:
      'This example uses the Neurosynth CBMA database to query activations.',
  },
  {
    id: 'neuro2',
    title: 'CBMA Multiple Terms',
    shortTitle: 'CBMA Multiple Terms',
    query: 'TermsToSelect("emotion")\nTermsToSelect("fear")',
    description: 'This example queries multiple term associations.',
  },
]

function makeExamplesResponse(examples: Example[]) {
  return {
    ok: true,
    json: async () => ({ status: 'ok', data: examples }),
  } as Response
}

// ---------------------------------------------------------------------------
// Helper: render with required contexts + optional engine preset
// ---------------------------------------------------------------------------

/**
 * Wrapper that renders ExampleQueries inside all required contexts.
 * Optionally sets the selected engine imperatively.
 */
function renderWithContexts(
  ui: React.ReactElement,
  selectedEngine: string | null = null,
) {
  function EngineSetterWrapper() {
    const { setSelectedEngine } = useEngine()
    React.useEffect(() => {
      if (selectedEngine !== null) {
        setSelectedEngine(selectedEngine)
      }
    }, [setSelectedEngine])
    return ui
  }

  return render(
    <EngineProvider>
      <QueryProvider>
        <EngineSetterWrapper />
      </QueryProvider>
    </EngineProvider>,
  )
}

/**
 * Helper component to read datalogText from context for test assertions.
 */
function DatalogReader({ onRead }: { onRead: (text: string) => void }) {
  const { datalogText } = useQuery()
  React.useEffect(() => {
    onRead(datalogText)
  })
  return null
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('ExampleQueries', () => {
  beforeEach(() => {
    vi.stubGlobal('fetch', vi.fn())
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  // -------------------------------------------------------------------------
  // No engine selected
  // -------------------------------------------------------------------------

  it('renders nothing (or placeholder) when no engine is selected', async () => {
    // With no engine selected, the component should not fetch or display examples
    renderWithContexts(<ExampleQueries />, null)
    // Should not show any example short titles
    expect(screen.queryByText('CBMA Single Term')).not.toBeInTheDocument()
  })

  // -------------------------------------------------------------------------
  // Loading state
  // -------------------------------------------------------------------------

  it('shows a loading state while fetching examples', async () => {
    vi.mocked(fetch).mockReturnValue(new Promise(() => {})) // never resolves
    renderWithContexts(<ExampleQueries />, 'neurosynth')
    await waitFor(() => {
      expect(
        screen.getByText(/loading examples/i),
      ).toBeInTheDocument()
    })
  })

  // -------------------------------------------------------------------------
  // Renders examples list
  // -------------------------------------------------------------------------

  it('fetches examples from /v2/examples/:engine on mount', async () => {
    vi.mocked(fetch).mockResolvedValue(makeExamplesResponse(MOCK_EXAMPLES))
    renderWithContexts(<ExampleQueries />, 'neurosynth')

    expect(fetch).toHaveBeenCalledWith('/v2/examples/neurosynth')
    await waitFor(() => {
      expect(screen.getByText('CBMA Single Term')).toBeInTheDocument()
    })
  })

  it('renders all example short titles', async () => {
    vi.mocked(fetch).mockResolvedValue(makeExamplesResponse(MOCK_EXAMPLES))
    renderWithContexts(<ExampleQueries />, 'neurosynth')

    await waitFor(() => {
      expect(screen.getByText('CBMA Single Term')).toBeInTheDocument()
      expect(screen.getByText('CBMA Multiple Terms')).toBeInTheDocument()
    })
  })

  it('re-fetches when engine changes', async () => {
    vi.mocked(fetch).mockResolvedValue(makeExamplesResponse(MOCK_EXAMPLES))

    // Render with a wrapper that can change engine
    function EngineChanger() {
      const { setSelectedEngine } = useEngine()
      return (
        <div>
          <button onClick={() => setSelectedEngine('destrieux')}>
            Switch to destrieux
          </button>
          <ExampleQueries />
        </div>
      )
    }

    render(
      <EngineProvider>
        <QueryProvider>
          <EngineChanger />
        </QueryProvider>
      </EngineProvider>,
    )

    // Initial fetch with no engine – nothing displayed
    expect(fetch).not.toHaveBeenCalled()

    // Switch to destrieux
    await userEvent.click(screen.getByText('Switch to destrieux'))

    await waitFor(() => {
      expect(fetch).toHaveBeenCalledWith('/v2/examples/destrieux')
    })
  })

  // -------------------------------------------------------------------------
  // Click loads query into editor
  // -------------------------------------------------------------------------

  it('clicking an example loads its query text via setDatalogText', async () => {
    vi.mocked(fetch).mockResolvedValue(makeExamplesResponse(MOCK_EXAMPLES))

    let capturedText = ''
    const handleRead = (text: string) => {
      capturedText = text
    }

    render(
      <EngineProvider>
        <QueryProvider>
          <>
            <ExampleEngineWrapper>
              <ExampleQueries />
            </ExampleEngineWrapper>
            <DatalogReader onRead={handleRead} />
          </>
        </QueryProvider>
      </EngineProvider>,
    )

    await waitFor(() => {
      expect(screen.getByText('CBMA Single Term')).toBeInTheDocument()
    })

    // Click the first example
    await userEvent.click(screen.getByText('CBMA Single Term'))

    await waitFor(() => {
      expect(capturedText).toBe(MOCK_EXAMPLES[0].query)
    })
  })

  // -------------------------------------------------------------------------
  // Description expandable
  // -------------------------------------------------------------------------

  it('description is not visible initially', async () => {
    vi.mocked(fetch).mockResolvedValue(makeExamplesResponse(MOCK_EXAMPLES))
    renderWithContexts(<ExampleQueries />, 'neurosynth')

    await waitFor(() => {
      expect(screen.getByText('CBMA Single Term')).toBeInTheDocument()
    })

    // Description text should not be visible initially
    expect(
      screen.queryByText(/Neurosynth CBMA database/),
    ).not.toBeInTheDocument()
  })

  it('clicking info/expand button reveals the description', async () => {
    vi.mocked(fetch).mockResolvedValue(makeExamplesResponse(MOCK_EXAMPLES))
    renderWithContexts(<ExampleQueries />, 'neurosynth')

    await waitFor(() => {
      expect(screen.getByText('CBMA Single Term')).toBeInTheDocument()
    })

    // Find and click the expand button for the first example
    // The button should be accessible by role or test-id
    const expandButtons = screen.getAllByRole('button', { name: /info|expand|description|▼|▶/i })
    await userEvent.click(expandButtons[0])

    await waitFor(() => {
      expect(
        screen.getByText(/Neurosynth CBMA database/),
      ).toBeInTheDocument()
    })
  })

  it('clicking the expand button again hides the description', async () => {
    vi.mocked(fetch).mockResolvedValue(makeExamplesResponse(MOCK_EXAMPLES))
    renderWithContexts(<ExampleQueries />, 'neurosynth')

    await waitFor(() => {
      expect(screen.getByText('CBMA Single Term')).toBeInTheDocument()
    })

    const expandButtons = screen.getAllByRole('button', { name: /info|expand|description|▼|▶/i })
    // Expand
    await userEvent.click(expandButtons[0])
    await waitFor(() => {
      expect(screen.getByText(/Neurosynth CBMA database/)).toBeInTheDocument()
    })

    // Collapse
    await userEvent.click(expandButtons[0])
    await waitFor(() => {
      expect(
        screen.queryByText(/Neurosynth CBMA database/),
      ).not.toBeInTheDocument()
    })
  })

  // -------------------------------------------------------------------------
  // Error state
  // -------------------------------------------------------------------------

  it('shows an error message when fetch fails', async () => {
    vi.mocked(fetch).mockRejectedValue(new Error('Network error'))
    renderWithContexts(<ExampleQueries />, 'neurosynth')

    await waitFor(() => {
      expect(screen.getByText(/error|failed/i)).toBeInTheDocument()
    })
  })

  // -------------------------------------------------------------------------
  // Empty state
  // -------------------------------------------------------------------------

  it('shows a placeholder when the engine has no examples', async () => {
    vi.mocked(fetch).mockResolvedValue(makeExamplesResponse([]))
    renderWithContexts(<ExampleQueries />, 'neurosynth')

    await waitFor(() => {
      expect(screen.getByText(/no examples/i)).toBeInTheDocument()
    })
  })

  // -------------------------------------------------------------------------
  // Panel collapsibility
  // -------------------------------------------------------------------------

  it('examples are shown in a collapsible panel', async () => {
    vi.mocked(fetch).mockResolvedValue(makeExamplesResponse(MOCK_EXAMPLES))
    renderWithContexts(<ExampleQueries />, 'neurosynth')

    await waitFor(() => {
      expect(screen.getByText('CBMA Single Term')).toBeInTheDocument()
    })

    // Find the panel toggle button (labeled "Examples" or similar)
    const panelToggle = screen.getByRole('button', { name: /examples/i })
    expect(panelToggle).toBeInTheDocument()

    // Clicking it should collapse the list
    await userEvent.click(panelToggle)

    await waitFor(() => {
      expect(screen.queryByText('CBMA Single Term')).not.toBeInTheDocument()
    })
  })
})

// ---------------------------------------------------------------------------
// Helper wrapper component for engine context
// ---------------------------------------------------------------------------

function ExampleEngineWrapper({ children }: { children: React.ReactNode }) {
  const { setSelectedEngine } = useEngine()
  React.useEffect(() => {
    setSelectedEngine('neurosynth')
  }, [setSelectedEngine])
  return <>{children}</>
}
