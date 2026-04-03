import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import PredicateBrowser from '../PredicateBrowser'
import { EngineProvider } from '../../context/EngineContext'
import { useEngine } from '../../context/useEngine'
import React from 'react'

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function renderWithContext(
  ui: React.ReactElement,
  initialEngine: string | null = null,
) {
  // Helper to set engine imperatively via a child component
  function EngineSetterWrapper() {
    const { setSelectedEngine } = useEngine()
    React.useEffect(() => {
      if (initialEngine !== null) {
        setSelectedEngine(initialEngine)
      }
    }, [setSelectedEngine])
    return ui
  }

  return render(
    <EngineProvider>
      <EngineSetterWrapper />
    </EngineProvider>,
    { wrapper: undefined },
  )
}

const MOCK_SCHEMA = {
  relations: [
    { name: 'PeakReported', type: 'relation', params: ['x', 'y', 'z', 's'] },
    { name: 'Study', type: 'relation', params: ['id'] },
  ],
  functions: [
    {
      name: 'agg_create_region',
      type: 'function',
      params: ['x', 'y', 'z'],
      docstring: 'Aggregate regions',
    },
  ],
  probabilistic: [
    { name: 'SelectedStudy', type: 'probabilistic', params: ['id'] },
  ],
}

function makeSchemaResponse(data = MOCK_SCHEMA) {
  return {
    ok: true,
    json: async () => ({ status: 'ok', data }),
  } as Response
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('PredicateBrowser', () => {
  beforeEach(() => {
    vi.stubGlobal('fetch', vi.fn())
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  // -------------------------------------------------------------------------
  // No engine selected
  // -------------------------------------------------------------------------

  it('shows engine-not-selected placeholder when no engine is selected', () => {
    render(
      <EngineProvider>
        <PredicateBrowser />
      </EngineProvider>,
    )
    expect(screen.getByText('Engine not selected')).toBeInTheDocument()
    expect(fetch).not.toHaveBeenCalled()
  })

  // -------------------------------------------------------------------------
  // Loading state
  // -------------------------------------------------------------------------

  it('shows loading state while fetching schema', async () => {
    // Never resolves so it stays in loading state
    vi.mocked(fetch).mockReturnValue(new Promise(() => {}))

    renderWithContext(<PredicateBrowser />, 'neurosynth')

    await waitFor(() => {
      expect(screen.getByText('Loading predicates...')).toBeInTheDocument()
    })
  })

  // -------------------------------------------------------------------------
  // Groups rendered correctly
  // -------------------------------------------------------------------------

  it('renders symbol groups after schema loads', async () => {
    vi.mocked(fetch).mockResolvedValue(makeSchemaResponse())

    renderWithContext(<PredicateBrowser />, 'neurosynth')

    await waitFor(() => {
      expect(screen.getByText('Relations')).toBeInTheDocument()
      expect(screen.getByText('Functions')).toBeInTheDocument()
      expect(screen.getByText('Probabilistic')).toBeInTheDocument()
    })
  })

  it('fetches schema from /v2/schema/:engine', async () => {
    vi.mocked(fetch).mockResolvedValue(makeSchemaResponse())

    renderWithContext(<PredicateBrowser />, 'neurosynth')

    await waitFor(() => {
      expect(fetch).toHaveBeenCalledWith('/v2/schema/neurosynth')
    })
  })

  it('shows symbol name with parameters in parentheses', async () => {
    vi.mocked(fetch).mockResolvedValue(makeSchemaResponse())

    renderWithContext(<PredicateBrowser />, 'neurosynth')

    await waitFor(() => {
      expect(screen.getByText('PeakReported(x, y, z, s)')).toBeInTheDocument()
      expect(screen.getByText('Study(id)')).toBeInTheDocument()
      expect(
        screen.getByText('agg_create_region(x, y, z)'),
      ).toBeInTheDocument()
      expect(screen.getByText('SelectedStudy(id)')).toBeInTheDocument()
    })
  })

  it('shows symbol name without parens when it has no parameters', async () => {
    vi.mocked(fetch).mockResolvedValue(
      makeSchemaResponse({
        relations: [{ name: 'MyRelation', type: 'relation', params: [] }],
        functions: [],
        probabilistic: [],
      }),
    )

    renderWithContext(<PredicateBrowser />, 'neurosynth')

    await waitFor(() => {
      expect(screen.getByText('MyRelation')).toBeInTheDocument()
    })
  })

  // -------------------------------------------------------------------------
  // Collapsible sections
  // -------------------------------------------------------------------------

  it('groups are open by default and can be collapsed', async () => {
    vi.mocked(fetch).mockResolvedValue(makeSchemaResponse())

    renderWithContext(<PredicateBrowser />, 'neurosynth')

    await waitFor(() => {
      expect(screen.getByText('PeakReported(x, y, z, s)')).toBeInTheDocument()
    })

    // The Relations group header button
    const relationsButton = screen.getByRole('button', { name: /Relations/i })
    expect(relationsButton).toHaveAttribute('aria-expanded', 'true')

    // Collapse the Relations group
    await userEvent.click(relationsButton)

    expect(relationsButton).toHaveAttribute('aria-expanded', 'false')
    expect(
      screen.queryByText('PeakReported(x, y, z, s)'),
    ).not.toBeInTheDocument()
  })

  it('re-expands a collapsed group when header is clicked again', async () => {
    vi.mocked(fetch).mockResolvedValue(makeSchemaResponse())

    renderWithContext(<PredicateBrowser />, 'neurosynth')

    await waitFor(() => {
      expect(screen.getByText('Relations')).toBeInTheDocument()
    })

    const relationsButton = screen.getByRole('button', { name: /Relations/i })
    // collapse
    await userEvent.click(relationsButton)
    expect(
      screen.queryByText('PeakReported(x, y, z, s)'),
    ).not.toBeInTheDocument()
    // expand
    await userEvent.click(relationsButton)
    expect(screen.getByText('PeakReported(x, y, z, s)')).toBeInTheDocument()
  })

  // -------------------------------------------------------------------------
  // Search / filter
  // -------------------------------------------------------------------------

  it('renders a search input', async () => {
    vi.mocked(fetch).mockResolvedValue(makeSchemaResponse())

    renderWithContext(<PredicateBrowser />, 'neurosynth')

    await waitFor(() => {
      expect(screen.getByRole('textbox')).toBeInTheDocument()
    })
  })

  it('filters symbols by search term (case-insensitive substring)', async () => {
    vi.mocked(fetch).mockResolvedValue(makeSchemaResponse())

    renderWithContext(<PredicateBrowser />, 'neurosynth')

    await waitFor(() => {
      expect(screen.getByText('PeakReported(x, y, z, s)')).toBeInTheDocument()
    })

    const searchInput = screen.getByRole('textbox')
    await userEvent.type(searchInput, 'peak')

    // PeakReported matches "peak" (case-insensitive)
    expect(screen.getByText('PeakReported(x, y, z, s)')).toBeInTheDocument()
    // Study should be hidden
    expect(screen.queryByText('Study(id)')).not.toBeInTheDocument()
    // agg_create_region should be hidden
    expect(
      screen.queryByText('agg_create_region(x, y, z)'),
    ).not.toBeInTheDocument()
  })

  it('shows no-match message when search has no results', async () => {
    vi.mocked(fetch).mockResolvedValue(makeSchemaResponse())

    renderWithContext(<PredicateBrowser />, 'neurosynth')

    await waitFor(() => {
      expect(screen.getByText('Relations')).toBeInTheDocument()
    })

    const searchInput = screen.getByRole('textbox')
    await userEvent.type(searchInput, 'zzznomatch')

    expect(
      screen.getByText('No predicates match "zzznomatch"'),
    ).toBeInTheDocument()
  })

  it('hides group entirely when all its symbols are filtered out', async () => {
    vi.mocked(fetch).mockResolvedValue(makeSchemaResponse())

    renderWithContext(<PredicateBrowser />, 'neurosynth')

    await waitFor(() => {
      expect(screen.getByText('Relations')).toBeInTheDocument()
    })

    const searchInput = screen.getByRole('textbox')
    // "agg" matches only the function symbol
    await userEvent.type(searchInput, 'agg')

    // Relations group should disappear
    expect(screen.queryByText('Relations')).not.toBeInTheDocument()
    // Functions group should appear
    expect(screen.getByText('Functions')).toBeInTheDocument()
    expect(
      screen.getByText('agg_create_region(x, y, z)'),
    ).toBeInTheDocument()
  })

  // -------------------------------------------------------------------------
  // Empty schema state
  // -------------------------------------------------------------------------

  it('shows empty message when schema has no symbols', async () => {
    vi.mocked(fetch).mockResolvedValue(
      makeSchemaResponse({
        relations: [],
        functions: [],
        probabilistic: [],
      }),
    )

    renderWithContext(<PredicateBrowser />, 'neurosynth')

    await waitFor(() => {
      expect(screen.getByText('No predicates available')).toBeInTheDocument()
    })
  })

  // -------------------------------------------------------------------------
  // Error state
  // -------------------------------------------------------------------------

  it('shows error when fetch fails', async () => {
    vi.mocked(fetch).mockRejectedValue(new Error('Network error'))

    renderWithContext(<PredicateBrowser />, 'neurosynth')

    await waitFor(() => {
      expect(screen.getByText('Network error')).toBeInTheDocument()
    })
  })

  it('shows error when response is not ok', async () => {
    vi.mocked(fetch).mockResolvedValue({
      ok: false,
      status: 503,
    } as Response)

    renderWithContext(<PredicateBrowser />, 'neurosynth')

    await waitFor(() => {
      expect(
        screen.getByText('Failed to fetch schema for neurosynth: 503'),
      ).toBeInTheDocument()
    })
  })

  // -------------------------------------------------------------------------
  // onSelect callback
  // -------------------------------------------------------------------------

  it('calls onSelect when a symbol is clicked', async () => {
    vi.mocked(fetch).mockResolvedValue(makeSchemaResponse())
    const onSelect = vi.fn()

    renderWithContext(<PredicateBrowser onSelect={onSelect} />, 'neurosynth')

    await waitFor(() => {
      expect(screen.getByText('PeakReported(x, y, z, s)')).toBeInTheDocument()
    })

    await userEvent.click(screen.getByText('PeakReported(x, y, z, s)'))

    expect(onSelect).toHaveBeenCalledTimes(1)
    expect(onSelect).toHaveBeenCalledWith({
      name: 'PeakReported',
      type: 'relation',
      params: ['x', 'y', 'z', 's'],
    })
  })

  it('calls onSelect when a symbol is activated with Enter key', async () => {
    vi.mocked(fetch).mockResolvedValue(makeSchemaResponse())
    const onSelect = vi.fn()

    renderWithContext(<PredicateBrowser onSelect={onSelect} />, 'neurosynth')

    await waitFor(() => {
      expect(screen.getByText('Study(id)')).toBeInTheDocument()
    })

    const studyItem = screen.getByText('Study(id)')
    studyItem.focus()
    await userEvent.keyboard('{Enter}')

    expect(onSelect).toHaveBeenCalledTimes(1)
    expect(onSelect).toHaveBeenCalledWith({
      name: 'Study',
      type: 'relation',
      params: ['id'],
    })
  })
})
