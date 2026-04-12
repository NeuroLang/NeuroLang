/**
 * ResultsPanel.test.tsx
 *
 * Tests for the results display area:
 *   1. ResultsPanel: renders after successful query, shows symbol selector + table
 *   2. SymbolSelector: populated from results keys, switching shows correct data
 *   3. DataTable: renders rows/columns, sortable columns, pagination (50/page),
 *      empty state, column type indicators, CSV download
 *   4. VBR columns: "View in brain" button detection and overlay add behaviour
 */
import React from 'react'
import {
  render,
  screen,
  fireEvent,
  act,
  within,
} from '@testing-library/react'
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import {
  ExecutionProvider,
  ExecutionContextValue,
} from '../../context/ExecutionContext'
import { useExecution } from '../../context/useExecution'
import { EngineProvider } from '../../context/EngineContext'
import { QueryProvider } from '../../context/QueryContext'
import { BrainOverlayProvider } from '../../context/BrainOverlayContext'
import { useBrainOverlay } from '../../context/useBrainOverlay'
import ResultsPanel from '../ResultsPanel'
import { parseColumnType, isVbrColumnType } from '../../utils/columnTypeUtils'

// ---------------------------------------------------------------------------
// Mock WebSocket (same pattern as QueryExecution.test.tsx)
// ---------------------------------------------------------------------------

class MockWebSocket {
  static lastInstance: MockWebSocket | null = null

  url: string
  onopen: ((event: Event) => void) | null = null
  onmessage: ((event: MessageEvent) => void) | null = null
  onerror: ((event: Event) => void) | null = null
  onclose: ((event: CloseEvent) => void) | null = null

  sentMessages: string[] = []
  closed = false

  constructor(url: string) {
    this.url = url
    MockWebSocket.lastInstance = this
  }

  send(data: string) {
    this.sentMessages.push(data)
  }

  close() {
    this.closed = true
    const closeEvent = new CloseEvent('close', { wasClean: true })
    this.onclose?.(closeEvent)
  }

  triggerOpen() {
    this.onopen?.(new Event('open'))
  }

  triggerMessage(data: unknown) {
    const event = new MessageEvent('message', {
      data: JSON.stringify(data),
    })
    this.onmessage?.(event)
  }
}

// ---------------------------------------------------------------------------
// Helper: build a mock query results payload
// ---------------------------------------------------------------------------

/**
 * Returns a mock results payload with the given symbol data.
 *
 * symbol structure:
 *   { columns: string[], row_type: string[], size: number, values: unknown[][] }
 */
function makeDoneMessage(results: Record<string, unknown>) {
  return {
    status: 'ok',
    data: {
      uuid: 'test-uuid',
      cancelled: false,
      running: false,
      done: true,
      results,
    },
  }
}

/** A small dataset with 3 rows and 3 columns. */
const SMALL_RESULTS = {
  ans: {
    columns: ['x', 'y', 'z'],
    row_type: ['<class \'float\'>', '<class \'str\'>', '<class \'int\'>'],
    size: 3,
    values: [
      [1.0, 'apple', 10],
      [2.5, 'banana', 20],
      [3.7, 'cherry', 30],
    ],
    probabilistic: false,
    last_parsed_symbol: true,
  },
}

/** A dataset with two symbols. */
const MULTI_SYMBOL_RESULTS = {
  ans: {
    columns: ['x', 'y'],
    row_type: ['<class \'float\'>', '<class \'str\'>'],
    size: 2,
    values: [
      [1.0, 'apple'],
      [2.5, 'banana'],
    ],
    probabilistic: false,
    last_parsed_symbol: true,
  },
  Study: {
    columns: ['id', 'name'],
    row_type: ['<class \'int\'>', '<class \'str\'>'],
    size: 2,
    values: [
      [101, 'Study A'],
      [102, 'Study B'],
    ],
    probabilistic: false,
    last_parsed_symbol: false,
  },
}

/** A large dataset with 120 rows (for pagination testing). */
function makeLargeResults(numRows = 120) {
  const values = Array.from({ length: numRows }, (_, i) => [i, `item_${i}`])
  return {
    big: {
      columns: ['id', 'name'],
      row_type: ['<class \'int\'>', '<class \'str\'>'],
      size: numRows,
      values,
      probabilistic: false,
      last_parsed_symbol: true,
    },
  }
}

/** Empty results (zero rows). */
const EMPTY_RESULTS = {
  ans: {
    columns: ['x'],
    row_type: ['<class \'str\'>'],
    size: 0,
    values: [],
    probabilistic: false,
    last_parsed_symbol: true,
  },
}

// ---------------------------------------------------------------------------
// Provider wrapper
// ---------------------------------------------------------------------------

function TestProviders({ children }: { children: React.ReactNode }): React.ReactElement {
  return (
    <EngineProvider>
      <QueryProvider>
        <ExecutionProvider>
          <BrainOverlayProvider>{children}</BrainOverlayProvider>
        </ExecutionProvider>
      </QueryProvider>
    </EngineProvider>
  )
}

/** Exposes ExecutionContext value. */
function ExecutionCapture({
  onRender,
}: {
  onRender: (v: ExecutionContextValue) => void
}): React.ReactElement {
  const val = useExecution()
  onRender(val)
  return <></>
}

/** Helper: render ResultsPanel inside providers and drive WS to done state. */
function renderWithResults(results: Record<string, unknown>) {
  let captured: ExecutionContextValue | null = null

  render(
    <TestProviders>
      <ExecutionCapture onRender={(v) => { captured = v }} />
      <ResultsPanel />
    </TestProviders>,
  )

  act(() => {
    captured!.submitQuery('ans(x) :- Study(x).', 'neurosynth')
  })

  const ws = MockWebSocket.lastInstance!
  act(() => { ws.triggerOpen() })

  act(() => {
    ws.triggerMessage(makeDoneMessage(results))
  })

  return { captured }
}

// ---------------------------------------------------------------------------
// Setup / teardown
// ---------------------------------------------------------------------------

beforeEach(() => {
  MockWebSocket.lastInstance = null
  vi.stubGlobal('WebSocket', MockWebSocket)
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true }))
})

afterEach(() => {
  vi.restoreAllMocks()
})

// ---------------------------------------------------------------------------
// ResultsPanel – visibility
// ---------------------------------------------------------------------------

describe('ResultsPanel – visibility', () => {
  it('does NOT render when execution status is idle', () => {
    render(
      <TestProviders>
        <ResultsPanel />
      </TestProviders>,
    )
    expect(screen.queryByTestId('results-panel')).not.toBeInTheDocument()
  })

  it('shows skeleton loading state when execution status is running', () => {
    let captured: ExecutionContextValue | null = null

    render(
      <TestProviders>
        <ExecutionCapture onRender={(v) => { captured = v }} />
        <ResultsPanel />
      </TestProviders>,
    )

    act(() => {
      captured!.submitQuery('ans(x) :- Study(x).', 'neurosynth')
    })

    expect(screen.getByTestId('results-panel')).toBeInTheDocument()
    expect(screen.getByTestId('results-table-skeleton')).toBeInTheDocument()
  })

  it('renders results-panel after a successful query', () => {
    renderWithResults(SMALL_RESULTS)
    expect(screen.getByTestId('results-panel')).toBeInTheDocument()
  })

  it('does NOT render when execution status is error', () => {
    let captured: ExecutionContextValue | null = null

    render(
      <TestProviders>
        <ExecutionCapture onRender={(v) => { captured = v }} />
        <ResultsPanel />
      </TestProviders>,
    )

    act(() => { captured!.submitQuery('bad', 'neurosynth') })
    const ws = MockWebSocket.lastInstance!
    act(() => { ws.triggerOpen() })
    act(() => {
      ws.triggerMessage({
        status: 'ok',
        data: {
          uuid: 'err',
          cancelled: false,
          running: false,
          done: true,
          errorName: "<class 'Exception'>",
          message: 'Error occurred',
        },
      })
    })

    expect(screen.queryByTestId('results-panel')).not.toBeInTheDocument()
  })
})

// ---------------------------------------------------------------------------
// SymbolSelector
// ---------------------------------------------------------------------------

describe('SymbolSelector', () => {
  it('renders a symbol selector dropdown when multiple symbols exist', () => {
    renderWithResults(MULTI_SYMBOL_RESULTS)
    expect(screen.getByTestId('symbol-selector')).toBeInTheDocument()
  })

  it('populates symbol selector with all result symbol names', () => {
    renderWithResults(MULTI_SYMBOL_RESULTS)
    const selector = screen.getByTestId('symbol-selector')
    expect(selector).toHaveTextContent('ans')
    expect(selector).toHaveTextContent('Study')
  })

  it('shows data for the initially selected symbol', () => {
    renderWithResults(MULTI_SYMBOL_RESULTS)
    // Should show columns/data for one of the symbols
    expect(screen.getByTestId('data-table')).toBeInTheDocument()
  })

  it('switching symbol shows different columns', () => {
    renderWithResults(MULTI_SYMBOL_RESULTS)

    const selector = screen.getByTestId('symbol-selector')

    // Switch to Study symbol
    act(() => {
      fireEvent.change(selector, { target: { value: 'Study' } })
    })

    const table = screen.getByTestId('data-table')
    expect(within(table).getByText('id')).toBeInTheDocument()
    expect(within(table).getByText('name')).toBeInTheDocument()
  })

  it('switching back to first symbol shows original columns', () => {
    renderWithResults(MULTI_SYMBOL_RESULTS)

    const selector = screen.getByTestId('symbol-selector')

    // Switch to Study
    act(() => {
      fireEvent.change(selector, { target: { value: 'Study' } })
    })

    // Switch back to ans
    act(() => {
      fireEvent.change(selector, { target: { value: 'ans' } })
    })

    const table = screen.getByTestId('data-table')
    expect(within(table).getByText('x')).toBeInTheDocument()
    expect(within(table).getByText('y')).toBeInTheDocument()
  })
})

// ---------------------------------------------------------------------------
// DataTable – rendering
// ---------------------------------------------------------------------------

describe('DataTable – rendering', () => {
  it('renders a table with data-table testid', () => {
    renderWithResults(SMALL_RESULTS)
    expect(screen.getByTestId('data-table')).toBeInTheDocument()
  })

  it('renders column headers', () => {
    renderWithResults(SMALL_RESULTS)
    const table = screen.getByTestId('data-table')
    expect(within(table).getByText('x')).toBeInTheDocument()
    expect(within(table).getByText('y')).toBeInTheDocument()
    expect(within(table).getByText('z')).toBeInTheDocument()
  })

  it('renders data rows', () => {
    renderWithResults(SMALL_RESULTS)
    const table = screen.getByTestId('data-table')
    expect(within(table).getByText('apple')).toBeInTheDocument()
    expect(within(table).getByText('banana')).toBeInTheDocument()
    expect(within(table).getByText('cherry')).toBeInTheDocument()
  })

  it('renders all 3 data rows', () => {
    renderWithResults(SMALL_RESULTS)
    const table = screen.getByTestId('data-table')
    const rows = within(table).getAllByRole('row')
    // header row + 3 data rows = 4
    expect(rows.length).toBeGreaterThanOrEqual(4)
  })
})

// ---------------------------------------------------------------------------
// DataTable – empty state
// ---------------------------------------------------------------------------

describe('DataTable – empty state', () => {
  it('shows "No results found" when query returns empty relation', () => {
    renderWithResults(EMPTY_RESULTS)
    expect(screen.getByTestId('no-results-message')).toBeInTheDocument()
    expect(screen.getByTestId('no-results-message')).toHaveTextContent(
      /no results found/i,
    )
  })

  it('does NOT render a table when results are empty', () => {
    renderWithResults(EMPTY_RESULTS)
    expect(screen.queryByTestId('data-table')).not.toBeInTheDocument()
  })
})

// ---------------------------------------------------------------------------
// DataTable – sorting
// ---------------------------------------------------------------------------

describe('DataTable – sorting', () => {
  it('clicking a column header toggles sort', () => {
    renderWithResults(SMALL_RESULTS)
    const table = screen.getByTestId('data-table')

    // Find the 'x' column header and click it
    const xHeader = within(table).getByText('x')
    act(() => {
      fireEvent.click(xHeader)
    })

    // After sorting ascending, first row should have the smallest x (1.0)
    const rows = within(table).getAllByRole('row')
    // rows[0] is header; rows[1] is first data row
    expect(rows[1]).toHaveTextContent('1')
  })

  it('clicking column header twice sorts descending', () => {
    renderWithResults(SMALL_RESULTS)
    const table = screen.getByTestId('data-table')

    const xHeader = within(table).getByText('x')

    // First click: ascending
    act(() => {
      fireEvent.click(xHeader)
    })

    // Second click: descending
    act(() => {
      fireEvent.click(xHeader)
    })

    // After sorting descending, first row should have largest x (3.7)
    const rows = within(table).getAllByRole('row')
    expect(rows[1]).toHaveTextContent('3')
  })

  it('sorted column header shows a sort indicator', () => {
    renderWithResults(SMALL_RESULTS)
    const table = screen.getByTestId('data-table')

    const xHeader = within(table).getByText('x')
    act(() => {
      fireEvent.click(xHeader)
    })

    // The header cell should have some indicator (aria-sort or text symbol)
    const headerCell = xHeader.closest('[role="columnheader"]') ?? xHeader.closest('th')
    expect(
      headerCell?.getAttribute('aria-sort') !== null ||
      headerCell?.textContent?.includes('▲') ||
      headerCell?.textContent?.includes('▼') ||
      headerCell?.textContent?.includes('↑') ||
      headerCell?.textContent?.includes('↓')
    ).toBe(true)
  })
})

// ---------------------------------------------------------------------------
// DataTable – pagination
// ---------------------------------------------------------------------------

describe('DataTable – pagination', () => {
  it('shows only 50 rows per page for large results', () => {
    renderWithResults(makeLargeResults(120))
    const table = screen.getByTestId('data-table')
    const rows = within(table).getAllByRole('row')
    // header + 50 data rows = 51
    expect(rows.length).toBe(51)
  })

  it('renders pagination controls for results over 50 rows', () => {
    renderWithResults(makeLargeResults(120))
    expect(screen.getByTestId('pagination-controls')).toBeInTheDocument()
  })

  it('does NOT render pagination controls for results of 50 or fewer rows', () => {
    renderWithResults(SMALL_RESULTS)
    expect(screen.queryByTestId('pagination-controls')).not.toBeInTheDocument()
  })

  it('clicking next page shows the next 50 rows', () => {
    renderWithResults(makeLargeResults(120))

    const nextBtn = screen.getByTestId('pagination-next')
    act(() => {
      fireEvent.click(nextBtn)
    })

    const table = screen.getByTestId('data-table')
    const rows = within(table).getAllByRole('row')
    // Page 2: rows 50-99, so 50 rows + header = 51
    expect(rows.length).toBe(51)
    // First data row on page 2 should have id=50
    expect(rows[1]).toHaveTextContent('50')
  })

  it('clicking prev page goes back to first page', () => {
    renderWithResults(makeLargeResults(120))

    const nextBtn = screen.getByTestId('pagination-next')
    act(() => {
      fireEvent.click(nextBtn)
    })

    const prevBtn = screen.getByTestId('pagination-prev')
    act(() => {
      fireEvent.click(prevBtn)
    })

    const table = screen.getByTestId('data-table')
    const rows = within(table).getAllByRole('row')
    // Back to page 1: first data row should have id=0
    expect(rows[1]).toHaveTextContent('0')
  })

  it('prev button is disabled on first page', () => {
    renderWithResults(makeLargeResults(120))
    expect(screen.getByTestId('pagination-prev')).toBeDisabled()
  })

  it('next button is disabled on last page', () => {
    renderWithResults(makeLargeResults(120))

    const nextBtn = screen.getByTestId('pagination-next')
    // 120 rows / 50 per page = 3 pages; click next twice to reach page 3
    act(() => { fireEvent.click(nextBtn) })
    act(() => { fireEvent.click(nextBtn) })

    expect(screen.getByTestId('pagination-next')).toBeDisabled()
  })

  it('shows current page / total pages info', () => {
    renderWithResults(makeLargeResults(120))
    const pagination = screen.getByTestId('pagination-controls')
    // Should show something like "Page 1 of 3"
    expect(pagination).toHaveTextContent(/1/)
    expect(pagination).toHaveTextContent(/3/)
  })
})

// ---------------------------------------------------------------------------
// DataTable – column type indicators
// ---------------------------------------------------------------------------

describe('DataTable – column type indicators', () => {
  it('renders type indicator for each column', () => {
    renderWithResults(SMALL_RESULTS)
    // Columns: x (float), y (str), z (int)
    const typeIndicators = screen.getAllByTestId('column-type-indicator')
    expect(typeIndicators.length).toBe(3)
  })

  it('shows "float" type indicator for float columns', () => {
    renderWithResults(SMALL_RESULTS)
    const typeIndicators = screen.getAllByTestId('column-type-indicator')
    // First column 'x' has row_type float
    const texts = typeIndicators.map((el) => el.textContent?.toLowerCase() ?? '')
    expect(texts.some((t) => t.includes('float') || t.includes('f'))).toBe(true)
  })

  it('shows "str" or "string" type indicator for string columns', () => {
    renderWithResults(SMALL_RESULTS)
    const typeIndicators = screen.getAllByTestId('column-type-indicator')
    const texts = typeIndicators.map((el) => el.textContent?.toLowerCase() ?? '')
    expect(texts.some((t) => t.includes('str') || t.includes('s'))).toBe(true)
  })

  it('shows "int" type indicator for integer columns', () => {
    renderWithResults(SMALL_RESULTS)
    const typeIndicators = screen.getAllByTestId('column-type-indicator')
    const texts = typeIndicators.map((el) => el.textContent?.toLowerCase() ?? '')
    expect(texts.some((t) => t.includes('int') || t.includes('i'))).toBe(true)
  })
})

// ---------------------------------------------------------------------------
// CSV Download
// ---------------------------------------------------------------------------

describe('CSV Download', () => {
  it('renders a CSV download button', () => {
    renderWithResults(SMALL_RESULTS)
    expect(screen.getByTestId('csv-download-btn')).toBeInTheDocument()
  })

  it('clicking download button triggers a download', () => {
    renderWithResults(SMALL_RESULTS)

    // Mock URL.createObjectURL - jsdom doesn't provide it by default
    // Set up AFTER render to avoid interfering with component rendering
    const createObjectURL = vi.fn(() => 'blob:test-url')
    const revokeObjectURL = vi.fn()
    Object.defineProperty(globalThis.URL, 'createObjectURL', {
      value: createObjectURL,
      writable: true,
      configurable: true,
    })
    Object.defineProperty(globalThis.URL, 'revokeObjectURL', {
      value: revokeObjectURL,
      writable: true,
      configurable: true,
    })

    const appendChildSpy = vi.spyOn(document.body, 'appendChild').mockImplementation((el) => el)
    const removeChildSpy = vi.spyOn(document.body, 'removeChild').mockImplementation((el) => el)

    const downloadBtn = screen.getByTestId('csv-download-btn')
    act(() => {
      fireEvent.click(downloadBtn)
    })

    expect(createObjectURL).toHaveBeenCalled()

    appendChildSpy.mockRestore()
    removeChildSpy.mockRestore()
  })

  it('CSV content includes column headers - download anchor is triggered', () => {
    renderWithResults(SMALL_RESULTS)

    // Set up mocks AFTER render
    const createObjectURL = vi.fn(() => 'blob:test-url')
    const revokeObjectURL = vi.fn()
    Object.defineProperty(globalThis.URL, 'createObjectURL', {
      value: createObjectURL,
      writable: true,
      configurable: true,
    })
    Object.defineProperty(globalThis.URL, 'revokeObjectURL', {
      value: revokeObjectURL,
      writable: true,
      configurable: true,
    })

    const clickSpy = vi.fn()
    const mockAnchor = {
      href: '',
      download: '',
      click: clickSpy,
      style: { display: '' },
    }

    const originalCreateElement = document.createElement.bind(document)
    vi.spyOn(document, 'createElement').mockImplementation((tag: string) => {
      if (tag === 'a') return mockAnchor as unknown as HTMLElement
      return originalCreateElement(tag)
    })
    vi.spyOn(document.body, 'appendChild').mockImplementation((el) => el)
    vi.spyOn(document.body, 'removeChild').mockImplementation((el) => el)

    act(() => {
      fireEvent.click(screen.getByTestId('csv-download-btn'))
    })

    expect(createObjectURL).toHaveBeenCalled()
    expect(clickSpy).toHaveBeenCalled()
    expect(mockAnchor.download).toMatch(/\.csv$/)
  })
})

// ---------------------------------------------------------------------------
// parseColumnType helper
// ---------------------------------------------------------------------------

describe('parseColumnType helper', () => {
  it('returns "VBROverlay" for ExplicitVBROverlay row types', () => {
    expect(parseColumnType("<class 'neurolang.regions.ExplicitVBROverlay'>")).toBe('VBROverlay')
  })

  it('returns "VBR" for ExplicitVBR row types (non-overlay)', () => {
    expect(parseColumnType("<class 'neurolang.regions.ExplicitVBR'>")).toBe('VBR')
  })

  it('returns "float" for float row types', () => {
    expect(parseColumnType("<class 'float'>")).toBe('float')
  })

  it('returns "int" for int row types', () => {
    expect(parseColumnType("<class 'int'>")).toBe('int')
  })

  it('returns "str" for str row types', () => {
    expect(parseColumnType("<class 'str'>")).toBe('str')
  })

  it('returns "?" for empty string', () => {
    expect(parseColumnType('')).toBe('?')
  })
})

describe('isVbrColumnType helper', () => {
  it('returns true for "VBR"', () => {
    expect(isVbrColumnType('VBR')).toBe(true)
  })

  it('returns true for "VBROverlay"', () => {
    expect(isVbrColumnType('VBROverlay')).toBe(true)
  })

  it('returns false for "float"', () => {
    expect(isVbrColumnType('float')).toBe(false)
  })

  it('returns false for "str"', () => {
    expect(isVbrColumnType('str')).toBe(false)
  })
})

// ---------------------------------------------------------------------------
// VBR columns – "View in brain" button
// ---------------------------------------------------------------------------

/** A dataset with a VBR column containing base64 data. */
const VBR_RESULTS = {
  ans: {
    columns: ['region', 'vbr_data'],
    row_type: [
      "<class 'str'>",
      "<class 'neurolang.regions.ExplicitVBR'>",
    ],
    size: 2,
    values: [
      ['left_motor', 'dGVzdGJhc2U2NA=='],  // non-empty base64 (string format)
      ['right_motor', 'Empty Region'],       // empty VBR
    ],
    probabilistic: false,
    last_parsed_symbol: true,
  },
}

/**
 * A dataset with a VBR column using the object format returned by the real
 * backend serializeVBR: {min, max, q95, image, hash, center, idx}
 */
const VBR_OBJECT_RESULTS = {
  ans: {
    columns: ['region', 'vbr_data'],
    row_type: [
      "<class 'str'>",
      "<class 'neurolang.regions.ExplicitVBR'>",
    ],
    size: 2,
    values: [
      [
        'left_motor',
        {
          min: 1.0,
          max: 5.0,
          q95: 4.5,
          image: 'dGVzdEltYWdl',  // base64 NIfTI
          hash: 'abc123',
          center: [0, 0, 0],
          idx: 0,
        },
      ],
      ['right_motor', 'Empty Region'],  // empty VBR
    ],
    probabilistic: false,
    last_parsed_symbol: true,
  },
}

/** A dataset with a VBROverlay column (probability data). */
const VBR_OVERLAY_RESULTS = {
  ans: {
    columns: ['region', 'probability_map'],
    row_type: [
      "<class 'str'>",
      "<class 'neurolang.regions.ExplicitVBROverlay'>",
    ],
    size: 1,
    values: [
      ['meta_analysis', 'cHJvYmFiaWxpdHlkYXRh'],
    ],
    probabilistic: true,
    last_parsed_symbol: true,
  },
}

describe('DataTable – VBR columns', () => {
  it('shows "View in brain" button for rows with non-empty VBR data', () => {
    renderWithResults(VBR_RESULTS)
    // First row has non-empty VBR data
    const btns = screen.getAllByTestId('view-in-brain-btn')
    expect(btns.length).toBeGreaterThanOrEqual(1)
  })

  it('does NOT show "View in brain" button for empty VBR rows', () => {
    renderWithResults(VBR_RESULTS)
    // Second row is "Empty Region" – should not have a button
    // There is only 1 non-empty VBR row (first row), so exactly 1 button
    const btns = screen.getAllByTestId('view-in-brain-btn')
    expect(btns.length).toBe(1)
  })

  it('does NOT show "View in brain" button for non-VBR results', () => {
    renderWithResults({
      ans: {
        columns: ['x', 'y'],
        row_type: ["<class 'float'>", "<class 'str'>"],
        size: 1,
        values: [[1.0, 'apple']],
        probabilistic: false,
        last_parsed_symbol: true,
      },
    })
    expect(screen.queryByTestId('view-in-brain-btn')).not.toBeInTheDocument()
  })

  it('clicking "View in brain" adds overlay to BrainOverlayContext', () => {
    let capturedOverlayCtx: ReturnType<typeof useBrainOverlay> | null = null

    function OverlayCapture() {
      capturedOverlayCtx = useBrainOverlay()
      return null
    }

    let capturedExec: ExecutionContextValue | null = null

    render(
      <TestProviders>
        <OverlayCapture />
        <ExecutionCapture onRender={(v) => { capturedExec = v }} />
        <ResultsPanel />
      </TestProviders>,
    )

    act(() => { capturedExec!.submitQuery('ans(x) :- R(x).', 'destrieux') })
    const ws = MockWebSocket.lastInstance!
    act(() => { ws.triggerOpen() })
    act(() => { ws.triggerMessage(makeDoneMessage(VBR_RESULTS)) })

    expect(capturedOverlayCtx!.overlays).toHaveLength(0)

    const btn = screen.getByTestId('view-in-brain-btn')
    act(() => { fireEvent.click(btn) })

    expect(capturedOverlayCtx!.overlays).toHaveLength(1)
    expect(capturedOverlayCtx!.overlays[0].base64).toBe('dGVzdGJhc2U2NA==')
  })

  it('VBR column header shows "VBR" type indicator', () => {
    renderWithResults(VBR_RESULTS)
    const indicators = screen.getAllByTestId('column-type-indicator')
    const texts = indicators.map((el) => el.textContent ?? '')
    expect(texts.some((t) => t === 'VBR')).toBe(true)
  })

  it('VBROverlay column shows "VBROverlay" type indicator', () => {
    renderWithResults(VBR_OVERLAY_RESULTS)
    const indicators = screen.getAllByTestId('column-type-indicator')
    const texts = indicators.map((el) => el.textContent ?? '')
    expect(texts.some((t) => t === 'VBROverlay')).toBe(true)
  })

  it('VBROverlay overlay is marked as probabilistic', () => {
    let capturedOverlayCtx: ReturnType<typeof useBrainOverlay> | null = null

    function OverlayCapture() {
      capturedOverlayCtx = useBrainOverlay()
      return null
    }

    let capturedExec: ExecutionContextValue | null = null

    render(
      <TestProviders>
        <OverlayCapture />
        <ExecutionCapture onRender={(v) => { capturedExec = v }} />
        <ResultsPanel />
      </TestProviders>,
    )

    act(() => { capturedExec!.submitQuery('ans(x) :- R(x).', 'neurosynth') })
    const ws = MockWebSocket.lastInstance!
    act(() => { ws.triggerOpen() })
    act(() => { ws.triggerMessage(makeDoneMessage(VBR_OVERLAY_RESULTS)) })

    const btn = screen.getByTestId('view-in-brain-btn')
    act(() => { fireEvent.click(btn) })

    expect(capturedOverlayCtx!.overlays[0].isProbabilistic).toBe(true)
    expect(capturedOverlayCtx!.overlays[0].colormap).toBe('hot')
  })
})

// ---------------------------------------------------------------------------
// VBR object format (backend serializeVBR returns {min,max,q95,image,...})
// ---------------------------------------------------------------------------

describe('DataTable – VBR object format', () => {
  it('shows "View in brain" button for rows with VBR data in object format', () => {
    renderWithResults(VBR_OBJECT_RESULTS)
    const btns = screen.getAllByTestId('view-in-brain-btn')
    expect(btns.length).toBeGreaterThanOrEqual(1)
  })

  it('does NOT show "View in brain" button for empty VBR rows (object format)', () => {
    renderWithResults(VBR_OBJECT_RESULTS)
    // Second row is "Empty Region" – should have exactly 1 button (first row)
    const btns = screen.getAllByTestId('view-in-brain-btn')
    expect(btns.length).toBe(1)
  })

  it('clicking "View in brain" on object-format VBR row extracts .image as base64', () => {
    let capturedOverlayCtx: ReturnType<typeof useBrainOverlay> | null = null

    function OverlayCapture() {
      capturedOverlayCtx = useBrainOverlay()
      return null
    }

    let capturedExec: ExecutionContextValue | null = null

    render(
      <TestProviders>
        <OverlayCapture />
        <ExecutionCapture onRender={(v) => { capturedExec = v }} />
        <ResultsPanel />
      </TestProviders>,
    )

    act(() => { capturedExec!.submitQuery('ans(x) :- R(x).', 'destrieux') })
    const ws = MockWebSocket.lastInstance!
    act(() => { ws.triggerOpen() })
    act(() => { ws.triggerMessage(makeDoneMessage(VBR_OBJECT_RESULTS)) })

    expect(capturedOverlayCtx!.overlays).toHaveLength(0)

    const btn = screen.getByTestId('view-in-brain-btn')
    act(() => { fireEvent.click(btn) })

    expect(capturedOverlayCtx!.overlays).toHaveLength(1)
    // The .image field from the object should be extracted as base64
    expect(capturedOverlayCtx!.overlays[0].base64).toBe('dGVzdEltYWdl')
  })

  it('renders VBR object cells as "VBR" badge (not raw object)', () => {
    renderWithResults(VBR_OBJECT_RESULTS)
    // The cell should render a VBR badge, not a raw object string like [object Object]
    const table = screen.getByTestId('data-table')
    expect(table).not.toHaveTextContent('[object Object]')
    // Should have a VBR badge somewhere in the table
    const cells = table.querySelectorAll('.cell-vbr')
    expect(cells.length).toBeGreaterThan(0)
  })
})

