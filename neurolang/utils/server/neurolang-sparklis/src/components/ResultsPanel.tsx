/**
 * ResultsPanel.tsx
 *
 * The results display area that appears below the query area after a successful
 * query execution. Contains:
 *   - SymbolSelector: dropdown for selecting which result symbol to view
 *   - DataTable: renders rows and columns with sortable columns, pagination
 *     (50 rows/page), column type indicators
 *   - Empty state: "No results found" message for empty relations
 *   - CSV download button
 */
import React, { useState, useMemo, useCallback } from 'react'
import { useExecution } from '../context/useExecution'

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/** One result symbol's data as returned by the backend. */
export interface SymbolData {
  columns: string[]
  row_type: string[]
  size: number
  values: unknown[][]
  probabilistic: boolean
  last_parsed_symbol: boolean
}

/** All results from a query execution. */
export type QueryResultsMap = Record<string, SymbolData>

/** Sort direction. */
type SortDirection = 'asc' | 'desc'

/** Sort state. */
interface SortState {
  columnIndex: number
  direction: SortDirection
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const PAGE_SIZE = 50

/**
 * Parse a row_type string like "<class 'float'>" or "<class 'int'>" into
 * a short label: "float", "int", "str", "VBR", "figure", or "?"
 */
function parseColumnType(rowTypeStr: string): string {
  if (!rowTypeStr) return '?'
  const lower = rowTypeStr.toLowerCase()
  if (lower.includes('explicitvbroverlay')) return 'VBR'
  if (lower.includes('explicitvbr')) return 'VBR'
  if (lower.includes('figure')) return 'figure'
  if (lower.includes('float')) return 'float'
  if (lower.includes('int')) return 'int'
  if (lower.includes('str')) return 'str'
  if (lower.includes('bool')) return 'bool'
  // Fallback: take the last part after the last dot or single-quote
  const match = rowTypeStr.match(/'([^']+)'/)
  if (match) {
    const parts = match[1].split('.')
    return parts[parts.length - 1]
  }
  return '?'
}

/**
 * Sort a 2D array of values by the given column index.
 */
function sortRows(
  values: unknown[][],
  colIndex: number,
  direction: SortDirection,
): unknown[][] {
  const sorted = [...values].sort((a, b) => {
    const va = a[colIndex]
    const vb = b[colIndex]
    if (va === null || va === undefined) return 1
    if (vb === null || vb === undefined) return -1

    // Numeric comparison
    const na = Number(va)
    const nb = Number(vb)
    if (!isNaN(na) && !isNaN(nb)) {
      return direction === 'asc' ? na - nb : nb - na
    }

    // String comparison
    const sa = String(va).toLowerCase()
    const sb = String(vb).toLowerCase()
    if (sa < sb) return direction === 'asc' ? -1 : 1
    if (sa > sb) return direction === 'asc' ? 1 : -1
    return 0
  })
  return sorted
}

/**
 * Convert a SymbolData to a CSV string.
 */
function symbolDataToCSV(data: SymbolData): string {
  const header = data.columns.join(',')
  const rows = data.values.map((row) =>
    row
      .map((cell) => {
        const s = String(cell ?? '')
        // Quote cells that contain commas, quotes, or newlines
        if (s.includes(',') || s.includes('"') || s.includes('\n')) {
          return '"' + s.replace(/"/g, '""') + '"'
        }
        return s
      })
      .join(','),
  )
  return [header, ...rows].join('\n')
}

/**
 * Trigger a CSV file download in the browser.
 */
function downloadCSV(csvContent: string, filename: string): void {
  const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' })
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = filename
  link.style.display = 'none'
  document.body.appendChild(link)
  link.click()
  document.body.removeChild(link)
  URL.revokeObjectURL(url)
}

// ---------------------------------------------------------------------------
// SymbolSelector
// ---------------------------------------------------------------------------

interface SymbolSelectorProps {
  symbols: string[]
  selected: string
  onChange: (symbol: string) => void
}

function SymbolSelector({
  symbols,
  selected,
  onChange,
}: SymbolSelectorProps): React.ReactElement {
  return (
    <select
      className="symbol-selector"
      value={selected}
      onChange={(e) => onChange(e.target.value)}
      aria-label="Select result symbol"
      data-testid="symbol-selector"
    >
      {symbols.map((sym) => (
        <option key={sym} value={sym}>
          {sym}
        </option>
      ))}
    </select>
  )
}

// ---------------------------------------------------------------------------
// DataTable
// ---------------------------------------------------------------------------

interface DataTableProps {
  data: SymbolData
  symbolName: string
}

function DataTable({ data, symbolName }: DataTableProps): React.ReactElement {
  const [sortState, setSortState] = useState<SortState | null>(null)
  const [page, setPage] = useState(0)

  // Sort the rows if needed
  const sortedValues = useMemo(() => {
    if (!sortState) return data.values
    return sortRows(data.values, sortState.columnIndex, sortState.direction)
  }, [data.values, sortState])

  const totalPages = Math.ceil(sortedValues.length / PAGE_SIZE)
  const pageRows = sortedValues.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE)

  const handleHeaderClick = useCallback(
    (colIndex: number) => {
      setSortState((prev) => {
        if (prev?.columnIndex === colIndex) {
          // Toggle direction
          return {
            columnIndex: colIndex,
            direction: prev.direction === 'asc' ? 'desc' : 'asc',
          }
        }
        return { columnIndex: colIndex, direction: 'asc' }
      })
      // Reset to first page on sort change
      setPage(0)
    },
    [],
  )

  const handleCSVDownload = useCallback(() => {
    const csv = symbolDataToCSV(data)
    downloadCSV(csv, `${symbolName}.csv`)
  }, [data, symbolName])

  const showPagination = totalPages > 1

  // Map each row_type to a short label
  const columnTypes = data.row_type.map(parseColumnType)

  return (
    <div className="data-table-container">
      {/* Table header controls */}
      <div className="data-table-toolbar">
        <span className="data-table-info">
          {data.size} row{data.size !== 1 ? 's' : ''}
        </span>
        <button
          className="csv-download-btn"
          onClick={handleCSVDownload}
          aria-label={`Download ${symbolName} as CSV`}
          data-testid="csv-download-btn"
        >
          ↓ Download CSV
        </button>
      </div>

      {/* Scrollable table wrapper */}
      <div className="data-table-scroll">
        <table className="data-table" data-testid="data-table" role="table">
          <thead>
            <tr>
              {data.columns.map((col, i) => {
                const isSortedCol = sortState?.columnIndex === i
                const direction = isSortedCol ? sortState!.direction : null
                const ariaSort =
                  isSortedCol
                    ? direction === 'asc'
                      ? 'ascending'
                      : 'descending'
                    : 'none'
                return (
                  <th
                    key={col}
                    role="columnheader"
                    aria-sort={ariaSort}
                    className={`data-table-th${isSortedCol ? ' data-table-th--sorted' : ''}`}
                    onClick={() => handleHeaderClick(i)}
                  >
                    <span className="data-table-col-name">{col}</span>
                    <span
                      className="column-type-indicator"
                      data-testid="column-type-indicator"
                      title={data.row_type[i] ?? ''}
                    >
                      {columnTypes[i]}
                    </span>
                    {isSortedCol && (
                      <span className="sort-indicator" aria-hidden="true">
                        {direction === 'asc' ? ' ▲' : ' ▼'}
                      </span>
                    )}
                  </th>
                )
              })}
            </tr>
          </thead>
          <tbody>
            {pageRows.map((row, rowIdx) => (
              <tr key={rowIdx} className="data-table-row">
                {row.map((cell, cellIdx) => (
                  <td key={cellIdx} className="data-table-td">
                    {renderCell(cell, columnTypes[cellIdx])}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Pagination controls */}
      {showPagination && (
        <div className="pagination-controls" data-testid="pagination-controls">
          <button
            className="pagination-btn pagination-btn--prev"
            onClick={() => setPage((p) => Math.max(0, p - 1))}
            disabled={page === 0}
            aria-label="Previous page"
            data-testid="pagination-prev"
          >
            ← Prev
          </button>
          <span className="pagination-info">
            Page {page + 1} of {totalPages}
          </span>
          <button
            className="pagination-btn pagination-btn--next"
            onClick={() => setPage((p) => Math.min(totalPages - 1, p + 1))}
            disabled={page >= totalPages - 1}
            aria-label="Next page"
            data-testid="pagination-next"
          >
            Next →
          </button>
        </div>
      )}
    </div>
  )
}

/**
 * Render a cell value. VBR cells get a special badge; others are rendered
 * as plain text.
 */
function renderCell(cell: unknown, typeLabel: string): React.ReactNode {
  if (typeLabel === 'VBR') {
    if (cell === null || cell === undefined || cell === 'Empty Region') {
      return <span className="cell-vbr cell-vbr--empty">Empty Region</span>
    }
    return <span className="cell-vbr">VBR</span>
  }
  if (typeLabel === 'figure') {
    if (typeof cell === 'string' && cell.length > 0) {
      return (
        <img
          src={`data:image/png;base64,${cell}`}
          alt="figure"
          className="cell-figure"
        />
      )
    }
    return <span className="cell-figure--empty">—</span>
  }
  return <span>{String(cell ?? '')}</span>
}

// ---------------------------------------------------------------------------
// ResultsPanel
// ---------------------------------------------------------------------------

function ResultsPanel(): React.ReactElement | null {
  const { executionStatus, queryResults } = useExecution()

  // Parse queryResults into typed map
  const resultsMap = useMemo<QueryResultsMap | null>(() => {
    if (!queryResults) return null
    // Filter to only symbols that have the shape of SymbolData
    // (some symbols may be functions with different shapes)
    const map: QueryResultsMap = {}
    for (const [key, val] of Object.entries(queryResults)) {
      if (
        val !== null &&
        typeof val === 'object' &&
        'columns' in (val as object) &&
        'values' in (val as object)
      ) {
        map[key] = val as SymbolData
      }
    }
    return Object.keys(map).length > 0 ? map : null
  }, [queryResults])

  const symbolNames = useMemo(
    () => (resultsMap ? Object.keys(resultsMap) : []),
    [resultsMap],
  )

  // Default selected symbol: prefer last_parsed_symbol=true, else first
  const defaultSymbol = useMemo(() => {
    if (!resultsMap || symbolNames.length === 0) return ''
    const lastParsed = symbolNames.find(
      (s) => resultsMap[s]?.last_parsed_symbol,
    )
    return lastParsed ?? symbolNames[0]
  }, [resultsMap, symbolNames])

  const [selectedSymbol, setSelectedSymbol] = useState<string>(defaultSymbol)

  // When results change (new query), reset to default symbol
  const effectiveSymbol =
    selectedSymbol && resultsMap && resultsMap[selectedSymbol]
      ? selectedSymbol
      : defaultSymbol

  // Only show when execution is done
  if (executionStatus !== 'done') return null
  if (!resultsMap) {
    // Query succeeded but returned no table data (e.g., all functions)
    return (
      <div className="results-panel" data-testid="results-panel">
        <p className="no-results-message" data-testid="no-results-message">
          No results found
        </p>
      </div>
    )
  }

  const currentData = resultsMap[effectiveSymbol]

  return (
    <div className="results-panel" data-testid="results-panel">
      {/* Header: symbol selector + label */}
      <div className="results-panel-header">
        <span className="results-panel-title">Results</span>
        {symbolNames.length > 1 && (
          <SymbolSelector
            symbols={symbolNames}
            selected={effectiveSymbol}
            onChange={setSelectedSymbol}
          />
        )}
      </div>

      {/* Table or empty state */}
      {currentData && currentData.size === 0 ? (
        <p className="no-results-message" data-testid="no-results-message">
          No results found
        </p>
      ) : currentData ? (
        <DataTable data={currentData} symbolName={effectiveSymbol} />
      ) : null}
    </div>
  )
}

export default ResultsPanel
