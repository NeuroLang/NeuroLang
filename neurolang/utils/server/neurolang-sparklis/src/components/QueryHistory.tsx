/**
 * QueryHistory.tsx
 *
 * A collapsible sidebar panel that displays executed query history.
 *
 * Features:
 *   - Lists up to 50 most-recent queries (most recent first)
 *   - Each entry shows a truncated query preview and relative/formatted timestamp
 *   - Clicking an entry loads the query into the code editor + visual builder
 *   - "Clear history" button removes all entries
 *   - Panel collapses/expands via a header button
 *   - State persists across page reloads (managed by QueryHistoryContext)
 */
import React, { useCallback, useState } from 'react'
import { useQueryHistory } from '../context/useQueryHistory'
import { useQuery } from '../context/useQuery'
import { type HistoryEntry } from '../context/QueryHistoryContext'

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/** Maximum characters to show in the query preview before truncation. */
const PREVIEW_MAX_LENGTH = 60

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Truncate a query string to PREVIEW_MAX_LENGTH, appending '…' if needed. */
function truncateQuery(query: string): string {
  const singleLine = query.replace(/\s+/g, ' ').trim()
  if (singleLine.length <= PREVIEW_MAX_LENGTH) return singleLine
  return singleLine.slice(0, PREVIEW_MAX_LENGTH) + '…'
}

/** Format an ISO-8601 timestamp into a human-readable label. */
function formatTimestamp(iso: string): string {
  try {
    const date = new Date(iso)
    const now = new Date()
    const diffMs = now.getTime() - date.getTime()
    const diffMin = Math.floor(diffMs / 60_000)
    const diffH = Math.floor(diffMin / 60)
    const diffD = Math.floor(diffH / 24)

    if (diffMin < 1) return 'just now'
    if (diffMin < 60) return `${diffMin}m ago`
    if (diffH < 24) return `${diffH}h ago`
    if (diffD < 7) return `${diffD}d ago`

    // Older than a week: show date
    return date.toLocaleDateString(undefined, {
      month: 'short',
      day: 'numeric',
    })
  } catch {
    return ''
  }
}

// ---------------------------------------------------------------------------
// HistoryItem sub-component
// ---------------------------------------------------------------------------

interface HistoryItemProps {
  entry: HistoryEntry
  onLoad: (query: string) => void
}

function HistoryItem({ entry, onLoad }: HistoryItemProps): React.ReactElement {
  const preview = truncateQuery(entry.query)
  const timeLabel = formatTimestamp(entry.timestamp)

  return (
    <li className="history-item">
      <button
        className="history-item-btn"
        onClick={() => onLoad(entry.query)}
        aria-label={`Load query: ${preview}`}
        title={entry.query}
      >
        <span className="history-item-preview">{preview}</span>
        <span className="history-item-meta">
          <span className="history-item-engine">{entry.engine}</span>
          {timeLabel && (
            <span className="history-item-time">{timeLabel}</span>
          )}
          {entry.resultSummary && (
            <span className="history-item-summary">{entry.resultSummary}</span>
          )}
        </span>
      </button>
    </li>
  )
}

// ---------------------------------------------------------------------------
// QueryHistory main component
// ---------------------------------------------------------------------------

function QueryHistory(): React.ReactElement {
  const { entries, clearHistory } = useQueryHistory()
  const { model, refresh, setDatalogText } = useQuery()
  const [panelOpen, setPanelOpen] = useState(true)

  const handleLoadQuery = useCallback(
    (query: string) => {
      // Reset the visual builder
      model.reset()
      refresh()
      // Update the code editor with the query text (triggers bidirectional sync)
      setDatalogText(query)
    },
    [model, refresh, setDatalogText],
  )

  const handleClearHistory = useCallback(() => {
    clearHistory()
  }, [clearHistory])

  return (
    <div className="query-history">
      {/* Panel header / toggle */}
      <div className="query-history-header-row">
        <button
          className="query-history-header"
          onClick={() => setPanelOpen((prev) => !prev)}
          aria-expanded={panelOpen}
          aria-label={
            panelOpen ? 'Collapse history panel' : 'Expand history panel'
          }
        >
          <span className="query-history-title">History</span>
          <span
            className={`query-history-chevron ${panelOpen ? 'query-history-chevron--open' : ''}`}
          >
            ▶
          </span>
        </button>
        {panelOpen && entries.length > 0 && (
          <button
            className="query-history-clear-btn"
            onClick={handleClearHistory}
            aria-label="Clear history"
            title="Clear all history"
          >
            Clear history
          </button>
        )}
      </div>

      {panelOpen && (
        <div className="query-history-body">
          {entries.length === 0 ? (
            <p className="query-history-empty">No history yet</p>
          ) : (
            <ul className="history-list">
              {entries.map((entry, index) => (
                <HistoryItem
                  key={`${entry.timestamp}-${index}`}
                  entry={entry}
                  onLoad={handleLoadQuery}
                />
              ))}
            </ul>
          )}
        </div>
      )}
    </div>
  )
}

export default QueryHistory
